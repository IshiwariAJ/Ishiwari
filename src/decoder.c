#include "model.h"
#include <string.h>

/*
 * Decoder layer forward (Post-LN, original paper):
 *   sa_out = masked_self_attn(xi, xi)
 *   x1  = LN1(xi + sa_out)
 *   ca_out = cross_attn(x1, enc_out)
 *   x2  = LN2(x1 + ca_out)
 *   ff_out = FFN(x2)
 *   xo  = LN3(x2 + ff_out)
 */
void dl_fwd(const Mat *x, const Mat *enc,
            const DL *w, DC *c, const Cfg *cfg) {
    int S  = x->r;
    int KS = enc->r;
    int D  = cfg->D;

    memcpy(c->xi.d, x->d,   (size_t)S *D*sizeof(float));

    /* masked self-attention */
    Mat xi_s = {c->xi.d, S, D};
    attn_fwd(&xi_s, &xi_s, &w->sa, cfg->H, 1, &c->sa);

    /* residual + LN1 */
    {   Mat ao_s = {c->sa.ao.d, S, D};
        for (int i = 0; i < S*D; i++) c->r1.d[i] = c->xi.d[i] + ao_s.d[i]; }
    {   Mat r1_s = {c->r1.d, S, D};
        Mat x1_s = {c->x1.d, S, D};
        ln_fwd(&r1_s, &w->ln1, &x1_s, c->mn1, c->vr1, cfg->eps); }

    /* cross-attention: Q from x1, K/V from enc */
    {   Mat x1_s = {c->x1.d, S, D};
        Mat enc_s = {(float*)enc->d, KS, D};
        attn_fwd(&x1_s, &enc_s, &w->ca, cfg->H, 0, &c->ca); }

    /* residual + LN2 */
    {   Mat ca_ao = {c->ca.ao.d, S, D};
        for (int i = 0; i < S*D; i++) c->r2.d[i] = c->x1.d[i] + ca_ao.d[i]; }
    {   Mat r2_s = {c->r2.d, S, D};
        Mat x2_s = {c->x2.d, S, D};
        ln_fwd(&r2_s, &w->ln2, &x2_s, c->mn2, c->vr2, cfg->eps); }

    /* FFN */
    {   Mat x2_s = {c->x2.d, S, D};
        Mat fh_s = {c->fh.d, S, cfg->F};
        Mat fo_s = {c->fo.d, S, D};
        ffn_fwd(&x2_s, &w->ff, &fh_s, &fo_s); }

    /* residual + LN3 */
    {   for (int i = 0; i < S*D; i++) c->r3.d[i] = c->x2.d[i] + c->fo.d[i]; }
    {   Mat r3_s = {c->r3.d, S, D};
        Mat xo_s = {c->xo.d, S, D};
        ln_fwd(&r3_s, &w->ln3, &xo_s, c->mn3, c->vr3, cfg->eps); }
}

/*
 * Decoder layer backward.
 * dy    : gradient of xo  (S×D)
 * dx    : accumulated gradient of xi  (S×D)
 * d_enc : accumulated gradient of enc (KS×D)
 */
void dl_bwd(const DL *w, DC *c, const Cfg *cfg,
            const Mat *dy, DL *dw, Mat *dx, Mat *d_enc) {
    int S  = c->sa.S;
    int KS = c->ca.KS;
    int D  = cfg->D;

    /* -- LN3 backward → dr3 -- */
    Mat dr3 = mat_new(S, D);
    {   Mat r3_s = {c->r3.d, S, D};
        ln_bwd(&r3_s, &w->ln3, c->mn3, c->vr3, dy, &dr3, (LN*)&dw->ln3, cfg->eps); }

    /* residual: dr2 += dr3,  dfo = dr3 */
    Mat dr2 = mat_new(S, D);
    Mat dfo = mat_new(S, D);
    for (int i = 0; i < S*D; i++) { dr2.d[i] = dr3.d[i]; dfo.d[i] = dr3.d[i]; }
    mat_del(&dr3);

    /* -- FFN backward -- */
    {   Mat x2_s = {c->x2.d, S, D};
        Mat fh_s = {c->fh.d, S, cfg->F};
        Mat dx2  = mat_new(S, D);
        ffn_bwd(&x2_s, &w->ff, &fh_s, &dfo, &dx2, (FW*)&dw->ff);
        mat_del(&dfo);

        /* -- LN2 backward -- */
        Mat r2_s = {c->r2.d, S, D};
        ln_bwd(&r2_s, &w->ln2, c->mn2, c->vr2, &dx2, &dr2, (LN*)&dw->ln2, cfg->eps);
        mat_del(&dx2);
    }

    /* residual: dx1 += dr2,  d_ca_ao = dr2 */
    Mat dx1 = mat_new(S, D);
    Mat d_ca_ao = mat_new(S, D);
    for (int i = 0; i < S*D; i++) { dx1.d[i] = dr2.d[i]; d_ca_ao.d[i] = dr2.d[i]; }
    mat_del(&dr2);

    /* -- Cross-attention backward -- */
    /* Q from x1, K/V from enc */
    Mat d_enc_tmp = mat_new(KS, D);
    attn_bwd(&w->ca, cfg->H, 0, &c->ca, &d_ca_ao, &dx1, &d_enc_tmp, (AW*)&dw->ca);
    mat_del(&d_ca_ao);
    /* accumulate into d_enc */
    for (int i = 0; i < KS*D; i++) d_enc->d[i] += d_enc_tmp.d[i];
    mat_del(&d_enc_tmp);

    /* -- LN1 backward -- */
    Mat dr1 = mat_new(S, D);
    {   Mat r1_s = {c->r1.d, S, D};
        ln_bwd(&r1_s, &w->ln1, c->mn1, c->vr1, &dx1, &dr1, (LN*)&dw->ln1, cfg->eps);
        mat_del(&dx1); }

    /* residual: dx += dr1,  d_sa_ao = dr1 */
    for (int i = 0; i < S*D; i++) dx->d[i] += dr1.d[i];

    /* -- Masked self-attention backward -- */
    Mat d_sa_ao = mat_new(S, D);
    for (int i = 0; i < S*D; i++) d_sa_ao.d[i] = dr1.d[i];
    mat_del(&dr1);

    attn_bwd(&w->sa, cfg->H, 1, &c->sa, &d_sa_ao, dx, dx, (AW*)&dw->sa);
    mat_del(&d_sa_ao);
}
