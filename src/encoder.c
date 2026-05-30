#include "model.h"
#include <string.h>

/*
 * Encoder layer forward (Pre-LN variant):
 *   x1  = LN1(xi)
 *   r1  = xi + self_attn(x1, x1)
 *   x2  = LN2(r1)
 *   xo  = r1 + FFN(x2)
 *
 * Using Post-LN (original paper):
 *   sa_out = self_attn(xi, xi)
 *   r1  = LN1(xi + sa_out)
 *   ff_out = FFN(r1)
 *   xo  = LN2(r1 + ff_out)
 */
void el_fwd(const Mat *x, const EL *w, EC *c, const Cfg *cfg, int causal) {
    int S = x->r, D = cfg->D;

    /* save input */
    memcpy(c->xi.d, x->d, (size_t)S*D*sizeof(float));

    /* self-attention (causal=0 for encoder, causal=1 for causal LM) */
    Mat xi_s = {c->xi.d, S, D};
    attn_fwd(&xi_s, &xi_s, &w->sa, cfg->H, causal, &c->sa);

    /* residual + LN1: r1 = xi + sa.ao,  x1 = LN1(r1) */
    {   Mat ao_s = {c->sa.ao.d, S, D};
        for (int i = 0; i < S*D; i++) c->r1.d[i] = c->xi.d[i] + ao_s.d[i]; }
    {   Mat r1_s = {c->r1.d, S, D};
        Mat x1_s = {c->x1.d, S, D};
        ln_fwd(&r1_s, &w->ln1, &x1_s, c->mn1, c->vr1, cfg->eps); }

    /* FFN */
    {   Mat x1_s = {c->x1.d, S, D};
        Mat fh_s = {c->fh.d, S, cfg->F};
        Mat fo_s = {c->fo.d, S, D};
        ffn_fwd(&x1_s, &w->ff, &fh_s, &fo_s); }

    /* residual + LN2: r2 = r1 + FFN_out,  xo = LN2(r2) */
    {   for (int i = 0; i < S*D; i++) c->r2.d[i] = c->r1.d[i] + c->fo.d[i]; }
    {   Mat r2_s = {c->r2.d, S, D};
        Mat xo_s = {c->xo.d, S, D};
        ln_fwd(&r2_s, &w->ln2, &xo_s, c->mn2, c->vr2, cfg->eps); }
}

/*
 * Encoder layer backward.
 * dy   : gradient of xo  (S×D)
 * dx   : accumulated gradient of the layer input xi (S×D)
 * dw   : accumulated weight gradients
 */
void el_bwd(const EL *w, EC *c, const Cfg *cfg,
            const Mat *dy, EL *dw, Mat *dx, int causal) {
    int S = c->sa.S, D = cfg->D;

    /* -- LN2 backward -- */
    /* grad flows to r2 */
    Mat dr2 = mat_new(S, D);
    {   Mat r2_s = {c->r2.d, S, D};
        ln_bwd(&r2_s, &w->ln2, c->mn2, c->vr2, dy, &dr2, (LN*)&dw->ln2, cfg->eps);
    }

    /* residual: dr1 += dr2,  dfo = dr2 */
    Mat dr1 = mat_new(S, D);
    Mat dfo = mat_new(S, D);
    for (int i = 0; i < S*D; i++) { dr1.d[i] = dr2.d[i]; dfo.d[i] = dr2.d[i]; }
    mat_del(&dr2);

    /* -- FFN backward -- */
    {   Mat x1_s  = {c->x1.d, S, D};
        Mat fh_s  = {c->fh.d, S, cfg->F};
        Mat dx1   = mat_new(S, D);
        ffn_bwd(&x1_s, &w->ff, &fh_s, &dfo, &dx1, (FW*)&dw->ff);
        mat_del(&dfo);

        /* -- LN1 backward -- */
        /* gradient from FFN goes through LN1 to r1 */
        Mat r1_s = {c->r1.d, S, D};
        ln_bwd(&r1_s, &w->ln1, c->mn1, c->vr1, &dx1, &dr1, (LN*)&dw->ln1, cfg->eps);
        mat_del(&dx1);
    }

    /* residual: dx += dr1,  d_sa_ao = dr1 */
    for (int i = 0; i < S*D; i++) dx->d[i] += dr1.d[i];

    /* -- Self-attention backward -- */
    Mat d_ao = mat_new(S, D);
    for (int i = 0; i < S*D; i++) d_ao.d[i] = dr1.d[i];
    mat_del(&dr1);

    /* dx (query src) and dkv (same as dx for self-attn) both go to dx */
    attn_bwd(&w->sa, cfg->H, causal, &c->sa, &d_ao, dx, dx, (AW*)&dw->sa);
    mat_del(&d_ao);
}
