#include "model.h"
#include "event.h"
#include "adapters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* --- helpers--- */

static int n_pass = 0, n_fail = 0;

static void check(const char *name, int ok) {
    if (ok) { printf("  PASS  %s\n", name); n_pass++; }
    else     { printf("  FAIL  %s\n", name); n_fail++; }
}

static int fclose_to(float a, float b, float tol) {
    return fabsf(a - b) < tol;
}

/* --- matrix--- */

static void test_mm(void) {
    printf("[mm]\n");
    /* A (2x3) @ B (3x2) = C (2x2)
     * A = [[1,2,3],[4,5,6]]
     * B = [[7,8],[9,10],[11,12]]
     * C = [[58,64],[139,154]]
     */
    Mat A = mat_new(2, 3);
    Mat B = mat_new(3, 2);
    Mat C = mat_new(2, 2);
    float av[] = {1,2,3,4,5,6};
    float bv[] = {7,8,9,10,11,12};
    memcpy(A.d, av, sizeof(av));
    memcpy(B.d, bv, sizeof(bv));
    mm(&A, &B, &C);
    check("C[0,0]==58",  fclose_to(C.d[0], 58.f,  1e-4f));
    check("C[0,1]==64",  fclose_to(C.d[1], 64.f,  1e-4f));
    check("C[1,0]==139", fclose_to(C.d[2], 139.f, 1e-4f));
    check("C[1,1]==154", fclose_to(C.d[3], 154.f, 1e-4f));
    mat_del(&A); mat_del(&B); mat_del(&C);
}

static void test_mm_at(void) {
    printf("[mm_at_add]\n");
    /* A^T (3x2) @ B (2x2) += C (3x2)
     * A = [[1,2,3],[4,5,6]]  A^T = [[1,4],[2,5],[3,6]]
     * B = [[1,0],[0,1]]
     * C += A^T @ B = [[1,4],[2,5],[3,6]]
     */
    Mat A = mat_new(2, 3);
    Mat B = mat_new(2, 2);
    Mat C = mat_new(3, 2);
    float av[] = {1,2,3,4,5,6};
    float bv[] = {1,0,0,1};
    memcpy(A.d, av, sizeof(av));
    memcpy(B.d, bv, sizeof(bv));
    mat_zero(&C);
    mm_at_add(&A, &B, &C);
    check("C[0,0]==1", fclose_to(C.d[0], 1.f, 1e-4f));
    check("C[0,1]==4", fclose_to(C.d[1], 4.f, 1e-4f));
    check("C[2,0]==3", fclose_to(C.d[4], 3.f, 1e-4f));
    check("C[2,1]==6", fclose_to(C.d[5], 6.f, 1e-4f));
    mat_del(&A); mat_del(&B); mat_del(&C);
}

/* --- GELU--- */

static void test_gelu(void) {
    printf("[gelu_fwd]\n");
    /* GELU(0) = 0, GELU(x) > 0 for x > 0, approx x for large x */
    Mat x = mat_new(1, 4);
    Mat y = mat_new(1, 4);
    x.d[0] = 0.f; x.d[1] = 1.f; x.d[2] = -1.f; x.d[3] = 3.f;
    gelu_fwd(&x, &y);
    check("GELU(0)==0",       fclose_to(y.d[0], 0.f,      1e-5f));
    check("GELU(1)~0.841",    fclose_to(y.d[1], 0.8413f,  2e-3f));
    check("GELU(-1)~-0.159",  fclose_to(y.d[2], -0.1587f, 2e-3f));
    check("GELU(3)~2.996",    fclose_to(y.d[3], 2.9960f,  2e-3f));
    mat_del(&x); mat_del(&y);
}

static void test_gelu_bwd(void) {
    printf("[gelu_bwd]\n");
    /* finite-difference check: d/dx GELU(x) at x=1 */
    float eps = 1e-4f;
    Mat x1 = mat_new(1,1); Mat y1 = mat_new(1,1);
    Mat x2 = mat_new(1,1); Mat y2 = mat_new(1,1);
    Mat dx = mat_new(1,1); Mat dy = mat_new(1,1);
    x1.d[0] = 1.f - eps; x2.d[0] = 1.f + eps;
    gelu_fwd(&x1, &y1); gelu_fwd(&x2, &y2);
    float fd_grad = (y2.d[0] - y1.d[0]) / (2*eps);

    Mat xc = mat_new(1,1);
    xc.d[0] = 1.f; dy.d[0] = 1.f; dx.d[0] = 0.f;
    gelu_bwd(&xc, &dy, &dx);
    check("gelu_bwd finite-diff", fclose_to(dx.d[0], fd_grad, 1e-3f));
    mat_del(&x1);mat_del(&y1);mat_del(&x2);mat_del(&y2);
    mat_del(&xc);mat_del(&dx);mat_del(&dy);
}

/* --- softmax--- */

static void test_softmax(void) {
    printf("[softmax_fwd]\n");
    Mat x = mat_new(2, 3);
    Mat y = mat_new(2, 3);
    x.d[0]=1.f; x.d[1]=2.f; x.d[2]=3.f;
    x.d[3]=0.f; x.d[4]=0.f; x.d[5]=0.f;
    softmax_fwd(&x, &y);

    float s0 = y.d[0]+y.d[1]+y.d[2];
    float s1 = y.d[3]+y.d[4]+y.d[5];
    check("row0 sum==1",    fclose_to(s0, 1.f, 1e-5f));
    check("row1 sum==1",    fclose_to(s1, 1.f, 1e-5f));
    check("row0 monotone",  y.d[0] < y.d[1] && y.d[1] < y.d[2]);
    check("row1 uniform",   fclose_to(y.d[3], 1.f/3.f, 1e-5f));
    mat_del(&x); mat_del(&y);
}

/* --- LayerNorm--- */

static void test_layernorm(void) {
    printf("[ln_fwd]\n");
    int D = 8;
    LN ln = ln_new(D);   /* gamma=1, beta=0 */
    Mat x = mat_new(2, D);
    Mat y = mat_new(2, D);
    float mn[2], vr[2];

    /* row 0: constant → should produce zeros */
    for (int j = 0; j < D; j++) x.d[j] = 3.f;
    /* row 1: 0,1,2,...,7 */
    for (int j = 0; j < D; j++) x.d[D+j] = (float)j;
    ln_fwd(&x, &ln, &y, mn, vr, 1e-5f);

    /* row 0: all zeros (constant input, gamma=1, beta=0) */
    int r0_zero = 1;
    for (int j = 0; j < D; j++) if (!fclose_to(y.d[j], 0.f, 1e-5f)) r0_zero = 0;
    check("const input → zero output", r0_zero);

    /* row 1: check mean≈0, variance≈1 */
    float mean1 = 0.f, var1 = 0.f;
    for (int j = 0; j < D; j++) mean1 += y.d[D+j];
    mean1 /= D;
    for (int j = 0; j < D; j++) { float d=y.d[D+j]-mean1; var1+=d*d; }
    var1 /= D;
    check("output mean≈0",  fclose_to(mean1, 0.f, 1e-5f));
    check("output var≈1",   fclose_to(var1, 1.f, 1e-4f));

    ln_del(&ln);
    mat_del(&x); mat_del(&y);
}

static void test_layernorm_bwd(void) {
    printf("[ln_bwd — finite diff]\n");
    int D = 4;
    LN ln = ln_new(D);
    /* set non-trivial gamma */
    ln.scale.w[0]=1.5f; ln.scale.w[1]=0.5f;
    ln.scale.w[2]=2.0f; ln.scale.w[3]=1.0f;

    Mat x = mat_new(1, D);
    x.d[0]=0.5f; x.d[1]=-1.f; x.d[2]=2.f; x.d[3]=0.f;

    /* forward */
    Mat y = mat_new(1, D);
    float mn, vr;
    ln_fwd(&x, &ln, &y, &mn, &vr, 1e-5f);

    /* upstream gradient = all ones */
    Mat dy = mat_new(1, D);
    for (int j = 0; j < D; j++) dy.d[j] = 1.f;

    /* analytical backward */
    Mat dx_anal = mat_new(1, D);
    LN dln; dln.scale = p_new(D); dln.shift = p_new(D);
    memset(dln.scale.w, 0, D*sizeof(float));
    memset(dln.shift.w, 0, D*sizeof(float));
    ln_bwd(&x, &ln, &mn, &vr, &dy, &dx_anal, &dln, 1e-5f);

    /* finite-difference for dx[0] */
    float eps = 1e-4f;
    Mat xp = mat_new(1,D); Mat xm = mat_new(1,D);
    Mat yp = mat_new(1,D); Mat ym = mat_new(1,D);
    memcpy(xp.d, x.d, D*sizeof(float)); xp.d[0] += eps;
    memcpy(xm.d, x.d, D*sizeof(float)); xm.d[0] -= eps;
    float mnp,vrp,mnm,vrm;
    ln_fwd(&xp,&ln,&yp,&mnp,&vrp,1e-5f);
    ln_fwd(&xm,&ln,&ym,&mnm,&vrm,1e-5f);
    /* loss = sum(y) with dy=1 */
    float fd = 0.f;
    for (int j = 0; j < D; j++) fd += yp.d[j] - ym.d[j];
    fd /= (2*eps);
    check("dx[0] matches finite-diff", fclose_to(dx_anal.d[0], fd, 1e-3f));

    ln_del(&ln); p_del(&dln.scale); p_del(&dln.shift);
    mat_del(&x);mat_del(&y);mat_del(&dy);mat_del(&dx_anal);
    mat_del(&xp);mat_del(&xm);mat_del(&yp);mat_del(&ym);
}

/* --- EventEmbed--- */

static void test_event_embed(void) {
    printf("[event_embed_fwd]\n");
    srand(42);
    int D = 8, V = 11, T = 16;
    EventEmbed *e = event_embed_new(D, V, T);
    event_embed_init(e);

    EventSeq *s = event_seq_new(8);
    int ids[] = {1, 3, 5};
    event_append_text(s, ids, 3, 0);
    float vals[] = {0.2f, 0.8f};
    event_append_scalar(s, vals, 2, MOD_TEMPERATURE, 0, 3);

    Mat out = mat_new(s->n, D);
    event_embed_fwd(s, e, &out);

    check("output rows == n", out.r == s->n);
    /* text row 0: should equal tok_emb[1] + mod_emb[TEXT] + chan_emb[0] + pos[0] */
    float expected0 = 0.f;
    for (int d = 0; d < D; d++)
        expected0 += e->tok_emb.w[1*D+d] + e->mod_emb.w[0] /* MOD_TEXT=0, d=0 just for sign check */;
    /* just verify it's non-zero and finite */
    float sum_text = 0.f;
    for (int d = 0; d < D; d++) sum_text += fabsf(out.d[d]);
    check("text row non-zero", sum_text > 0.f);

    float sum_scalar = 0.f;
    for (int d = 0; d < D; d++) sum_scalar += fabsf(out.d[3*D+d]);
    check("scalar row non-zero", sum_scalar > 0.f);

    /* verify scalar rows differ (different values → different embeddings) */
    float diff = 0.f;
    for (int d = 0; d < D; d++) diff += fabsf(out.d[3*D+d] - out.d[4*D+d]);
    check("scalar rows differ for different values", diff > 0.f);

    mat_del(&out);
    event_seq_del(s);
    event_embed_del(e);
}

/* --- Shape / integration tests--- */

static void test_attn_shape(void) {
    printf("[attn_fwd shape]\n");
    int S=5, D=16, H=4;
    AW w = aw_new(D);
    /* init weights to small random to avoid NaN */
    for (int i = 0; i < D*D; i++) {
        w.Wq.w[i]=w.Wk.w[i]=w.Wv.w[i]=w.Wo.w[i]= 0.01f*(i%7-3);
    }
    AC *c = ac_new(S, D, H);

    Mat x = mat_new(S, D);
    for (int i = 0; i < S*D; i++) x.d[i] = 0.01f*(i%11-5);

    attn_fwd(&x, &x, &w, H, 0, c);    /* self-attention, non-causal */
    check("ao.r == S",  c->ao.r == S);
    check("ao.c == D",  c->ao.c == D);

    attn_fwd(&x, &x, &w, H, 1, c);    /* self-attention, causal */
    check("causal ao.r == S", c->ao.r == S);
    check("causal ao.c == D", c->ao.c == D);

    /* cross-attention: Q from x(S=5), KV from kv(KS=7) */
    int KS = 7;
    Mat kv = mat_new(KS, D);
    for (int i = 0; i < KS*D; i++) kv.d[i] = 0.01f*(i%13-6);
    AC *cc = ac_new(KS > S ? KS : S, D, H);
    attn_fwd(&x, &kv, &w, H, 0, cc);
    /* ao is pre-allocated to T rows; actual query length tracked via ac.S */
    check("cross c->S == S",  cc->S  == S);
    check("cross ao.c == D",  cc->ao.c == D);
    check("cross KS saved",   cc->KS == KS);

    ac_del(c); ac_del(cc); aw_del(&w);
    mat_del(&x); mat_del(&kv);
}

static void test_ffn_shape(void) {
    printf("[ffn_fwd shape]\n");
    int S=4, D=16, F=32;
    FW w = fw_new(D, F);
    Mat x  = mat_new(S, D);
    Mat h  = mat_new(S, F);
    Mat out= mat_new(S, D);
    for (int i=0;i<S*D;i++) x.d[i]=0.01f*(i%7-3);
    ffn_fwd(&x, &w, &h, &out);
    check("ffn h.r==S",   h.r  == S);
    check("ffn h.c==F",   h.c  == F);
    check("ffn out.r==S", out.r == S);
    check("ffn out.c==D", out.c == D);
    fw_del(&w); mat_del(&x); mat_del(&h); mat_del(&out);
}

static void test_encoder_shape(void) {
    printf("[el_fwd shape]\n");
    Cfg cfg = {.V=11,.T=8,.D=16,.H=4,.F=32,.L=1,.eps=1e-5f};
    EL w = el_new(cfg.D, cfg.F);
    EC *c = ec_new(cfg.T, cfg.D, cfg.F, cfg.H);
    int S = 5;
    Mat x = mat_new(S, cfg.D);
    for (int i=0;i<S*cfg.D;i++) x.d[i]=0.01f*(i%9-4);
    /* init weights */
    for(int i=0;i<cfg.D*cfg.D;i++){
        w.sa.Wq.w[i]=w.sa.Wk.w[i]=w.sa.Wv.w[i]=w.sa.Wo.w[i]=0.01f*(i%7-3);}
    for(int i=0;i<cfg.D*cfg.F;i++) w.ff.W1.w[i]=0.01f*(i%5-2);
    for(int i=0;i<cfg.F*cfg.D;i++) w.ff.W2.w[i]=0.01f*(i%5-2);
    for(int i=0;i<cfg.D;i++) w.ln1.scale.w[i]=w.ln2.scale.w[i]=1.f;
    el_fwd(&x, &w, c, &cfg, 0);
    /* xo pre-allocated to T rows; actual length tracked in self-attn cache */
    check("ec sa.S == S",  c->sa.S == S);
    check("ec xo.c == D",  c->xo.c == cfg.D);
    el_del(&w); ec_del(c); mat_del(&x);
}

static void test_decoder_shape(void) {
    printf("[dl_fwd shape]\n");
    Cfg cfg = {.V=11,.T=8,.D=16,.H=4,.F=32,.L=1,.eps=1e-5f};
    DL w = dl_new(cfg.D, cfg.F);
    DC *c = dc_new(cfg.T, cfg.D, cfg.F, cfg.H);
    int S=4, KS=6;
    Mat x   = mat_new(S,  cfg.D);
    Mat enc = mat_new(KS, cfg.D);
    for(int i=0;i<S*cfg.D;i++)  x.d[i]  =0.01f*(i%9-4);
    for(int i=0;i<KS*cfg.D;i++) enc.d[i] =0.01f*(i%7-3);
    /* init weights (small) */
    for(int i=0;i<cfg.D*cfg.D;i++){
        w.sa.Wq.w[i]=w.sa.Wk.w[i]=w.sa.Wv.w[i]=w.sa.Wo.w[i]=0.01f*(i%7-3);
        w.ca.Wq.w[i]=w.ca.Wk.w[i]=w.ca.Wv.w[i]=w.ca.Wo.w[i]=0.01f*(i%7-3);}
    for(int i=0;i<cfg.D*cfg.F;i++) w.ff.W1.w[i]=0.01f*(i%5-2);
    for(int i=0;i<cfg.F*cfg.D;i++) w.ff.W2.w[i]=0.01f*(i%5-2);
    for(int i=0;i<cfg.D;i++){
        w.ln1.scale.w[i]=w.ln2.scale.w[i]=w.ln3.scale.w[i]=1.f;}
    dl_fwd(&x, &enc, &w, c, &cfg);
    /* xo pre-allocated to T rows; actual length tracked in self-attn cache */
    check("dc sa.S == S",  c->sa.S == S);
    check("dc xo.c == D",  c->xo.c == cfg.D);
    check("dc ca.KS == KS",c->ca.KS == KS);
    dl_del(&w); dc_del(c); mat_del(&x); mat_del(&enc);
}

static void test_attn_bwd_finite_diff(void) {
    printf("[attn_bwd finite-diff dx[0]]\n");
    int S=3, D=8, H=2;
    AW w = aw_new(D);
    /* Xavier init */
    float s = 0.1f;
    for(int i=0;i<D*D;i++){
        w.Wq.w[i]=s*(i%5-2); w.Wk.w[i]=s*(i%7-3);
        w.Wv.w[i]=s*(i%11-5); w.Wo.w[i]=s*(i%3-1); }

    Mat x = mat_new(S, D);
    for(int i=0;i<S*D;i++) x.d[i]=0.05f*(i%9-4);

    AC *c  = ac_new(S, D, H);
    AC *cp = ac_new(S, D, H);
    AC *cm = ac_new(S, D, H);

    /* forward */
    attn_fwd(&x, &x, &w, H, 0, c);

    /* upstream dy = small random */
    Mat dao = mat_new(S, D);
    for(int i=0;i<S*D;i++) dao.d[i]=0.1f*(i%7-3);

    /* backward */
    Mat dx   = mat_new(S, D);  mat_zero(&dx);
    Mat dkv  = mat_new(S, D);  mat_zero(&dkv);
    AW  dw   = aw_new(D);
    attn_bwd(&w, H, 0, c, &dao, &dx, &dkv, &dw);

    /* analytical dx[self] = dx + dkv (self-attn: query and kv are the same x) */
    /* finite diff on x[0][0] */
    float eps = 1e-4f;
    Mat xp = mat_new(S,D); Mat xm = mat_new(S,D);
    memcpy(xp.d, x.d, S*D*sizeof(float)); xp.d[0] += eps;
    memcpy(xm.d, x.d, S*D*sizeof(float)); xm.d[0] -= eps;
    attn_fwd(&xp, &xp, &w, H, 0, cp);
    attn_fwd(&xm, &xm, &w, H, 0, cm);
    /* loss = sum(ao * dao) */
    float lp=0.f, lm=0.f;
    for(int i=0;i<S*D;i++){lp+=cp->ao.d[i]*dao.d[i]; lm+=cm->ao.d[i]*dao.d[i];}
    float fd = (lp-lm)/(2*eps);
    /* analytical: dx[0][0] + dkv[0][0] (self-attn both go to same x) */
    float anal = dx.d[0] + dkv.d[0];
    check("attn dx[0][0] finite-diff", fclose_to(anal, fd, 1e-2f));

    ac_del(c); ac_del(cp); ac_del(cm);
    aw_del(&w); aw_del(&dw);
    mat_del(&x); mat_del(&xp); mat_del(&xm);
    mat_del(&dao); mat_del(&dx); mat_del(&dkv);
}

/* --- Causal: no future peeking--- */
/*
 * Verify that with causal=1, changing position 2 does NOT affect
 * the hidden state at position 0. With causal=0 it does (information leak).
 */
static void test_causal_no_future_peek(void) {
    printf("[el_fwd causal=1: no future peeking]\n");
    Cfg cfg = {.V=11,.T=8,.D=16,.H=4,.F=32,.L=1,.eps=1e-5f};
    EL w = el_new(cfg.D, cfg.F);
    EC *c1 = ec_new(cfg.T, cfg.D, cfg.F, cfg.H);
    EC *c2 = ec_new(cfg.T, cfg.D, cfg.F, cfg.H);
    int S = 3;

    /* init non-trivial weights */
    for(int i=0;i<cfg.D*cfg.D;i++){
        w.sa.Wq.w[i]=0.05f*(i%7-3); w.sa.Wk.w[i]=0.05f*(i%11-5);
        w.sa.Wv.w[i]=0.05f*(i%5-2); w.sa.Wo.w[i]=0.05f*(i%3-1);}
    for(int i=0;i<cfg.D*cfg.F;i++) w.ff.W1.w[i]=0.02f*(i%5-2);
    for(int i=0;i<cfg.F*cfg.D;i++) w.ff.W2.w[i]=0.02f*(i%5-2);
    for(int i=0;i<cfg.D;i++) w.ln1.scale.w[i]=w.ln2.scale.w[i]=1.f;

    /* sequence x: positions 0,1,2 */
    Mat x1 = mat_new(S, cfg.D);
    Mat x2 = mat_new(S, cfg.D);
    for(int i=0;i<S*cfg.D;i++) x1.d[i]=x2.d[i]=0.01f*(i%9-4);
    /* change ONLY position 2 in x2 */
    for(int d=0;d<cfg.D;d++) x2.d[2*cfg.D+d] += 1.0f;

    el_fwd(&x1, &w, c1, &cfg, 1);  /* causal */
    el_fwd(&x2, &w, c2, &cfg, 1);

    /* position 0 output must be IDENTICAL (causal=1 means pos 0 can't see pos 2) */
    float diff0 = 0.f;
    for(int d=0;d<cfg.D;d++) diff0 += fabsf(c1->xo.d[d] - c2->xo.d[d]);
    check("pos0 unchanged after pos2 change (causal=1)", diff0 < 1e-5f);

    /* position 2 output MUST differ (it received the change) */
    float diff2 = 0.f;
    for(int d=0;d<cfg.D;d++) diff2 += fabsf(c1->xo.d[2*cfg.D+d] - c2->xo.d[2*cfg.D+d]);
    check("pos2 changes after pos2 input change", diff2 > 1e-4f);

    /* sanity: with causal=0, pos0 WOULD change (information flows from pos2) */
    EC *c3 = ec_new(cfg.T, cfg.D, cfg.F, cfg.H);
    EC *c4 = ec_new(cfg.T, cfg.D, cfg.F, cfg.H);
    el_fwd(&x1, &w, c3, &cfg, 0);
    el_fwd(&x2, &w, c4, &cfg, 0);
    float diff0_nc = 0.f;
    for(int d=0;d<cfg.D;d++) diff0_nc += fabsf(c3->xo.d[d] - c4->xo.d[d]);
    check("pos0 changes with causal=0 (confirms non-causal leaks)", diff0_nc > 1e-5f);

    el_del(&w); ec_del(c1); ec_del(c2); ec_del(c3); ec_del(c4);
    mat_del(&x1); mat_del(&x2);
}

static void test_event_head_to_seq(void) {
    printf("[event_head_to_seq]\n");
    int V=11+8;  /* TEXT_VOCAB=11, TEMP_BINS=8 */
    TextAdapter ta = {.vocab_size = 11};
    ScalarBinAdapter sba = {.modality=MOD_TEMPERATURE,.channel=0,.n_bins=8,.vocab_offset=11};

    /* logits: 3 positions, argmax = [1 (BOS), 13 (TEMP bin 2), 2 (EOS)] */
    Mat logits = mat_new(3, V);
    logits.d[0*V + 1]  = 5.f;   /* pos0: text BOS */
    logits.d[1*V + 13] = 5.f;   /* pos1: temp bin 2 (11+2) */
    logits.d[2*V + 2]  = 5.f;   /* pos2: text EOS */

    EventSeq *out = event_seq_new(8);
    int n = event_head_to_seq(&logits, &ta, &sba, 1, out, 2 /*EOS*/);

    check("decoded 3 events", n == 3);
    check("pos0 is TEXT", out->modality[0] == MOD_TEXT);
    check("pos0 token_id == 1", out->token_id[0] == 1);
    check("pos1 is TEMPERATURE", out->modality[1] == MOD_TEMPERATURE);
    check("pos1 value == 2/7", fclose_to(out->value[1], 2.f/7.f, 1e-4f));
    check("pos2 is TEXT EOS", out->token_id[2] == 2);

    mat_del(&logits);
    event_seq_del(out);
}

static void test_event_embed_validation(void) {
    printf("[event_embed_fwd validation]\n");
    int D=8, V=11, T=8;
    EventEmbed *e = event_embed_new(D, V, T);
    event_embed_init(e);
    Mat out = mat_new(1, D);

    /* valid event */
    EventSeq *s = event_seq_new(4);
    int tok = 3;
    event_append_text(s, &tok, 1, 0);
    check("valid event returns 0", event_embed_fwd(s, e, &out) == 0);

    /* invalid modality */
    s->modality[0] = EVENT_MAX_MOD + 1;
    check("bad modality returns -1", event_embed_fwd(s, e, &out) == -1);
    s->modality[0] = MOD_TEXT;

    /* invalid time_index */
    s->time_index[0] = T + 5;
    check("bad time_index returns -1", event_embed_fwd(s, e, &out) == -1);
    s->time_index[0] = 0;

    /* invalid token_id */
    s->token_id[0] = V + 1;
    check("bad token_id returns -1", event_embed_fwd(s, e, &out) == -1);

    mat_del(&out);
    event_seq_del(s);
    event_embed_del(e);
}

static void test_sba_guard(void) {
    printf("[ScalarBinAdapter n_bins guard]\n");
    ScalarBinAdapter bad = {.modality=MOD_TEMPERATURE,.channel=0,.n_bins=1,.vocab_offset=0};
    EventSeq *s = event_seq_new(4);
    check("sba_encode n_bins=1 returns -1", sba_encode(&bad, s, 0.5f, 0) == -1);
    check("sba_decode n_bins=1 returns -1", fclose_to(sba_decode(&bad, 0), -1.f, 1e-5f));
    check("sba_label  n_bins=1 returns offset", sba_label(&bad, 0.5f) == bad.vocab_offset);
    event_seq_del(s);
}

/* ---- serialization round-trip tests ---- */
static void test_model_serialize(void) {
    printf("[model_save/load round-trip]\n");
    Cfg cfg = {.V=11,.T=8,.D=16,.H=4,.F=32,.L=2,.eps=1e-5f};
    Model *m = model_new(&cfg);
    model_init(m);
    const char *path = "test_model.bin";
    check("model_save ok", model_save(m, path) == 0);
    Model *m2 = model_load(path);
    check("model_load non-null", m2 != NULL);
    if (m2) {
        check("cfg.V preserved", m2->c.V == cfg.V);
        check("cfg.D preserved", m2->c.D == cfg.D);
        check("cfg.L preserved", m2->c.L == cfg.L);
        int same = 1;
        for (int i=0;i<cfg.V*cfg.D;i++) if (m->se.w[i]!=m2->se.w[i]) {same=0;break;}
        check("se weights match", same);
        same = 1;
        for (int i=0;i<cfg.D*cfg.V;i++) if (m->proj.w[i]!=m2->proj.w[i]) {same=0;break;}
        check("proj weights match", same);
        same = 1;
        for (int i=0;i<cfg.D*cfg.D;i++)
            if (m->enc[0].sa.Wq.w[i]!=m2->enc[0].sa.Wq.w[i]) {same=0;break;}
        check("enc[0].sa.Wq match", same);
        same = 1;
        for (int i=0;i<cfg.D*cfg.D;i++)
            if (m->dec[1].ca.Wv.w[i]!=m2->dec[1].ca.Wv.w[i]) {same=0;break;}
        check("dec[1].ca.Wv match", same);
        model_del(m2);
    }
    remove(path);
    model_del(m);
}

static void test_event_serialize(void) {
    printf("[event_embed/head save/load round-trip]\n");
    srand(7);
    EventEmbed *e = event_embed_new(16, 11, 8);
    event_embed_init(e);
    const char *ep = "test_ee.bin";
    check("event_embed_save ok", event_embed_save(e, ep) == 0);
    EventEmbed *e2 = event_embed_load(ep);
    check("event_embed_load non-null", e2 != NULL);
    if (e2) {
        int same = 1;
        for (int i=0;i<e->tok_emb.n;i++) if (e->tok_emb.w[i]!=e2->tok_emb.w[i]) {same=0;break;}
        check("tok_emb match", same);
        same = 1;
        for (int i=0;i<e->val_w.n;i++) if (e->val_w.w[i]!=e2->val_w.w[i]) {same=0;break;}
        check("val_w match", same);
        event_embed_del(e2);
    }
    remove(ep);
    event_embed_del(e);

    EventHead *h = event_head_new(16, 19);
    event_head_init(h);
    const char *hp = "test_eh.bin";
    check("event_head_save ok", event_head_save(h, hp) == 0);
    EventHead *h2 = event_head_load(hp);
    check("event_head_load non-null", h2 != NULL);
    if (h2) {
        int same = 1;
        for (int i=0;i<h->proj.n;i++) if (h->proj.w[i]!=h2->proj.w[i]) {same=0;break;}
        check("head proj match", same);
        event_head_del(h2);
    }
    remove(hp);
    event_head_del(h);
}

static void test_load_bad_magic(void) {
    printf("[model_load rejects bad file]\n");
    const char *path = "test_bad.bin";
    FILE *f = fopen(path, "wb");
    if (f) { fputs("XXXXgarbage", f); fclose(f); }
    check("model_load returns NULL on bad magic", model_load(path) == NULL);
    remove(path);
}

static int write_test_model_file(const char *path, int V, int T, int D, int H, int F, int L, int include_weights) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int ver = 1;
    float eps = 1e-5f;
    fwrite("MYLM", 1, 4, f);
    fwrite(&ver, sizeof(int), 1, f);
    fwrite(&V, sizeof(int), 1, f);
    fwrite(&T, sizeof(int), 1, f);
    fwrite(&D, sizeof(int), 1, f);
    fwrite(&H, sizeof(int), 1, f);
    fwrite(&F, sizeof(int), 1, f);
    fwrite(&L, sizeof(int), 1, f);
    fwrite(&eps, sizeof(float), 1, f);
    (void)include_weights;
    fclose(f);
    return 0;
}

static void test_model_load_bounds(void) {
    printf("[model_load bounds/overflow]\n");
    const char *path = "test_bounds.bin";

    /* test: V exceeds MAX_V (100000) */
    write_test_model_file(path, 200000, 8, 8, 2, 16, 1, 0);
    check("model_load rejects V>MAX_V", model_load(path) == NULL);

    /* test: D exceeds MAX_D (2048) */
    write_test_model_file(path, 1000, 8, 3000, 2, 16, 1, 0);
    check("model_load rejects D>MAX_D", model_load(path) == NULL);

    /* test: L exceeds MAX_L (64) */
    write_test_model_file(path, 1000, 8, 64, 2, 128, 100, 0);
    check("model_load rejects L>MAX_L", model_load(path) == NULL);

    /* test: total params exceed MAX_PARAMS (500M) - large L with moderate D */
    write_test_model_file(path, 50000, 1024, 1024, 8, 4096, 60, 0);
    check("model_load rejects total>MAX_PARAMS", model_load(path) == NULL);

    /* test: negative V */
    write_test_model_file(path, -1, 8, 8, 2, 16, 1, 0);
    check("model_load rejects negative V", model_load(path) == NULL);

    /* test: truncated file (missing weights) - valid header but no data */
    write_test_model_file(path, 8, 4, 8, 2, 16, 1, 0);
    check("model_load rejects truncated file", model_load(path) == NULL);

    remove(path);
}

/* ---- input validation tests for model_fwd / model_encode etc. ---- */
static void test_model_fwd_validation(void) {
    printf("[model_fwd input validation]\n");
    Cfg cfg = {.V=8,.T=4,.D=8,.H=2,.F=16,.L=1,.eps=1e-5f};
    Model *m = model_new(&cfg);
    model_init(m);

    EC *ec[1]; ec[0] = ec_new(cfg.T, cfg.D, cfg.F, cfg.H);
    DC *dc[1]; dc[0] = dc_new(cfg.T, cfg.D, cfg.F, cfg.H);
    Mat enc_out = mat_new(cfg.T, cfg.D);
    Mat dec_out = mat_new(cfg.T, cfg.D);
    Mat logits  = mat_new(cfg.T, cfg.V);

    int src_ok[2] = {1, 2};
    int tgt_ok[2] = {0, 1};
    int lbl_ok[2] = {1, 2};

    /* valid input - also verify output shapes */
    check("model_fwd valid returns 0", model_fwd(m, src_ok, 2, tgt_ok, 2, ec, dc, &enc_out, &dec_out, &logits) == 0);
    check("enc_out.r == SL", enc_out.r == 2);
    check("enc_out.c == D", enc_out.c == cfg.D);
    check("dec_out.r == TL", dec_out.r == 2);
    check("dec_out.c == D", dec_out.c == cfg.D);
    check("logits.r == TL", logits.r == 2);
    check("logits.c == V", logits.c == cfg.V);

    /* SL out of range */
    check("model_fwd SL=0 returns -1", model_fwd(m, src_ok, 0, tgt_ok, 2, ec, dc, &enc_out, &dec_out, &logits) == -1);
    check("model_fwd SL>T returns -1", model_fwd(m, src_ok, cfg.T+1, tgt_ok, 2, ec, dc, &enc_out, &dec_out, &logits) == -1);

    /* TL out of range */
    check("model_fwd TL=0 returns -1", model_fwd(m, src_ok, 2, tgt_ok, 0, ec, dc, &enc_out, &dec_out, &logits) == -1);
    check("model_fwd TL>T returns -1", model_fwd(m, src_ok, 2, tgt_ok, cfg.T+1, ec, dc, &enc_out, &dec_out, &logits) == -1);

    /* src token out of range */
    int src_bad[2] = {1, cfg.V};
    check("model_fwd src>=V returns -1", model_fwd(m, src_bad, 2, tgt_ok, 2, ec, dc, &enc_out, &dec_out, &logits) == -1);
    int src_neg[2] = {-1, 2};
    check("model_fwd src<0 returns -1", model_fwd(m, src_neg, 2, tgt_ok, 2, ec, dc, &enc_out, &dec_out, &logits) == -1);

    /* tgt token out of range */
    int tgt_bad[2] = {0, cfg.V};
    check("model_fwd tgt>=V returns -1", model_fwd(m, src_ok, 2, tgt_bad, 2, ec, dc, &enc_out, &dec_out, &logits) == -1);

    /* model_loss_bwd tests - need valid forward pass first */
    model_fwd(m, src_ok, 2, tgt_ok, 2, ec, dc, &enc_out, &dec_out, &logits);  /* setup valid state */
    check("model_loss_bwd valid returns >=0", model_loss_bwd(m, src_ok, 2, tgt_ok, 2, lbl_ok, ec, dc, &enc_out, &dec_out, &logits) >= 0.f);
    check("model_loss_bwd SL=0 returns -1", model_loss_bwd(m, src_ok, 0, tgt_ok, 2, lbl_ok, ec, dc, &enc_out, &dec_out, &logits) < 0.f);
    int lbl_bad[2] = {1, cfg.V};
    check("model_loss_bwd lbl>=V returns -1", model_loss_bwd(m, src_ok, 2, tgt_ok, 2, lbl_bad, ec, dc, &enc_out, &dec_out, &logits) < 0.f);

    mat_del(&enc_out); mat_del(&dec_out); mat_del(&logits);
    ec_del(ec[0]); dc_del(dc[0]);
    model_del(m);
}

static void test_model_encode_validation(void) {
    printf("[model_encode input validation]\n");
    Cfg cfg = {.V=8,.T=4,.D=8,.H=2,.F=16,.L=1,.eps=1e-5f};
    Model *m = model_new(&cfg);
    model_init(m);
    EC *ec[1]; ec[0] = ec_new(cfg.T, cfg.D, cfg.F, cfg.H);
    Mat enc_out = mat_new(cfg.T, cfg.D);

    int src_ok[2] = {1, 2};
    check("model_encode valid returns 0", model_encode(m, src_ok, 2, ec, &enc_out) == 0);
    check("model_encode SL=0 returns -1", model_encode(m, src_ok, 0, ec, &enc_out) == -1);
    check("model_encode SL>T returns -1", model_encode(m, src_ok, cfg.T+1, ec, &enc_out) == -1);
    int src_bad[2] = {1, cfg.V};
    check("model_encode src>=V returns -1", model_encode(m, src_bad, 2, ec, &enc_out) == -1);

    mat_del(&enc_out);
    ec_del(ec[0]);
    model_del(m);
}

static void test_decode_cache_precompute_validation(void) {
    printf("[decode_cache_precompute_cross validation]\n");
    Cfg cfg = {.V=8,.T=4,.D=8,.H=2,.F=16,.L=1,.eps=1e-5f};
    Model *m = model_new(&cfg);
    model_init(m);
    DecodeCache *dkv = decode_cache_new(&cfg);

    Mat enc_ok = mat_new(2, cfg.D);  /* SL=2, within T=4 */
    check("precompute valid returns 0", decode_cache_precompute_cross(dkv, m, &enc_ok) == 0);

    Mat enc_too_long = mat_new(cfg.T+1, cfg.D);  /* SL > max_len */
    check("precompute SL>max_len returns -1", decode_cache_precompute_cross(dkv, m, &enc_too_long) == -1);

    mat_del(&enc_ok); mat_del(&enc_too_long);
    decode_cache_del(dkv);
    model_del(m);
}

int main(void) {
    srand(0);
    printf("=== MyLLM unit tests ===\n\n");
    test_mm();
    test_mm_at();
    test_gelu();
    test_gelu_bwd();
    test_softmax();
    test_layernorm();
    test_layernorm_bwd();
    test_event_embed();
    test_attn_shape();
    test_ffn_shape();
    test_encoder_shape();
    test_decoder_shape();
    test_attn_bwd_finite_diff();
    test_causal_no_future_peek();
    test_event_head_to_seq();
    test_event_embed_validation();
    test_sba_guard();
    test_model_serialize();
    test_event_serialize();
    test_load_bad_magic();
    test_model_load_bounds();
    test_model_fwd_validation();
    test_model_encode_validation();
    test_decode_cache_precompute_validation();
    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
