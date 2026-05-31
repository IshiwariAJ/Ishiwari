#include "model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* --- P--- */
P p_new(int n) {
    P p; p.n = n;
    p.w = (float*)calloc(n, sizeof(float));
    p.g = (float*)calloc(n, sizeof(float));
    p.m = (float*)calloc(n, sizeof(float));
    p.v = (float*)calloc(n, sizeof(float));
    if (!p.w||!p.g||!p.m||!p.v) { fprintf(stderr,"OOM p_new(%d)\n",n); exit(1); }
    return p;
}
void p_del(P *p) { free(p->w); free(p->g); free(p->m); free(p->v); p->n=0; }
void p_zg(P *p)  { memset(p->g, 0, p->n * sizeof(float)); }

/* --- LN--- */
LN ln_new(int d) {
    LN ln; ln.scale = p_new(d); ln.shift = p_new(d);
    /* init gamma=1, beta=0 */
    for (int i = 0; i < d; i++) ln.scale.w[i] = 1.f;
    return ln;
}
void ln_del(LN *ln) { p_del(&ln->scale); p_del(&ln->shift); }
void ln_zg(LN *ln)  { p_zg(&ln->scale); p_zg(&ln->shift); }

/* --- AW--- */
AW aw_new(int D) {
    AW a;
    a.Wq = p_new(D*D); a.Wk = p_new(D*D);
    a.Wv = p_new(D*D); a.Wo = p_new(D*D);
    a.bq = p_new(D);   a.bk = p_new(D);
    a.bv = p_new(D);   a.bo = p_new(D);
    return a;
}
void aw_del(AW *a) {
    p_del(&a->Wq); p_del(&a->Wk); p_del(&a->Wv); p_del(&a->Wo);
    p_del(&a->bq); p_del(&a->bk); p_del(&a->bv); p_del(&a->bo);
}
void aw_zg(AW *a) {
    p_zg(&a->Wq); p_zg(&a->Wk); p_zg(&a->Wv); p_zg(&a->Wo);
    p_zg(&a->bq); p_zg(&a->bk); p_zg(&a->bv); p_zg(&a->bo);
}

/* --- FW--- */
FW fw_new(int D, int F) {
    FW f;
    f.W1 = p_new(D*F); f.b1 = p_new(F);
    f.W2 = p_new(F*D); f.b2 = p_new(D);
    return f;
}
void fw_del(FW *f) {
    p_del(&f->W1); p_del(&f->b1); p_del(&f->W2); p_del(&f->b2);
}
void fw_zg(FW *f) {
    p_zg(&f->W1); p_zg(&f->b1); p_zg(&f->W2); p_zg(&f->b2);
}

/* --- EL / DL--- */
EL el_new(int D, int F) {
    EL e; e.sa=aw_new(D); e.ff=fw_new(D,F);
    e.ln1=ln_new(D); e.ln2=ln_new(D);
    return e;
}
DL dl_new(int D, int F) {
    DL d; d.sa=aw_new(D); d.ca=aw_new(D); d.ff=fw_new(D,F);
    d.ln1=ln_new(D); d.ln2=ln_new(D); d.ln3=ln_new(D);
    return d;
}
void el_del(EL *e) { aw_del(&e->sa); fw_del(&e->ff); ln_del(&e->ln1); ln_del(&e->ln2); }
void dl_del(DL *d) {
    aw_del(&d->sa); aw_del(&d->ca); fw_del(&d->ff);
    ln_del(&d->ln1); ln_del(&d->ln2); ln_del(&d->ln3);
}
void el_zg(EL *e) { aw_zg(&e->sa); fw_zg(&e->ff); ln_zg(&e->ln1); ln_zg(&e->ln2); }
void dl_zg(DL *d) {
    aw_zg(&d->sa); aw_zg(&d->ca); fw_zg(&d->ff);
    ln_zg(&d->ln1); ln_zg(&d->ln2); ln_zg(&d->ln3);
}

/* --- Caches--- */
static void ac_init(AC *c, int T, int D, int H) {
    c->Q  = mat_new(T, D); c->K  = mat_new(T, D); c->V  = mat_new(T, D);
    c->hs = (float*)calloc((size_t)H*T*T, sizeof(float));
    c->ha = (float*)calloc((size_t)H*T*T, sizeof(float));
    c->co = mat_new(T, D); c->ao = mat_new(T, D);
    c->xi = mat_new(T, D); c->ki = mat_new(T, D);
}
AC *ac_new(int T, int D, int H) {
    AC *c = (AC*)calloc(1, sizeof(AC));
    ac_init(c, T, D, H);
    return c;
}
EC *ec_new(int T, int D, int F, int H) {
    EC *c = (EC*)calloc(1, sizeof(EC));
    c->xi = mat_new(T,D);
    ac_init(&c->sa, T, D, H);
    c->r1 = mat_new(T,D); c->x1 = mat_new(T,D);
    c->mn1 = (float*)calloc(T, sizeof(float));
    c->vr1 = (float*)calloc(T, sizeof(float));
    c->fh = mat_new(T,F); c->fo = mat_new(T,D);
    c->r2 = mat_new(T,D); c->xo = mat_new(T,D);
    c->mn2 = (float*)calloc(T, sizeof(float));
    c->vr2 = (float*)calloc(T, sizeof(float));
    return c;
}
DC *dc_new(int T, int D, int F, int H) {
    DC *c = (DC*)calloc(1, sizeof(DC));
    c->xi = mat_new(T,D);
    ac_init(&c->sa, T, D, H);
    c->r1 = mat_new(T,D); c->x1 = mat_new(T,D);
    c->mn1=(float*)calloc(T,sizeof(float)); c->vr1=(float*)calloc(T,sizeof(float));
    ac_init(&c->ca, T, D, H);
    c->r2 = mat_new(T,D); c->x2 = mat_new(T,D);
    c->mn2=(float*)calloc(T,sizeof(float)); c->vr2=(float*)calloc(T,sizeof(float));
    c->fh = mat_new(T,F); c->fo = mat_new(T,D);
    c->r3 = mat_new(T,D); c->xo = mat_new(T,D);
    c->mn3=(float*)calloc(T,sizeof(float)); c->vr3=(float*)calloc(T,sizeof(float));
    return c;
}
void ac_del(AC *a) {
    mat_del(&a->Q); mat_del(&a->K); mat_del(&a->V);
    free(a->hs); free(a->ha);
    mat_del(&a->co); mat_del(&a->ao);
    mat_del(&a->xi); mat_del(&a->ki);
    free(a);
}
void ec_del(EC *e) {
    mat_del(&e->xi);
    mat_del(&e->sa.Q); mat_del(&e->sa.K); mat_del(&e->sa.V);
    free(e->sa.hs); free(e->sa.ha);
    mat_del(&e->sa.co); mat_del(&e->sa.ao);
    mat_del(&e->sa.xi); mat_del(&e->sa.ki);
    mat_del(&e->r1); mat_del(&e->x1);
    free(e->mn1); free(e->vr1);
    mat_del(&e->fh); mat_del(&e->fo);
    mat_del(&e->r2); mat_del(&e->xo);
    free(e->mn2); free(e->vr2);
    free(e);
}
void dc_del(DC *d) {
    mat_del(&d->xi);
    mat_del(&d->sa.Q); mat_del(&d->sa.K); mat_del(&d->sa.V);
    free(d->sa.hs); free(d->sa.ha);
    mat_del(&d->sa.co); mat_del(&d->sa.ao);
    mat_del(&d->sa.xi); mat_del(&d->sa.ki);
    mat_del(&d->r1); mat_del(&d->x1);
    free(d->mn1); free(d->vr1);
    mat_del(&d->ca.Q); mat_del(&d->ca.K); mat_del(&d->ca.V);
    free(d->ca.hs); free(d->ca.ha);
    mat_del(&d->ca.co); mat_del(&d->ca.ao);
    mat_del(&d->ca.xi); mat_del(&d->ca.ki);
    mat_del(&d->r2); mat_del(&d->x2);
    free(d->mn2); free(d->vr2);
    mat_del(&d->fh); mat_del(&d->fo);
    mat_del(&d->r3); mat_del(&d->xo);
    free(d->mn3); free(d->vr3);
    free(d);
}

/* --- Model alloc/del/init--- */
static void make_pos(float *pos, int T, int D) {
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++) {
            float freq = powf(10000.f, -(float)(d/2*2)/(float)D);
            pos[t*D+d] = (d%2==0) ? sinf(t*freq) : cosf(t*freq);
        }
}

static void xavier(P *p, int fan_in, int fan_out) {
    float s = sqrtf(2.f / (fan_in + fan_out));
    for (int i = 0; i < p->n; i++) {
        float u = ((float)rand()+1)/((float)RAND_MAX+2);
        float v = ((float)rand()+1)/((float)RAND_MAX+2);
        p->w[i] = sqrtf(-2.f*logf(u))*cosf(6.2831853f*v)*s;
    }
}

Model *model_new(const Cfg *cfg) {
    Model *m = (Model*)calloc(1, sizeof(Model));
    m->c = *cfg;
    int V=cfg->V, D=cfg->D, F=cfg->F, T=cfg->T, L=cfg->L;
    m->se  = p_new(V*D); m->te = p_new(V*D);
    m->pos = (float*)calloc((size_t)T*D, sizeof(float));
    make_pos(m->pos, T, D);
    m->enc = (EL*)malloc(L * sizeof(EL));
    m->dec = (DL*)malloc(L * sizeof(DL));
    for (int i = 0; i < L; i++) { m->enc[i] = el_new(D,F); m->dec[i] = dl_new(D,F); }
    m->proj   = p_new(D*V);
    m->proj_b = p_new(V);
    return m;
}
void model_del(Model *m) {
    p_del(&m->se); p_del(&m->te); free(m->pos);
    for (int i = 0; i < m->c.L; i++) { el_del(&m->enc[i]); dl_del(&m->dec[i]); }
    free(m->enc); free(m->dec);
    p_del(&m->proj); p_del(&m->proj_b);
    free(m);
}
void model_zg(Model *m) {
    p_zg(&m->se); p_zg(&m->te);
    for (int i = 0; i < m->c.L; i++) { el_zg(&m->enc[i]); dl_zg(&m->dec[i]); }
    p_zg(&m->proj); p_zg(&m->proj_b);
}
void model_init(Model *m) {
    int D=m->c.D, F=m->c.F, V=m->c.V;
    xavier(&m->se, V, D); xavier(&m->te, V, D);
    for (int i = 0; i < m->c.L; i++) {
        xavier(&m->enc[i].sa.Wq,D,D); xavier(&m->enc[i].sa.Wk,D,D);
        xavier(&m->enc[i].sa.Wv,D,D); xavier(&m->enc[i].sa.Wo,D,D);
        xavier(&m->enc[i].ff.W1,D,F); xavier(&m->enc[i].ff.W2,F,D);
        xavier(&m->dec[i].sa.Wq,D,D); xavier(&m->dec[i].sa.Wk,D,D);
        xavier(&m->dec[i].sa.Wv,D,D); xavier(&m->dec[i].sa.Wo,D,D);
        xavier(&m->dec[i].ca.Wq,D,D); xavier(&m->dec[i].ca.Wk,D,D);
        xavier(&m->dec[i].ca.Wv,D,D); xavier(&m->dec[i].ca.Wo,D,D);
        xavier(&m->dec[i].ff.W1,D,F); xavier(&m->dec[i].ff.W2,F,D);
    }
    xavier(&m->proj, D, V);
}
