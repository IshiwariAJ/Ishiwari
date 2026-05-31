#include "model.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Adam update for a single Param */
static void adam_p(P *p, float lr, float b1, float b2, float eps, int t) {
    float bc1 = 1.f - powf(b1, t);
    float bc2 = 1.f - powf(b2, t);
    float lr_t = lr * sqrtf(bc2) / bc1;
    for (int i = 0; i < p->n; i++) {
        p->m[i] = b1 * p->m[i] + (1.f-b1) * p->g[i];
        p->v[i] = b2 * p->v[i] + (1.f-b2) * p->g[i] * p->g[i];
        p->w[i] -= lr_t * p->m[i] / (sqrtf(p->v[i]) + eps);
    }
}

static void adam_ln(LN *ln, float lr, float b1, float b2, float eps, int t) {
    adam_p(&ln->scale, lr, b1, b2, eps, t);
    adam_p(&ln->shift,  lr, b1, b2, eps, t);
}

static void adam_aw(AW *a, float lr, float b1, float b2, float eps, int t) {
    adam_p(&a->Wq, lr,b1,b2,eps,t); adam_p(&a->Wk, lr,b1,b2,eps,t);
    adam_p(&a->Wv, lr,b1,b2,eps,t); adam_p(&a->Wo, lr,b1,b2,eps,t);
    adam_p(&a->bq, lr,b1,b2,eps,t); adam_p(&a->bk, lr,b1,b2,eps,t);
    adam_p(&a->bv, lr,b1,b2,eps,t); adam_p(&a->bo, lr,b1,b2,eps,t);
}

static void adam_fw(FW *f, float lr, float b1, float b2, float eps, int t) {
    adam_p(&f->W1, lr,b1,b2,eps,t); adam_p(&f->b1, lr,b1,b2,eps,t);
    adam_p(&f->W2, lr,b1,b2,eps,t); adam_p(&f->b2, lr,b1,b2,eps,t);
}

void opt_step(Model *m, Opt *o) {
    o->step++;
    int t = o->step;
    float lr=o->lr, b1=o->b1, b2=o->b2, eps=o->eps;

    adam_p(&m->se, lr,b1,b2,eps,t);
    adam_p(&m->te, lr,b1,b2,eps,t);

    for (int i = 0; i < m->c.L; i++) {
        adam_aw(&m->enc[i].sa, lr,b1,b2,eps,t);
        adam_fw(&m->enc[i].ff, lr,b1,b2,eps,t);
        adam_ln(&m->enc[i].ln1, lr,b1,b2,eps,t);
        adam_ln(&m->enc[i].ln2, lr,b1,b2,eps,t);

        adam_aw(&m->dec[i].sa, lr,b1,b2,eps,t);
        adam_aw(&m->dec[i].ca, lr,b1,b2,eps,t);
        adam_fw(&m->dec[i].ff, lr,b1,b2,eps,t);
        adam_ln(&m->dec[i].ln1, lr,b1,b2,eps,t);
        adam_ln(&m->dec[i].ln2, lr,b1,b2,eps,t);
        adam_ln(&m->dec[i].ln3, lr,b1,b2,eps,t);
    }

    adam_p(&m->proj,   lr,b1,b2,eps,t);
    adam_p(&m->proj_b, lr,b1,b2,eps,t);
}

#define OPT_STATE_MAGIC   "OPTS"
#define OPT_STATE_VERSION 1

static int opt_wr(const void *p, size_t sz, size_t n, FILE *f) {
    return fwrite(p, sz, n, f) == n;
}

static int opt_rd(void *p, size_t sz, size_t n, FILE *f) {
    return fread(p, sz, n, f) == n;
}

static int save_p_state(FILE *f, const P *p) {
    return opt_wr(p->m, sizeof(float), p->n, f) &&
           opt_wr(p->v, sizeof(float), p->n, f);
}

static int load_p_state(FILE *f, P *p) {
    return opt_rd(p->m, sizeof(float), p->n, f) &&
           opt_rd(p->v, sizeof(float), p->n, f);
}

static void copy_p_state(P *dst, const P *src) {
    memcpy(dst->m, src->m, sizeof(float) * dst->n);
    memcpy(dst->v, src->v, sizeof(float) * dst->n);
}

static int save_ln_state(FILE *f, const LN *ln) {
    return save_p_state(f, &ln->scale) && save_p_state(f, &ln->shift);
}

static int load_ln_state(FILE *f, LN *ln) {
    return load_p_state(f, &ln->scale) && load_p_state(f, &ln->shift);
}

static void copy_ln_state(LN *dst, const LN *src) {
    copy_p_state(&dst->scale, &src->scale);
    copy_p_state(&dst->shift, &src->shift);
}

static int save_aw_state(FILE *f, const AW *a) {
    return save_p_state(f, &a->Wq) && save_p_state(f, &a->Wk) &&
           save_p_state(f, &a->Wv) && save_p_state(f, &a->Wo) &&
           save_p_state(f, &a->bq) && save_p_state(f, &a->bk) &&
           save_p_state(f, &a->bv) && save_p_state(f, &a->bo);
}

static int load_aw_state(FILE *f, AW *a) {
    return load_p_state(f, &a->Wq) && load_p_state(f, &a->Wk) &&
           load_p_state(f, &a->Wv) && load_p_state(f, &a->Wo) &&
           load_p_state(f, &a->bq) && load_p_state(f, &a->bk) &&
           load_p_state(f, &a->bv) && load_p_state(f, &a->bo);
}

static void copy_aw_state(AW *dst, const AW *src) {
    copy_p_state(&dst->Wq, &src->Wq);
    copy_p_state(&dst->Wk, &src->Wk);
    copy_p_state(&dst->Wv, &src->Wv);
    copy_p_state(&dst->Wo, &src->Wo);
    copy_p_state(&dst->bq, &src->bq);
    copy_p_state(&dst->bk, &src->bk);
    copy_p_state(&dst->bv, &src->bv);
    copy_p_state(&dst->bo, &src->bo);
}

static int save_fw_state(FILE *f, const FW *w) {
    return save_p_state(f, &w->W1) && save_p_state(f, &w->b1) &&
           save_p_state(f, &w->W2) && save_p_state(f, &w->b2);
}

static int load_fw_state(FILE *f, FW *w) {
    return load_p_state(f, &w->W1) && load_p_state(f, &w->b1) &&
           load_p_state(f, &w->W2) && load_p_state(f, &w->b2);
}

static void copy_fw_state(FW *dst, const FW *src) {
    copy_p_state(&dst->W1, &src->W1);
    copy_p_state(&dst->b1, &src->b1);
    copy_p_state(&dst->W2, &src->W2);
    copy_p_state(&dst->b2, &src->b2);
}

static int save_el_state(FILE *f, const EL *e) {
    return save_aw_state(f, &e->sa) && save_fw_state(f, &e->ff) &&
           save_ln_state(f, &e->ln1) && save_ln_state(f, &e->ln2);
}

static int load_el_state(FILE *f, EL *e) {
    return load_aw_state(f, &e->sa) && load_fw_state(f, &e->ff) &&
           load_ln_state(f, &e->ln1) && load_ln_state(f, &e->ln2);
}

static void copy_el_state(EL *dst, const EL *src) {
    copy_aw_state(&dst->sa, &src->sa);
    copy_fw_state(&dst->ff, &src->ff);
    copy_ln_state(&dst->ln1, &src->ln1);
    copy_ln_state(&dst->ln2, &src->ln2);
}

static int save_dl_state(FILE *f, const DL *d) {
    return save_aw_state(f, &d->sa) && save_aw_state(f, &d->ca) &&
           save_fw_state(f, &d->ff) && save_ln_state(f, &d->ln1) &&
           save_ln_state(f, &d->ln2) && save_ln_state(f, &d->ln3);
}

static int load_dl_state(FILE *f, DL *d) {
    return load_aw_state(f, &d->sa) && load_aw_state(f, &d->ca) &&
           load_fw_state(f, &d->ff) && load_ln_state(f, &d->ln1) &&
           load_ln_state(f, &d->ln2) && load_ln_state(f, &d->ln3);
}

static void copy_dl_state(DL *dst, const DL *src) {
    copy_aw_state(&dst->sa, &src->sa);
    copy_aw_state(&dst->ca, &src->ca);
    copy_fw_state(&dst->ff, &src->ff);
    copy_ln_state(&dst->ln1, &src->ln1);
    copy_ln_state(&dst->ln2, &src->ln2);
    copy_ln_state(&dst->ln3, &src->ln3);
}

static void copy_model_opt_state(Model *dst, const Model *src) {
    copy_p_state(&dst->se, &src->se);
    copy_p_state(&dst->te, &src->te);
    for (int i = 0; i < dst->c.L; i++) copy_el_state(&dst->enc[i], &src->enc[i]);
    for (int i = 0; i < dst->c.L; i++) copy_dl_state(&dst->dec[i], &src->dec[i]);
    copy_p_state(&dst->proj, &src->proj);
    copy_p_state(&dst->proj_b, &src->proj_b);
}

static int cfg_same(const Cfg *a, const Cfg *b) {
    return a->V == b->V && a->T == b->T && a->D == b->D &&
           a->H == b->H && a->F == b->F && a->L == b->L &&
           a->eps == b->eps;
}

int opt_state_save(const Model *m, const Opt *o, const char *path) {
    if (!m || !o || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int ver = OPT_STATE_VERSION;
    int ok = 1;
    ok &= opt_wr(OPT_STATE_MAGIC, 1, 4, f);
    ok &= opt_wr(&ver, sizeof(int), 1, f);
    ok &= opt_wr(&m->c.V, sizeof(int), 1, f);
    ok &= opt_wr(&m->c.T, sizeof(int), 1, f);
    ok &= opt_wr(&m->c.D, sizeof(int), 1, f);
    ok &= opt_wr(&m->c.H, sizeof(int), 1, f);
    ok &= opt_wr(&m->c.F, sizeof(int), 1, f);
    ok &= opt_wr(&m->c.L, sizeof(int), 1, f);
    ok &= opt_wr(&m->c.eps, sizeof(float), 1, f);
    ok &= opt_wr(&o->lr, sizeof(float), 1, f);
    ok &= opt_wr(&o->b1, sizeof(float), 1, f);
    ok &= opt_wr(&o->b2, sizeof(float), 1, f);
    ok &= opt_wr(&o->eps, sizeof(float), 1, f);
    ok &= opt_wr(&o->step, sizeof(int), 1, f);

    ok &= save_p_state(f, &m->se);
    ok &= save_p_state(f, &m->te);
    for (int i = 0; i < m->c.L && ok; i++) ok &= save_el_state(f, &m->enc[i]);
    for (int i = 0; i < m->c.L && ok; i++) ok &= save_dl_state(f, &m->dec[i]);
    ok &= save_p_state(f, &m->proj);
    ok &= save_p_state(f, &m->proj_b);

    fclose(f);
    if (!ok) {
        remove(path);
        return -1;
    }
    return 0;
}

int opt_state_load(Model *m, Opt *o, const char *path) {
    if (!m || !o || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[4];
    int ver;
    Cfg stored;
    Opt loaded;

    if (!opt_rd(magic, 1, 4, f) || memcmp(magic, OPT_STATE_MAGIC, 4) != 0 ||
        !opt_rd(&ver, sizeof(int), 1, f) || ver != OPT_STATE_VERSION ||
        !opt_rd(&stored.V, sizeof(int), 1, f) ||
        !opt_rd(&stored.T, sizeof(int), 1, f) ||
        !opt_rd(&stored.D, sizeof(int), 1, f) ||
        !opt_rd(&stored.H, sizeof(int), 1, f) ||
        !opt_rd(&stored.F, sizeof(int), 1, f) ||
        !opt_rd(&stored.L, sizeof(int), 1, f) ||
        !opt_rd(&stored.eps, sizeof(float), 1, f) ||
        !cfg_same(&stored, &m->c) ||
        !opt_rd(&loaded.lr, sizeof(float), 1, f) ||
        !opt_rd(&loaded.b1, sizeof(float), 1, f) ||
        !opt_rd(&loaded.b2, sizeof(float), 1, f) ||
        !opt_rd(&loaded.eps, sizeof(float), 1, f) ||
        !opt_rd(&loaded.step, sizeof(int), 1, f)) {
        fclose(f);
        return -1;
    }

    Model *tmp = model_new(&m->c);
    if (!tmp) {
        fclose(f);
        return -1;
    }

    int ok = 1;
    ok &= load_p_state(f, &tmp->se);
    ok &= load_p_state(f, &tmp->te);
    for (int i = 0; i < tmp->c.L && ok; i++) ok &= load_el_state(f, &tmp->enc[i]);
    for (int i = 0; i < tmp->c.L && ok; i++) ok &= load_dl_state(f, &tmp->dec[i]);
    ok &= load_p_state(f, &tmp->proj);
    ok &= load_p_state(f, &tmp->proj_b);

    fclose(f);
    if (!ok) {
        model_del(tmp);
        return -1;
    }
    copy_model_opt_state(m, tmp);
    model_del(tmp);
    *o = loaded;
    return 0;
}
