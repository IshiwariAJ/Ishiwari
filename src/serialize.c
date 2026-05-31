#include "model.h"
#include "serialize_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
 * Upper bounds for config values to prevent huge allocations.
 * These are conservative limits for a small experimental project.
 * Total parameter count is also checked to prevent large L bypassing individual limits.
 */
#define MAX_V 100000    /* max vocab size: 100K */
#define MAX_T 4096      /* max seq len: 4K */
#define MAX_D 2048      /* max d_model: 2K */
#define MAX_H 64        /* max heads */
#define MAX_F 8192      /* max d_ff: 8K */
#define MAX_L 64        /* max layers */
#define MAX_PARAMS ((int64_t)500 * 1000 * 1000)  /* max 500M parameters */

/*
 * Binary model serialization.
 *
 * File layout:
 *   magic   : 4 bytes "MYLM"
 *   version : int32
 *   cfg     : V,T,D,H,F,L (int32 each), eps (float32)
 *   params  : se, te, [enc layers], [dec layers], proj, proj_b
 *
 * Only learnable weights (P.w) are stored — not optimizer moments (m,v)
 * nor the sinusoidal positional table (pos), which model_new regenerates.
 * Loading reconstructs the model from the stored Cfg, so shapes always match.
 */

#define MODEL_MAGIC   "MYLM"
#define MODEL_VERSION 1

/* --- low-level helpers--- */

static int wr(const void *p, size_t sz, size_t n, FILE *f) {
    return fwrite(p, sz, n, f) == n;
}
static int rd(void *p, size_t sz, size_t n, FILE *f) {
    return fread(p, sz, n, f) == n;
}

static int save_p(FILE *f, const P *p) { return wr(p->w, sizeof(float), p->n, f); }
static int load_p(FILE *f, P *p)       { return rd(p->w, sizeof(float), p->n, f); }

static int save_aw(FILE *f, const AW *a) {
    return save_p(f,&a->Wq)&&save_p(f,&a->Wk)&&save_p(f,&a->Wv)&&save_p(f,&a->Wo)
        && save_p(f,&a->bq)&&save_p(f,&a->bk)&&save_p(f,&a->bv)&&save_p(f,&a->bo);
}
static int load_aw(FILE *f, AW *a) {
    return load_p(f,&a->Wq)&&load_p(f,&a->Wk)&&load_p(f,&a->Wv)&&load_p(f,&a->Wo)
        && load_p(f,&a->bq)&&load_p(f,&a->bk)&&load_p(f,&a->bv)&&load_p(f,&a->bo);
}

static int save_fw(FILE *f, const FW *w) {
    return save_p(f,&w->W1)&&save_p(f,&w->b1)&&save_p(f,&w->W2)&&save_p(f,&w->b2);
}
static int load_fw(FILE *f, FW *w) {
    return load_p(f,&w->W1)&&load_p(f,&w->b1)&&load_p(f,&w->W2)&&load_p(f,&w->b2);
}

static int save_ln(FILE *f, const LN *l) { return save_p(f,&l->scale)&&save_p(f,&l->shift); }
static int load_ln(FILE *f, LN *l)       { return load_p(f,&l->scale)&&load_p(f,&l->shift); }

static int save_el(FILE *f, const EL *e) {
    return save_aw(f,&e->sa)&&save_fw(f,&e->ff)&&save_ln(f,&e->ln1)&&save_ln(f,&e->ln2);
}
static int load_el(FILE *f, EL *e) {
    return load_aw(f,&e->sa)&&load_fw(f,&e->ff)&&load_ln(f,&e->ln1)&&load_ln(f,&e->ln2);
}

static int save_dl(FILE *f, const DL *d) {
    return save_aw(f,&d->sa)&&save_aw(f,&d->ca)&&save_fw(f,&d->ff)
        && save_ln(f,&d->ln1)&&save_ln(f,&d->ln2)&&save_ln(f,&d->ln3);
}
static int load_dl(FILE *f, DL *d) {
    return load_aw(f,&d->sa)&&load_aw(f,&d->ca)&&load_fw(f,&d->ff)
        && load_ln(f,&d->ln1)&&load_ln(f,&d->ln2)&&load_ln(f,&d->ln3);
}

/* --- internal save/load to FILE* --- */

static int model_save_to_fp(const Model *m, FILE *f) {
    int ok = 1;
    int ver = MODEL_VERSION;
    const Cfg *c = &m->c;
    ok &= wr(MODEL_MAGIC, 1, 4, f);
    ok &= wr(&ver, sizeof(int), 1, f);
    ok &= wr(&c->V, sizeof(int), 1, f);
    ok &= wr(&c->T, sizeof(int), 1, f);
    ok &= wr(&c->D, sizeof(int), 1, f);
    ok &= wr(&c->H, sizeof(int), 1, f);
    ok &= wr(&c->F, sizeof(int), 1, f);
    ok &= wr(&c->L, sizeof(int), 1, f);
    ok &= wr(&c->eps, sizeof(float), 1, f);

    ok &= save_p(f, &m->se);
    ok &= save_p(f, &m->te);
    for (int l = 0; l < c->L && ok; l++) ok &= save_el(f, &m->enc[l]);
    for (int l = 0; l < c->L && ok; l++) ok &= save_dl(f, &m->dec[l]);
    ok &= save_p(f, &m->proj);
    ok &= save_p(f, &m->proj_b);

    return ok ? 0 : -1;
}

static Model *model_load_from_fp(FILE *f) {
    char magic[4];
    int ver;
    Cfg c;
    if (!rd(magic, 1, 4, f) || memcmp(magic, MODEL_MAGIC, 4) != 0) return NULL;
    if (!rd(&ver, sizeof(int), 1, f) || ver != MODEL_VERSION) return NULL;
    if (!rd(&c.V, sizeof(int), 1, f) || !rd(&c.T, sizeof(int), 1, f) ||
        !rd(&c.D, sizeof(int), 1, f) || !rd(&c.H, sizeof(int), 1, f) ||
        !rd(&c.F, sizeof(int), 1, f) || !rd(&c.L, sizeof(int), 1, f) ||
        !rd(&c.eps, sizeof(float), 1, f)) return NULL;

    /* basic sanity on shapes */
    if (c.V<=0 || c.T<=0 || c.D<=0 || c.H<=0 || c.F<=0 || c.L<=0 || c.D % c.H != 0) {
        return NULL;
    }
    /* upper bounds */
    if (c.V>MAX_V || c.T>MAX_T || c.D>MAX_D || c.H>MAX_H || c.F>MAX_F || c.L>MAX_L) {
        return NULL;
    }
    /* total parameter count check */
    {
        int64_t D = c.D, F = c.F, V = c.V, L = c.L;
        int64_t emb_params = 2 * V * D;
        int64_t layer_params = L * (12 * D * D + 4 * D * F + 24 * D);
        int64_t proj_params = D * V + V;
        int64_t total = emb_params + layer_params + proj_params;
        if (total > MAX_PARAMS) return NULL;
    }

    Model *m = model_new(&c);

    int ok = 1;
    ok &= load_p(f, &m->se);
    ok &= load_p(f, &m->te);
    for (int l = 0; l < c.L && ok; l++) ok &= load_el(f, &m->enc[l]);
    for (int l = 0; l < c.L && ok; l++) ok &= load_dl(f, &m->dec[l]);
    ok &= load_p(f, &m->proj);
    ok &= load_p(f, &m->proj_b);

    if (!ok) { model_del(m); return NULL; }
    return m;
}

/* --- public API--- */

int model_save(const Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int result = model_save_to_fp(m, f);
    fclose(f);
    return result;
}

Model *model_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    Model *m = model_load_from_fp(f);
    fclose(f);
    return m;
}

/* --- FILE* API for bundle direct serialization --- */

int model_save_fp(const Model *m, FILE *f) {
    return model_save_to_fp(m, f);
}

Model *model_load_fp(FILE *f) {
    return model_load_from_fp(f);
}
