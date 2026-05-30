#include "model.h"
#include <math.h>

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
