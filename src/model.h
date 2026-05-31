#ifndef MODEL_H
#define MODEL_H
#include "matrix.h"

/* --- Config--- */
typedef struct {
    int V;      /* vocab size */
    int T;      /* max seq len */
    int D;      /* d_model */
    int H;      /* n_heads (D must be divisible by H) */
    int F;      /* d_ff */
    int L;      /* n_layers */
    float eps;  /* layer-norm epsilon */
} Cfg;

/* --- Trainable parameter--- */
typedef struct { float *w, *g, *m, *v; int n; } P;
P    p_new(int n);
void p_del(P *p);
void p_zg(P *p);

/* --- Layer-norm: scale (γ) + shift (β)--- */
typedef struct { P scale, shift; } LN;
LN   ln_new(int d);
void ln_del(LN *ln);
void ln_zg(LN *ln);

/* --- Attention weight block--- */
typedef struct { P Wq,Wk,Wv,Wo, bq,bk,bv,bo; } AW;
AW   aw_new(int D);
void aw_del(AW *a);
void aw_zg(AW *a);

/* --- FFN weight block--- */
typedef struct { P W1,b1,W2,b2; } FW;
FW   fw_new(int D, int F);
void fw_del(FW *f);
void fw_zg(FW *f);

/* --- Encoder / Decoder layers--- */
typedef struct { AW sa;      FW ff; LN ln1,ln2;     } EL;
typedef struct { AW sa,ca;   FW ff; LN ln1,ln2,ln3; } DL;

EL el_new(int D, int F);
DL dl_new(int D, int F);
void el_del(EL *e);
void dl_del(DL *d);
void el_zg(EL *e);
void dl_zg(DL *d);

/* --- Model--- */
typedef struct {
    Cfg c;
    P   se, te;   /* src/tgt embeddings (V×D) */
    float *pos;   /* sinusoidal positional encoding (T×D) */
    EL *enc;      /* [L] encoder layers */
    DL *dec;      /* [L] decoder layers */
    P   proj;     /* output projection (D×V) */
    P   proj_b;   /* output bias (V,) */
} Model;

Model *model_new(const Cfg *cfg);
void   model_del(Model *m);
void   model_zg (Model *m);
void   model_init(Model *m);

/* --- Attention cache--- */
typedef struct {
    Mat Q, K, V;          /* (T×D) projections */
    float *hs, *ha;       /* head scores / attn weights: [H×T×T] */
    Mat co;               /* concatenated head outputs (T×D) */
    Mat ao;               /* after Wo projection (T×D) */
    Mat xi, ki;           /* saved inputs: query src, key-value src */
    int S, KS;            /* actual seq lengths used this forward */
} AC;

/* --- Encoder layer cache--- */
typedef struct {
    Mat xi;               /* layer input */
    AC  sa;               /* self-attention */
    Mat r1;               /* residual after sa: xi + sa.ao */
    Mat x1;               /* after LN1 */
    float *mn1, *vr1;     /* LN1 stats (S,) */
    Mat fh;               /* FFN hidden (T×F) */
    Mat fo;               /* FFN output (T×D) */
    Mat r2;               /* residual after FFN */
    Mat xo;               /* layer output after LN2 */
    float *mn2, *vr2;
} EC;

/* --- Decoder layer cache--- */
typedef struct {
    Mat xi;
    AC  sa;               /* masked self-attention */
    Mat r1; Mat x1;
    float *mn1, *vr1;
    AC  ca;               /* cross-attention */
    Mat r2; Mat x2;
    float *mn2, *vr2;
    Mat fh; Mat fo;
    Mat r3; Mat xo;
    float *mn3, *vr3;
} DC;

AC *ac_new(int T, int D, int H);
EC *ec_new(int T, int D, int F, int H);
DC *dc_new(int T, int D, int F, int H);
void ac_del(AC *a);
void ec_del(EC *e);
void dc_del(DC *d);

/* --- Layer-norm fwd/bwd--- */
void ln_fwd(const Mat *x, const LN *w, Mat *y,
            float *mean, float *var, float eps);
/* dx +=, dw accumulated */
void ln_bwd(const Mat *x, const LN *w,
            const float *mean, const float *var,
            const Mat *dy, Mat *dx, LN *dw, float eps);

/* --- Attention fwd/bwd--- */
/* kv == x for self-attention */
void attn_fwd(const Mat *x, const Mat *kv,
              const AW *w, int H, int causal, AC *c);
/* dao: gradient of output ao; dx/dkv accumulated */
void attn_bwd(const AW *w, int H, int causal, AC *c,
              const Mat *dao, Mat *dx, Mat *dkv, AW *dw);

/* --- FFN fwd/bwd--- */
void ffn_fwd(const Mat *x, const FW *w, Mat *h, Mat *out);
/* dx accumulated */
void ffn_bwd(const Mat *x, const FW *w, const Mat *h,
             const Mat *dout, Mat *dx, FW *dw);

/* --- Encoder/Decoder layer fwd/bwd--- */
/* causal=0 for bidirectional encoder, causal=1 for causal LM */
void el_fwd(const Mat *x, const EL *w, EC *c, const Cfg *cfg, int causal);
void dl_fwd(const Mat *x, const Mat *enc, const DL *w, DC *c, const Cfg *cfg);
/* dx/d_enc accumulated */
void el_bwd(const EL *w, EC *c, const Cfg *cfg,
            const Mat *dy, EL *dw, Mat *dx, int causal);
void dl_bwd(const DL *w, DC *c, const Cfg *cfg,
            const Mat *dy, DL *dw, Mat *dx, Mat *d_enc);

/* --- Full model--- */
/* Returns 0 on success, -1 on invalid input (SL/TL out of range, bad token id). */
int   model_fwd(Model *m,
                const int *src, int SL,
                const int *tgt, int TL,
                EC **ec, DC **dc,
                Mat *enc_out, Mat *dec_out, Mat *logits);

/* Returns loss on success, -1.0f on invalid input. */
float model_loss_bwd(Model *m,
                     const int *src, int SL,
                     const int *tgt, int TL,
                     const int *lbl,          /* target labels (TL,) */
                     EC **ec, DC **dc,
                     const Mat *enc_out, const Mat *dec_out,
                     const Mat *logits);

/* --- Adam optimizer--- */
typedef struct { float lr,b1,b2,eps; int step; } Opt;
void opt_step(Model *m, Opt *o);

/* --- KV cache (inference only)--- */
typedef struct { Mat K, V; int len; } KV;
typedef struct { KV self, cross; } DLayerKV;
typedef struct { DLayerKV *layers; int max_len, L; } DecodeCache;

DecodeCache *decode_cache_new(const Cfg *cfg);
void         decode_cache_del(DecodeCache *c);
void         decode_cache_reset(DecodeCache *c);

/* Returns 0 on success, -1 on invalid input. */
int  model_encode(Model *m, const int *src, int SL, EC **ec, Mat *enc_out);
/* Returns 0 on success, -1 if enc_out rows exceed cache max_len. */
int  decode_cache_precompute_cross(DecodeCache *c, Model *m, const Mat *enc_out);
/* Returns 0 on success, -1 on invalid args or cache overflow. */
int  model_decode_step(Model *m, int token, int pos,
                       DecodeCache *kv, Mat *logits_1xV);

/* --- Serialization--- */
/* Save/load all learnable weights. Format carries a version + Cfg header,
 * so model_load reconstructs the model with matching shapes.
 * model_save: 0 on success, -1 on failure.
 * model_load: returns a new Model (caller frees with model_del), NULL on failure. */
int    model_save(const Model *m, const char *path);
Model *model_load(const char *path);

#endif
