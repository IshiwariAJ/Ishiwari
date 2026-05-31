#include "model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* --- DecodeCache--- */

DecodeCache *decode_cache_new(const Cfg *cfg) {
    DecodeCache *c = (DecodeCache*)calloc(1, sizeof(DecodeCache));
    c->L       = cfg->L;
    c->max_len = cfg->T;
    c->layers  = (DLayerKV*)calloc(cfg->L, sizeof(DLayerKV));
    int D = cfg->D, T = cfg->T;
    for (int l = 0; l < cfg->L; l++) {
        c->layers[l].self.K  = mat_new(T, D);
        c->layers[l].self.V  = mat_new(T, D);
        c->layers[l].self.len = 0;
        c->layers[l].cross.K = mat_new(T, D);
        c->layers[l].cross.V = mat_new(T, D);
        c->layers[l].cross.len = 0;
    }
    return c;
}

void decode_cache_del(DecodeCache *c) {
    for (int l = 0; l < c->L; l++) {
        mat_del(&c->layers[l].self.K);
        mat_del(&c->layers[l].self.V);
        mat_del(&c->layers[l].cross.K);
        mat_del(&c->layers[l].cross.V);
    }
    free(c->layers);
    free(c);
}

void decode_cache_reset(DecodeCache *c) {
    for (int l = 0; l < c->L; l++)
        c->layers[l].self.len = 0;
}

/* --- model_encode--- */

int model_encode(Model *m, const int *src, int SL, EC **ec, Mat *enc_out) {
    int D = m->c.D, V = m->c.V, T = m->c.T;

    /* input validation */
    if (SL <= 0 || SL > T) {
        fprintf(stderr, "model_encode: SL=%d out of range (0,%d]\n", SL, T);
        return -1;
    }
    for (int i = 0; i < SL; i++) {
        if (src[i] < 0 || src[i] >= V) {
            fprintf(stderr, "model_encode: src[%d]=%d out of range [0,%d)\n", i, src[i], V);
            return -1;
        }
    }

    Mat enc_in = mat_new(SL, D);
    for (int t = 0; t < SL; t++) {
        float *dst = enc_in.d + t*D;
        float *emb = m->se.w  + src[t]*D;
        float *pos = m->pos   + t*D;
        for (int d = 0; d < D; d++) dst[d] = emb[d] + pos[d];
    }
    Mat cur = enc_in;
    for (int l = 0; l < m->c.L; l++) {
        el_fwd(&cur, &m->enc[l], ec[l], &m->c, 0);
        cur.d = ec[l]->xo.d;
        cur.r = SL; cur.c = D;
    }
    memcpy(enc_out->d, cur.d, (size_t)SL * D * sizeof(float));
    mat_del(&enc_in);

    /* update output shape to actual size */
    enc_out->r = SL;
    enc_out->c = D;
    return 0;
}

/* --- decode_cache_precompute_cross--- */

int decode_cache_precompute_cross(DecodeCache *c, Model *m,
                                   const Mat *enc_out) {
    int D  = m->c.D;
    int SL = enc_out->r;

    /* input validation */
    if (SL <= 0 || SL > c->max_len) {
        fprintf(stderr, "decode_cache_precompute_cross: SL=%d out of range (0,%d]\n", SL, c->max_len);
        return -1;
    }

    for (int l = 0; l < m->c.L; l++) {
        AW *ca = &m->dec[l].ca;
        Mat Wk  = {ca->Wk.w, D, D};
        Mat Wv  = {ca->Wv.w, D, D};
        Mat K_s = {c->layers[l].cross.K.d, SL, D};
        Mat V_s = {c->layers[l].cross.V.d, SL, D};
        mm(enc_out, &Wk, &K_s); add_bias(&K_s, ca->bk.w);
        mm(enc_out, &Wv, &V_s); add_bias(&V_s, ca->bv.w);
        c->layers[l].cross.len = SL;
    }
    return 0;
}

/* --- single-query multi-head attention--- */
/*
 * q   : (D,)  query for one token
 * K,V : (len x D)  cached keys/values
 * out : (D,)  attention output
 */
static void attn_1q(const float *q, const float *K, const float *V,
                    int len, int D, int H, float *out) {
    int dk = D / H;
    float scale = 1.f / sqrtf((float)dk);
    float *scores = (float*)calloc(len, sizeof(float));

    memset(out, 0, (size_t)D * sizeof(float));
    for (int h = 0; h < H; h++) {
        int off = h * dk;
        for (int j = 0; j < len; j++) {
            float dot = 0.f;
            for (int d = 0; d < dk; d++)
                dot += q[off+d] * K[j*D + off+d];
            scores[j] = dot * scale;
        }
        float mx = scores[0];
        for (int j = 1; j < len; j++) if (scores[j] > mx) mx = scores[j];
        float s = 0.f;
        for (int j = 0; j < len; j++) { scores[j] = expf(scores[j]-mx); s += scores[j]; }
        for (int j = 0; j < len; j++) scores[j] /= s;
        for (int d = 0; d < dk; d++) {
            float sum = 0.f;
            for (int j = 0; j < len; j++) sum += scores[j] * V[j*D + off+d];
            out[off+d] = sum;
        }
    }
    free(scores);
}

/* --- model_decode_step--- */
/*
 * Runs one decoder step using KV cache.
 * token        : current input token id
 * pos          : position index (0-based, same as tgt length so far)
 * kv           : KV cache (self updated here; cross must be precomputed)
 * logits_1xV   : output logits mat, must be pre-allocated as (1 x V)
 *
 * Decoder flow (Post-LN, matches dl_fwd):
 *   sa_proj = self_attn_proj(h)    — causal: attends to all cached K/V
 *   h = LN1(h + sa_proj)
 *   ca_proj = cross_attn_proj(h)
 *   h = LN2(h + ca_proj)
 *   h = LN3(h + FFN(h))
 */
int model_decode_step(Model *m, int token, int pos,
                      DecodeCache *kv, Mat *logits_1xV) {
    if (pos < 0 || pos >= m->c.T) {
        fprintf(stderr, "model_decode_step: pos=%d out of range [0,%d)\n", pos, m->c.T);
        return -1;
    }
    if (token < 0 || token >= m->c.V) {
        fprintf(stderr, "model_decode_step: token=%d out of range [0,%d)\n", token, m->c.V);
        return -1;
    }
    for (int l = 0; l < kv->L; l++) {
        if (kv->layers[l].self.len >= kv->max_len) {
            fprintf(stderr, "model_decode_step: self KV cache full at layer %d\n", l);
            return -1;
        }
        if (kv->layers[l].cross.len <= 0) {
            fprintf(stderr, "model_decode_step: cross KV not precomputed at layer %d\n", l);
            return -1;
        }
    }
    int D = m->c.D, V = m->c.V, H = m->c.H, F = m->c.F;

    float *h   = (float*)calloc(D, sizeof(float));
    float *tmp = (float*)calloc(D, sizeof(float));
    float *q   = (float*)calloc(D, sizeof(float));
    float *ao  = (float*)calloc(D, sizeof(float));
    float *ffh = (float*)calloc(F, sizeof(float));
    float mn, vr;

    /* embed token + positional encoding */
    float *emb  = m->te.w + token * D;
    float *penc = m->pos  + pos   * D;
    for (int d = 0; d < D; d++) h[d] = emb[d] + penc[d];

    for (int l = 0; l < m->c.L; l++) {
        DL     *dl  = &m->dec[l];
        DLayerKV *lkv = &kv->layers[l];
        int len = lkv->self.len;

        /* --- Masked self-attention--- */
        /* Q = h @ Wq + bq */
        {   Mat Wq={dl->sa.Wq.w,D,D}; Mat h_m={h,1,D}; Mat q_m={q,1,D};
            mm(&h_m,&Wq,&q_m); add_bias(&q_m,dl->sa.bq.w); }
        /* append K,V for current token to self cache */
        {   Mat Wk={dl->sa.Wk.w,D,D}; Mat Wv={dl->sa.Wv.w,D,D};
            Mat h_m={h,1,D};
            Mat Kr={lkv->self.K.d+len*D,1,D};
            Mat Vr={lkv->self.V.d+len*D,1,D};
            mm(&h_m,&Wk,&Kr); add_bias(&Kr,dl->sa.bk.w);
            mm(&h_m,&Wv,&Vr); add_bias(&Vr,dl->sa.bv.w); }
        lkv->self.len++;
        /* attend to all cached K,V (includes current token) */
        attn_1q(q, lkv->self.K.d, lkv->self.V.d, lkv->self.len, D, H, ao);
        /* tmp = ao @ Wo + bo */
        {   Mat Wo={dl->sa.Wo.w,D,D}; Mat ao_m={ao,1,D}; Mat t_m={tmp,1,D};
            mm(&ao_m,&Wo,&t_m); add_bias(&t_m,dl->sa.bo.w); }
        /* h = LN1(h + tmp) */
        {   for(int d=0;d<D;d++) tmp[d]+=h[d];
            Mat r={tmp,1,D}; Mat h_m={h,1,D};
            ln_fwd(&r,&dl->ln1,&h_m,&mn,&vr,m->c.eps); }

        /* --- Cross-attention--- */
        /* Q = h @ Wq_ca + bq_ca */
        {   Mat Wq={dl->ca.Wq.w,D,D}; Mat h_m={h,1,D}; Mat q_m={q,1,D};
            mm(&h_m,&Wq,&q_m); add_bias(&q_m,dl->ca.bq.w); }
        attn_1q(q, lkv->cross.K.d, lkv->cross.V.d, lkv->cross.len, D, H, ao);
        /* tmp = ao @ Wo_ca + bo_ca */
        {   Mat Wo={dl->ca.Wo.w,D,D}; Mat ao_m={ao,1,D}; Mat t_m={tmp,1,D};
            mm(&ao_m,&Wo,&t_m); add_bias(&t_m,dl->ca.bo.w); }
        /* h = LN2(h + tmp) */
        {   for(int d=0;d<D;d++) tmp[d]+=h[d];
            Mat r={tmp,1,D}; Mat h_m={h,1,D};
            ln_fwd(&r,&dl->ln2,&h_m,&mn,&vr,m->c.eps); }

        /* --- FFN--- */
        {   Mat W1={dl->ff.W1.w,D,F}; Mat h_m={h,1,D}; Mat ffh_m={ffh,1,F};
            mm(&h_m,&W1,&ffh_m); add_bias(&ffh_m,dl->ff.b1.w);
            gelu_fwd(&ffh_m,&ffh_m); }
        {   Mat W2={dl->ff.W2.w,F,D}; Mat ffh_m={ffh,1,F}; Mat t_m={tmp,1,D};
            mm(&ffh_m,&W2,&t_m); add_bias(&t_m,dl->ff.b2.w); }
        /* h = LN3(h + tmp) */
        {   for(int d=0;d<D;d++) tmp[d]+=h[d];
            Mat r={tmp,1,D}; Mat h_m={h,1,D};
            ln_fwd(&r,&dl->ln3,&h_m,&mn,&vr,m->c.eps); }
    }

    /* output projection: logits = h @ proj + proj_b */
    {   Mat proj={m->proj.w,D,V}; Mat h_m={h,1,D};
        mm(&h_m,&proj,logits_1xV);
        add_bias(logits_1xV,m->proj_b.w); }

    free(h); free(tmp); free(q); free(ao); free(ffh);
    return 0;
}
