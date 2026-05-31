#include "model.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * Full model forward pass.
 *   src[SL]  : source token ids
 *   tgt[TL]  : target token ids (decoder input, teacher forcing)
 *   ec[L]    : encoder caches  (one per layer)
 *   dc[L]    : decoder caches  (one per layer)
 *   enc_out  : (SL × D)  final encoder output
 *   dec_out  : (TL × D)  final decoder output
 *   logits   : (TL × V)  pre-softmax output
 * Returns 0 on success, -1 on invalid input.
 */
int model_fwd(Model *m,
               const int *src, int SL,
               const int *tgt, int TL,
               EC **ec, DC **dc,
               Mat *enc_out, Mat *dec_out, Mat *logits) {
    int D = m->c.D, V = m->c.V, T = m->c.T;

    /* input validation */
    if (SL <= 0 || SL > T) {
        fprintf(stderr, "model_fwd: SL=%d out of range (0,%d]\n", SL, T);
        return -1;
    }
    if (TL <= 0 || TL > T) {
        fprintf(stderr, "model_fwd: TL=%d out of range (0,%d]\n", TL, T);
        return -1;
    }
    for (int i = 0; i < SL; i++) {
        if (src[i] < 0 || src[i] >= V) {
            fprintf(stderr, "model_fwd: src[%d]=%d out of range [0,%d)\n", i, src[i], V);
            return -1;
        }
    }
    for (int i = 0; i < TL; i++) {
        if (tgt[i] < 0 || tgt[i] >= V) {
            fprintf(stderr, "model_fwd: tgt[%d]=%d out of range [0,%d)\n", i, tgt[i], V);
            return -1;
        }
    }

    /* -- Encoder -- */
    /* embed src tokens + positional encoding */
    Mat enc_in = mat_new(SL, D);
    for (int t = 0; t < SL; t++) {
        int tok = src[t];
        float *dst = enc_in.d + t*D;
        float *emb = m->se.w + tok*D;
        float *pos = m->pos  + t*D;
        for (int d = 0; d < D; d++) dst[d] = emb[d] + pos[d];
    }

    /* pass through encoder layers */
    Mat enc_cur = enc_in;
    for (int l = 0; l < m->c.L; l++) {
        el_fwd(&enc_cur, &m->enc[l], ec[l], &m->c, 0);
        enc_cur.d = ec[l]->xo.d;  /* point to layer output */
        enc_cur.r = SL; enc_cur.c = D;
    }
    /* copy final encoder output */
    memcpy(enc_out->d, enc_cur.d, (size_t)SL*D*sizeof(float));
    mat_del(&enc_in);

    /* -- Decoder -- */
    Mat dec_in = mat_new(TL, D);
    for (int t = 0; t < TL; t++) {
        int tok = tgt[t];
        float *dst = dec_in.d + t*D;
        float *emb = m->te.w + tok*D;
        float *pos = m->pos  + t*D;
        for (int d = 0; d < D; d++) dst[d] = emb[d] + pos[d];
    }

    Mat dec_cur = dec_in;
    for (int l = 0; l < m->c.L; l++) {
        dl_fwd(&dec_cur, enc_out, &m->dec[l], dc[l], &m->c);
        dec_cur.d = dc[l]->xo.d;
        dec_cur.r = TL; dec_cur.c = D;
    }
    memcpy(dec_out->d, dec_cur.d, (size_t)TL*D*sizeof(float));
    mat_del(&dec_in);

    /* -- Output projection: logits = dec_out @ proj + proj_b  (TL × V) */
    Mat proj = {m->proj.w, D, V};
    mm(dec_out, &proj, logits);
    add_bias(logits, m->proj_b.w);

    /* update output shapes to actual sizes */
    enc_out->r = SL; enc_out->c = D;
    dec_out->r = TL; dec_out->c = D;
    logits->r  = TL; logits->c  = V;
    return 0;
}

/*
 * Compute cross-entropy loss and run full backward pass.
 * lbl[TL] : ground-truth token ids for each decoder output position
 * Returns average cross-entropy loss, or -1.0f on invalid input.
 */
float model_loss_bwd(Model *m,
                     const int *src, int SL,
                     const int *tgt, int TL,
                     const int *lbl,
                     EC **ec, DC **dc,
                     const Mat *enc_out, const Mat *dec_out,
                     const Mat *logits) {
    int D = m->c.D, V = m->c.V, L = m->c.L, T = m->c.T;
    (void)enc_out;

    /* input validation */
    if (SL <= 0 || SL > T) {
        fprintf(stderr, "model_loss_bwd: SL=%d out of range (0,%d]\n", SL, T);
        return -1.0f;
    }
    if (TL <= 0 || TL > T) {
        fprintf(stderr, "model_loss_bwd: TL=%d out of range (0,%d]\n", TL, T);
        return -1.0f;
    }
    for (int i = 0; i < SL; i++) {
        if (src[i] < 0 || src[i] >= V) {
            fprintf(stderr, "model_loss_bwd: src[%d]=%d out of range [0,%d)\n", i, src[i], V);
            return -1.0f;
        }
    }
    for (int i = 0; i < TL; i++) {
        if (tgt[i] < 0 || tgt[i] >= V) {
            fprintf(stderr, "model_loss_bwd: tgt[%d]=%d out of range [0,%d)\n", i, tgt[i], V);
            return -1.0f;
        }
    }
    for (int i = 0; i < TL; i++) {
        if (lbl[i] < 0 || lbl[i] >= V) {
            fprintf(stderr, "model_loss_bwd: lbl[%d]=%d out of range [0,%d)\n", i, lbl[i], V);
            return -1.0f;
        }
    }

    /* softmax + cross-entropy loss, compute d_logits */
    Mat d_logits = mat_new(TL, V);
    float loss = 0.f;
    for (int t = 0; t < TL; t++) {
        const float *row = logits->d + t*V;
        float mx = row[0];
        for (int j = 1; j < V; j++) if (row[j]>mx) mx=row[j];
        float s = 0.f;
        float *dl = d_logits.d + t*V;
        for (int j = 0; j < V; j++) { dl[j]=expf(row[j]-mx); s+=dl[j]; }
        loss -= logf(dl[lbl[t]] / s);
        for (int j = 0; j < V; j++) dl[j] = dl[j]/s - (j==lbl[t] ? 1.f : 0.f);
    }
    loss /= TL;
    /* scale gradients by 1/TL */
    for (int i = 0; i < TL*V; i++) d_logits.d[i] /= TL;

    /* -- d_dec_out = d_logits @ proj^T  (TL×D) */
    Mat d_dec_out = mat_new(TL, D);
    {   Mat proj = {m->proj.w, D, V};
        mm_bt_add(&d_logits, &proj, &d_dec_out);
        /* dproj += dec_out^T @ d_logits */
        Mat dproj = {m->proj.g, D, V};
        mm_at_add(dec_out, &d_logits, &dproj);
        /* dproj_b */
        bias_bwd(&d_logits, m->proj_b.g);
    }
    mat_del(&d_logits);

    /* -- Decoder layers backward -- */
    Mat d_enc_acc = mat_new(SL, D);   /* accumulated grad from all dec layers */

    Mat d_dec_cur = d_dec_out;
    /* iterate backward through decoder layers */
    for (int l = L-1; l >= 0; l--) {
        Mat d_dec_prev = mat_new(TL, D);
        dl_bwd(&m->dec[l], dc[l], &m->c,
               &d_dec_cur, &m->dec[l], &d_dec_prev, &d_enc_acc);
        if (l < L-1) mat_del(&d_dec_cur);
        d_dec_cur = d_dec_prev;
    }

    /* gradient for target embeddings */
    for (int t = 0; t < TL; t++) {
        int tok = tgt[t];
        float *ge = m->te.g + tok*D;
        float *gd = d_dec_cur.d + t*D;
        for (int d = 0; d < D; d++) ge[d] += gd[d];
    }
    mat_del(&d_dec_cur);
    mat_del(&d_dec_out);

    /* -- Encoder layers backward -- */
    Mat d_enc_cur = d_enc_acc;
    for (int l = L-1; l >= 0; l--) {
        Mat d_enc_prev = mat_new(SL, D);
        el_bwd(&m->enc[l], ec[l], &m->c,
               &d_enc_cur, &m->enc[l], &d_enc_prev, 0);
        if (l < L-1) mat_del(&d_enc_cur);
        d_enc_cur = d_enc_prev;
    }

    /* gradient for source embeddings */
    for (int t = 0; t < SL; t++) {
        int tok = src[t];
        float *ge = m->se.g + tok*D;
        float *gd = d_enc_cur.d + t*D;
        for (int d = 0; d < D; d++) ge[d] += gd[d];
    }
    mat_del(&d_enc_cur);
    mat_del(&d_enc_acc);

    return loss;
}
