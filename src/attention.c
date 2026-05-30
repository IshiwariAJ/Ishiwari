#include "model.h"
#include <math.h>
#include <string.h>
#include <float.h>

/*
 * Multi-Head Attention forward.
 *
 * x   : query source    (S × D)
 * kv  : key/value source (KS × D)  [== x for self-attention]
 * H   : number of heads, d_k = D/H
 * causal : apply lower-triangular mask (for decoder self-attn)
 *
 * Saves to cache c: Q,K,V,hs,ha,co,ao,xi,ki
 */
void attn_fwd(const Mat *x, const Mat *kv,
              const AW *w, int H, int causal, AC *c) {
    int S  = x->r, KS = kv->r, D = x->c;
    int dk = D / H;
    c->S = S; c->KS = KS;

    /* save inputs */
    memcpy(c->xi.d, x->d,  (size_t)S *D*sizeof(float));
    memcpy(c->ki.d, kv->d, (size_t)KS*D*sizeof(float));

    /* Q = x @ Wq + bq  (S×D) */
    {   Mat Wq = {w->Wq.w, D, D};
        mm(x, &Wq, &c->Q); add_bias(&c->Q, w->bq.w); }
    /* K = kv @ Wk + bk  (KS×D) */
    {   Mat Wk = {w->Wk.w, D, D};
        /* K needs to be KS×D, but c->K is allocated T×D; use first KS rows */
        Mat Ktmp = {c->K.d, KS, D};
        mm(kv, &Wk, &Ktmp); add_bias(&Ktmp, w->bk.w); }
    /* V = kv @ Wv + bv  (KS×D) */
    {   Mat Wv = {w->Wv.w, D, D};
        Mat Vtmp = {c->V.d, KS, D};
        mm(kv, &Wv, &Vtmp); add_bias(&Vtmp, w->bv.w); }

    /* per-head: scores → softmax → attn × V → write to co */
    float scale = 1.f / sqrtf((float)dk);
    mat_zero(&c->co);

    for (int h = 0; h < H; h++) {
        int off = h * dk;
        float *hs_h = c->hs + h*S*KS;
        float *ha_h = c->ha + h*S*KS;

        /* scores_h = Q_h @ K_h^T / sqrt(dk)  (S × KS) */
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < KS; j++) {
                float dot = 0.f;
                for (int d = 0; d < dk; d++)
                    dot += c->Q.d[i*D+off+d] * c->K.d[j*D+off+d];
                hs_h[i*KS+j] = dot * scale;
            }
        }

        /* causal mask: mask positions j > i with -inf */
        if (causal) {
            for (int i = 0; i < S; i++)
                for (int j = i+1; j < KS; j++)
                    hs_h[i*KS+j] = -FLT_MAX / 2.f;
        }

        /* softmax over KS dimension */
        for (int i = 0; i < S; i++) {
            float *row = hs_h + i*KS;
            float *out = ha_h + i*KS;
            float mx = row[0];
            for (int j = 1; j < KS; j++) if (row[j]>mx) mx=row[j];
            float s = 0.f;
            for (int j = 0; j < KS; j++) { out[j]=expf(row[j]-mx); s+=out[j]; }
            for (int j = 0; j < KS; j++) out[j] /= s;
        }

        /* head_out = attn_h @ V_h  (S × dk), write into co[:, off:off+dk] */
        for (int i = 0; i < S; i++) {
            for (int d = 0; d < dk; d++) {
                float sum = 0.f;
                for (int j = 0; j < KS; j++)
                    sum += ha_h[i*KS+j] * c->V.d[j*D+off+d];
                c->co.d[i*D+off+d] = sum;
            }
        }
    }

    /* out = co @ Wo + bo  (S×D) */
    {   Mat Wo = {w->Wo.w, D, D};
        Mat co_s = {c->co.d, S, D};
        Mat ao_s = {c->ao.d, S, D};
        mm(&co_s, &Wo, &ao_s);
        add_bias(&ao_s, w->bo.w); }
}

/*
 * Multi-Head Attention backward.
 *
 * dao : gradient of ao  (S×D)
 * dx  : accumulated gradient for x  (S×D)
 * dkv : accumulated gradient for kv (KS×D)
 * dw  : accumulated weight gradients
 */
void attn_bwd(const AW *w, int H, int causal, AC *c,
              const Mat *dao, Mat *dx, Mat *dkv, AW *dw) {
    int S = c->S, KS = c->KS, D = c->xi.c;
    int dk = D / H;
    float scale = 1.f / sqrtf((float)dk);

    /* -- backward through Wo -- */
    /* dco = dao @ Wo^T  (S×D) */
    Mat dco = mat_new(S, D);
    {   Mat WoT = {w->Wo.w,  D, D};
        Mat dao_s = {(float*)dao->d, S, D};
        mm_bt_add(&dao_s, &WoT, &dco);  /* dco = dao @ Wo^T */
        /* dWo += co^T @ dao */
        Mat Wo_g = {dw->Wo.g, D, D};
        Mat co_s = {c->co.d, S, D};
        mm_at_add(&co_s, &dao_s, &Wo_g);
        /* dbo += sum_rows(dao) */
        bias_bwd(dao, dw->bo.g);
    }

    /* dQ, dK, dV accumulators */
    Mat dQ = mat_new(S, D);
    Mat dK = mat_new(KS, D);
    Mat dV = mat_new(KS, D);

    for (int h = 0; h < H; h++) {
        int off = h * dk;
        float *ha_h = c->ha + h*S*KS;

        /* d_head_out_h = dco[:, off:off+dk]  (S×dk) */
        /* dV_h += attn_h^T @ d_head_out_h */
        for (int j = 0; j < KS; j++)
            for (int d = 0; d < dk; d++) {
                float s = 0.f;
                for (int i = 0; i < S; i++)
                    s += ha_h[i*KS+j] * dco.d[i*D+off+d];
                dV.d[j*D+off+d] += s;
            }

        /* dattn_h = d_head_out_h @ V_h^T  (S×KS) */
        float *dattn_h = (float*)calloc((size_t)S*KS, sizeof(float));
        for (int i = 0; i < S; i++)
            for (int j = 0; j < KS; j++) {
                float s = 0.f;
                for (int d = 0; d < dk; d++)
                    s += dco.d[i*D+off+d] * c->V.d[j*D+off+d];
                dattn_h[i*KS+j] = s;
            }

        /* dscores_h: softmax backward (S×KS) */
        float *dscores_h = (float*)calloc((size_t)S*KS, sizeof(float));
        for (int i = 0; i < S; i++) {
            float *yi  = ha_h     + i*KS;
            float *dyi = dattn_h  + i*KS;
            float *dxi = dscores_h+ i*KS;
            float dot = 0.f;
            for (int j = 0; j < KS; j++) dot += yi[j]*dyi[j];
            for (int j = 0; j < KS; j++) {
                dxi[j] = yi[j]*(dyi[j]-dot);
                /* causal mask: masked positions have 0 gradient */
                if (causal && j >= i+1) dxi[j] = 0.f;
            }
        }

        /* scale */
        for (int i = 0; i < S*KS; i++) dscores_h[i] *= scale;

        /* dQ_h += dscores_h @ K_h  (S×dk) */
        for (int i = 0; i < S; i++)
            for (int d = 0; d < dk; d++) {
                float s = 0.f;
                for (int j = 0; j < KS; j++)
                    s += dscores_h[i*KS+j] * c->K.d[j*D+off+d];
                dQ.d[i*D+off+d] += s;
            }
        /* dK_h += dscores_h^T @ Q_h  (KS×dk) */
        for (int j = 0; j < KS; j++)
            for (int d = 0; d < dk; d++) {
                float s = 0.f;
                for (int i = 0; i < S; i++)
                    s += dscores_h[i*KS+j] * c->Q.d[i*D+off+d];
                dK.d[j*D+off+d] += s;
            }
        free(dattn_h); free(dscores_h);
    }
    mat_del(&dco);

    /* -- backward through Wq/Wk/Wv -- */
    /* dWq += xi^T @ dQ,  dx += dQ @ Wq^T */
    {   Mat Wq = {w->Wq.w, D, D};
        Mat dWq = {dw->Wq.g, D, D};
        Mat xi_s = {c->xi.d, S, D};
        Mat dQ_s = {dQ.d, S, D};
        mm_at_add(&xi_s, &dQ_s, &dWq);
        bias_bwd(&dQ_s, dw->bq.g);
        mm_bt_add(&dQ_s, &Wq, dx);   /* dx (S×D) += dQ @ Wq^T */
    }
    {   Mat Wk = {w->Wk.w, D, D};
        Mat dWk = {dw->Wk.g, D, D};
        Mat ki_s = {c->ki.d, KS, D};
        Mat dK_s = {dK.d, KS, D};
        mm_at_add(&ki_s, &dK_s, &dWk);
        bias_bwd(&dK_s, dw->bk.g);
        mm_bt_add(&dK_s, &Wk, dkv);  /* dkv (KS×D) += dK @ Wk^T */
    }
    {   Mat Wv = {w->Wv.w, D, D};
        Mat dWv = {dw->Wv.g, D, D};
        Mat ki_s = {c->ki.d, KS, D};
        Mat dV_s = {dV.d, KS, D};
        mm_at_add(&ki_s, &dV_s, &dWv);
        bias_bwd(&dV_s, dw->bv.g);
        mm_bt_add(&dV_s, &Wv, dkv);  /* dkv += dV @ Wv^T */
    }
    mat_del(&dQ); mat_del(&dK); mat_del(&dV);
}
