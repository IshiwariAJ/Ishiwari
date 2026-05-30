#include "event.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

EventSeq *event_seq_new(int capacity) {
    EventSeq *s = (EventSeq*)calloc(1, sizeof(EventSeq));
    s->capacity   = capacity;
    s->modality   = (int*)  calloc(capacity, sizeof(int));
    s->channel    = (int*)  calloc(capacity, sizeof(int));
    s->time_index = (int*)  calloc(capacity, sizeof(int));
    s->value      = (float*)calloc(capacity, sizeof(float));
    s->token_id   = (int*)  calloc(capacity, sizeof(int));
    s->n = 0;
    if (!s->modality || !s->channel || !s->time_index ||
        !s->value    || !s->token_id) {
        fprintf(stderr, "OOM event_seq_new(%d)\n", capacity);
        exit(1);
    }
    for (int i = 0; i < capacity; i++) s->token_id[i] = -1;
    return s;
}

void event_seq_del(EventSeq *s) {
    free(s->modality); free(s->channel); free(s->time_index);
    free(s->value); free(s->token_id);
    free(s);
}

void event_seq_clear(EventSeq *s) {
    s->n = 0;
}

int event_append_text(EventSeq *s, const int *ids, int len, int start_time) {
    if (s->n + len > s->capacity) {
        fprintf(stderr, "event_append_text: overflow (n=%d + len=%d > cap=%d)\n",
                s->n, len, s->capacity);
        return -1;
    }
    for (int i = 0; i < len; i++) {
        int k = s->n + i;
        s->modality[k]   = MOD_TEXT;
        s->channel[k]    = 0;
        s->time_index[k] = start_time + i;
        s->value[k]      = 0.f;
        s->token_id[k]   = ids[i];
    }
    s->n += len;
    return 0;
}

int event_append_scalar(EventSeq *s, const float *vals, int len,
                        int modality, int channel, int start_time) {
    if (s->n + len > s->capacity) {
        fprintf(stderr, "event_append_scalar: overflow (n=%d + len=%d > cap=%d)\n",
                s->n, len, s->capacity);
        return -1;
    }
    for (int i = 0; i < len; i++) {
        int k = s->n + i;
        s->modality[k]   = modality;
        s->channel[k]    = channel;
        s->time_index[k] = start_time + i;
        s->value[k]      = vals[i];
        s->token_id[k]   = -1;
    }
    s->n += len;
    return 0;
}

/* ── EventEmbed ──────────────────────────────────────── */

static void make_sinusoidal(float *pos, int T, int D) {
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++) {
            float freq = powf(10000.f, -(float)(d/2*2)/(float)D);
            pos[t*D+d] = (d%2==0) ? sinf(t*freq) : cosf(t*freq);
        }
}

static void xavier_arr(float *w, int n, int fan_in, int fan_out) {
    float s = sqrtf(2.f / (fan_in + fan_out));
    for (int i = 0; i < n; i++) {
        float u = ((float)rand()+1)/((float)RAND_MAX+2);
        float v = ((float)rand()+1)/((float)RAND_MAX+2);
        w[i] = sqrtf(-2.f*logf(u))*cosf(6.2831853f*v)*s;
    }
}

EventEmbed *event_embed_new(int D, int V, int max_time) {
    EventEmbed *e = (EventEmbed*)calloc(1, sizeof(EventEmbed));
    e->D = D; e->V = V; e->max_time = max_time;
    e->mod_emb  = p_new(EVENT_MAX_MOD  * D);
    e->chan_emb = p_new(EVENT_MAX_CHAN * D);
    e->tok_emb  = p_new(V * D);
    e->val_w    = p_new(D);
    e->val_b    = p_new(D);
    e->pos      = (float*)calloc((size_t)max_time * D, sizeof(float));
    make_sinusoidal(e->pos, max_time, D);
    return e;
}

void event_embed_del(EventEmbed *e) {
    p_del(&e->mod_emb); p_del(&e->chan_emb); p_del(&e->tok_emb);
    p_del(&e->val_w);   p_del(&e->val_b);
    free(e->pos);
    free(e);
}

void event_embed_init(EventEmbed *e) {
    int D = e->D, V = e->V;
    xavier_arr(e->mod_emb.w,  EVENT_MAX_MOD  * D, EVENT_MAX_MOD,  D);
    xavier_arr(e->chan_emb.w, EVENT_MAX_CHAN * D, EVENT_MAX_CHAN, D);
    xavier_arr(e->tok_emb.w,  V * D, V, D);
    xavier_arr(e->val_w.w, D, 1, D);
    /* val_b stays zero */
}

void event_embed_zg(EventEmbed *e) {
    p_zg(&e->mod_emb); p_zg(&e->chan_emb); p_zg(&e->tok_emb);
    p_zg(&e->val_w);   p_zg(&e->val_b);
}

int event_embed_fwd(const EventSeq *s, const EventEmbed *e, Mat *out) {
    int D = e->D, n = s->n;
    /* validate all event fields before touching any array */
    for (int i = 0; i < n; i++) {
        if (s->modality[i] < 0 || s->modality[i] >= EVENT_MAX_MOD) {
            fprintf(stderr, "event_embed_fwd: modality[%d]=%d out of range [0,%d)\n",
                    i, s->modality[i], EVENT_MAX_MOD);
            return -1;
        }
        if (s->channel[i] < 0 || s->channel[i] >= EVENT_MAX_CHAN) {
            fprintf(stderr, "event_embed_fwd: channel[%d]=%d out of range [0,%d)\n",
                    i, s->channel[i], EVENT_MAX_CHAN);
            return -1;
        }
        if (s->time_index[i] < 0 || s->time_index[i] >= e->max_time) {
            fprintf(stderr, "event_embed_fwd: time_index[%d]=%d out of range [0,%d)\n",
                    i, s->time_index[i], e->max_time);
            return -1;
        }
        int tid = s->token_id[i];
        if (tid != -1 && (tid < 0 || tid >= e->V)) {
            fprintf(stderr, "event_embed_fwd: token_id[%d]=%d out of range [-1,%d)\n",
                    i, tid, e->V);
            return -1;
        }
    }
    for (int i = 0; i < n; i++) {
        float *dst  = out->d + i * D;
        int mod = s->modality[i];
        int ch  = s->channel[i];
        int t   = s->time_index[i];
        int tid = s->token_id[i];

        float *m_emb  = e->mod_emb.w  + mod * D;
        float *c_emb  = e->chan_emb.w + ch  * D;
        float *p_enc  = e->pos         + t   * D;

        if (tid >= 0) {
            /* text token: lookup + modality + channel + position */
            float *t_emb = e->tok_emb.w + tid * D;
            for (int d = 0; d < D; d++)
                dst[d] = t_emb[d] + m_emb[d] + c_emb[d] + p_enc[d];
        } else {
            /* scalar: value * val_w + val_b + modality + channel + position */
            float v = s->value[i];
            for (int d = 0; d < D; d++)
                dst[d] = v * e->val_w.w[d] + e->val_b.w[d]
                       + m_emb[d] + c_emb[d] + p_enc[d];
        }
    }
    return 0;
}

void event_embed_bwd(const EventSeq *s, const EventEmbed *e,
                     const Mat *dout, EventEmbed *de) {
    int D = e->D, n = s->n;
    for (int i = 0; i < n; i++) {
        const float *g  = dout->d + i * D;
        int mod = s->modality[i];
        int ch  = s->channel[i];
        int tid = s->token_id[i];

        /* skip if fields are out-of-range (mirrors event_embed_fwd validation) */
        if (mod < 0 || mod >= EVENT_MAX_MOD)  continue;
        if (ch  < 0 || ch  >= EVENT_MAX_CHAN) continue;
        if (s->time_index[i] < 0 || s->time_index[i] >= e->max_time) continue;
        if (tid != -1 && (tid < 0 || tid >= e->V)) continue;

        /* modality and channel embedding gradients (shared lookup) */
        float *dm_emb  = de->mod_emb.g  + mod * D;
        float *dc_emb  = de->chan_emb.g + ch  * D;
        for (int d = 0; d < D; d++) {
            dm_emb[d] += g[d];
            dc_emb[d] += g[d];
        }
        /* pos is sinusoidal (non-learnable) — no gradient */

        if (tid >= 0) {
            /* text: gradient flows to tok_emb[tid] */
            float *dt_emb = de->tok_emb.g + tid * D;
            for (int d = 0; d < D; d++) dt_emb[d] += g[d];
        } else {
            /* scalar: gradient flows to val_w and val_b */
            float v = s->value[i];
            for (int d = 0; d < D; d++) {
                de->val_w.g[d] += g[d] * v;
                de->val_b.g[d] += g[d];
            }
        }
    }
}

/* ── EventHead ───────────────────────────────────────── */

EventHead *event_head_new(int D, int V) {
    EventHead *h = (EventHead*)calloc(1, sizeof(EventHead));
    h->D = D; h->V = V;
    h->proj   = p_new(D * V);
    h->proj_b = p_new(V);
    return h;
}

void event_head_del(EventHead *h) {
    p_del(&h->proj); p_del(&h->proj_b);
    free(h);
}

void event_head_init(EventHead *h) {
    int D = h->D, V = h->V;
    float s = sqrtf(2.f / (D + V));
    for (int i = 0; i < D*V; i++) {
        float u = ((float)rand()+1)/((float)RAND_MAX+2);
        float v = ((float)rand()+1)/((float)RAND_MAX+2);
        h->proj.w[i] = sqrtf(-2.f*logf(u))*cosf(6.2831853f*v)*s;
    }
    /* proj_b stays zero */
}

void event_head_zg(EventHead *h) {
    p_zg(&h->proj); p_zg(&h->proj_b);
}

void event_head_fwd(const EventHead *h, const Mat *hidden, Mat *logits) {
    Mat proj = {h->proj.w, h->D, h->V};
    mm(hidden, &proj, logits);
    add_bias(logits, h->proj_b.w);
}

void event_head_bwd(const EventHead *h, const Mat *hidden,
                    const Mat *d_logits, Mat *d_hidden, EventHead *dh) {
    Mat proj  = {h->proj.w,  h->D, h->V};
    Mat dproj = {dh->proj.g, h->D, h->V};
    mm_bt_add(d_logits, &proj, d_hidden);
    mm_at_add(hidden, d_logits, &dproj);
    bias_bwd(d_logits, dh->proj_b.g);
}

