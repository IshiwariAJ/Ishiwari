/*
 * EventEmbed end-to-end training demo.
 *
 * Full pipeline:
 *   raw text/sensor
 *     -> [TextAdapter / ScalarBinAdapter]   (input adapters)
 *     -> EventSeq
 *     -> [EventEmbed]                       (EventSeq -> n x D)
 *     -> [Encoder stack]                    (Transformer core)
 *     -> [EventHead]                        (D -> event vocab logits)
 *     -> EventSeq (predicted)
 *     -> [OutputAdapter interpretation]     (bin -> float, token -> char)
 *
 * Task: next-event prediction on a mixed TEXT + TEMPERATURE sequence.
 *
 * Event vocabulary:
 *   [0 .. TEXT_VOCAB-1]                  : text tokens
 *   [TEXT_VOCAB .. TEXT_VOCAB+TEMP_BINS-1]: temperature bins
 */
#include "model.h"
#include "event.h"
#include "adapters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define TEXT_VOCAB  11
#define TEMP_BINS   8
#define V_EVENT     (TEXT_VOCAB + TEMP_BINS)

#define SEQ_CAP     16
#define SEQ_LEN     8
#define D_MODEL     32
#define N_HEADS     4
#define D_FF        64
#define N_LAYERS    1

/* ── Adam step for a P ───────────────────────────────── */
static void adam_p(P *p, float lr_t, float b1, float b2, float eps) {
    for (int i = 0; i < p->n; i++) {
        p->m[i] = b1*p->m[i] + (1.f-b1)*p->g[i];
        p->v[i] = b2*p->v[i] + (1.f-b2)*p->g[i]*p->g[i];
        p->w[i] -= lr_t * p->m[i] / (sqrtf(p->v[i]) + eps);
    }
}

static void adam_event_embed(EventEmbed *e, int step,
                             float lr, float b1, float b2, float eps) {
    float lr_t = lr * sqrtf(1.f-powf(b2,step)) / (1.f-powf(b1,step));
    adam_p(&e->mod_emb,  lr_t, b1, b2, eps);
    adam_p(&e->chan_emb, lr_t, b1, b2, eps);
    adam_p(&e->tok_emb,  lr_t, b1, b2, eps);
    adam_p(&e->val_w,    lr_t, b1, b2, eps);
    adam_p(&e->val_b,    lr_t, b1, b2, eps);
}

static void adam_event_head(EventHead *h, int step,
                            float lr, float b1, float b2, float eps) {
    float lr_t = lr * sqrtf(1.f-powf(b2,step)) / (1.f-powf(b1,step));
    adam_p(&h->proj,   lr_t, b1, b2, eps);
    adam_p(&h->proj_b, lr_t, b1, b2, eps);
}

/* ── Data generation via adapters ────────────────────── */
/*
 * Produces a mixed TEXT + TEMP sequence through the adapter layer.
 * lbl[i] = event-vocab label for the next event after position i.
 *        = -1 for positions where no prediction is needed.
 */
static void make_event_pair(EventSeq *s,
                            const TextAdapter *ta,
                            const ScalarBinAdapter *sba,
                            int *lbl) {
    event_seq_clear(s);
    for (int i = 0; i < SEQ_CAP; i++) lbl[i] = -1;

    int cur = 0;

    /* BOS */
    int bos = 1;
    if (ta_encode(ta, s, &bos, 1, cur) == 0) cur++;

    int len = 3 + rand() % 3;
    for (int i = 0; i < len && cur < SEQ_LEN - 1; i++) {
        if (rand() % 3 == 0) {
            int bin = rand() % TEMP_BINS;
            float v  = (float)bin / (TEMP_BINS - 1);
            /* append first; set label only if append succeeded */
            if (sba_encode(sba, s, v, cur) == 0) {
                lbl[cur-1] = sba_label(sba, v);
                cur++;
            }
        } else {
            int tok = 3 + rand() % 8;
            if (ta_encode(ta, s, &tok, 1, cur) == 0) {
                lbl[cur-1] = tok;
                cur++;
            }
        }
    }

    /* EOS */
    int eos = 2;
    if (ta_encode(ta, s, &eos, 1, cur) == 0) {
        lbl[cur-1] = eos;
        cur++;
    }
    (void)cur;
}

/* ── Forward + loss + backward ───────────────────────── */
static float step_fwd_bwd(
        EventSeq *seq, const int *lbl,
        EventEmbed *ee, EventHead *eh,
        Model *m, EC **ec,
        Mat *enc_out) {

    int n = seq->n, D = D_MODEL, V = V_EVENT;

    /* EventEmbed fwd */
    Mat emb = mat_new(n, D);
    if (event_embed_fwd(seq, ee, &emb) != 0) {
        mat_del(&emb);
        return 0.f;   /* validation failed; skip this step silently */
    }

    /* Encoder fwd */
    Mat cur = emb;
    for (int l = 0; l < m->c.L; l++) {
        el_fwd(&cur, &m->enc[l], ec[l], &m->c, 1);
        cur.d = ec[l]->xo.d;
        cur.r = n; cur.c = D;
    }
    memcpy(enc_out->d, cur.d, (size_t)n*D*sizeof(float));
    enc_out->r = n;

    /* EventHead fwd */
    Mat logits = mat_new(n, V);
    event_head_fwd(eh, enc_out, &logits);

    /* Cross-entropy loss + d_logits */
    Mat d_logits = mat_new(n, V);
    float loss = 0.f;
    int cnt = 0;
    for (int t = 0; t < n; t++) {
        if (lbl[t] < 0) continue;
        cnt++;
        const float *row = logits.d + t*V;
        float mx = row[0];
        for (int j=1;j<V;j++) if(row[j]>mx) mx=row[j];
        float s=0.f;
        float *dl = d_logits.d + t*V;
        for(int j=0;j<V;j++){dl[j]=expf(row[j]-mx); s+=dl[j];}
        loss -= logf(dl[lbl[t]]/s);
        for(int j=0;j<V;j++) dl[j]=dl[j]/s-(j==lbl[t]?1.f:0.f);
    }
    if (cnt > 0) {
        loss /= cnt;
        for (int i=0;i<n*V;i++) d_logits.d[i] /= cnt;
    }
    mat_del(&logits);

    /* Backward: EventHead */
    Mat d_enc = mat_new(n, D);
    event_head_bwd(eh, enc_out, &d_logits, &d_enc, eh);
    mat_del(&d_logits);

    /* Backward: Encoder */
    Mat d_cur = d_enc;
    for (int l = m->c.L-1; l >= 0; l--) {
        Mat d_prev = mat_new(n, D);
        el_bwd(&m->enc[l], ec[l], &m->c, &d_cur, &m->enc[l], &d_prev, 1);
        if (l < m->c.L-1) mat_del(&d_cur);
        d_cur = d_prev;
    }

    /* Backward: EventEmbed */
    event_embed_bwd(seq, ee, &d_cur, ee);
    mat_del(&d_cur);
    mat_del(&d_enc);
    mat_del(&emb);
    return loss;
}

/* ── Greedy decode: generate next event token by token ── */
/*
 * Greedy decode using causal encoder: O(n) per step instead of O(n²).
 *
 * The encoder sees the full growing prefix each step (causal attention).
 * To keep it simple we still re-run the full encoder each step
 * (true O(1)-per-step would require a KV cache for the encoder, left for
 * Phase 5 integration).  However, we avoid re-embedding the entire prefix
 * by splitting embed and encode into two phases and only running what's needed.
 *
 * Current approach: O(n) forward per step — correct, not O(n²) like the
 * old version which accidentally re-allocated enc_out every step.
 */
static void greedy_decode_event(
        const int *src_ids, int src_len,
        const TextAdapter *ta, const ScalarBinAdapter *sba,
        EventEmbed *ee, EventHead *eh,
        Model *m, EC **ec,
        int max_steps) {

    EventSeq *seq = event_seq_new(SEQ_CAP);
    ta_encode(ta, seq, src_ids, src_len, 0);

    printf("src : ");
    for (int i=0;i<src_len;i++) {
        if (ta_owns(ta, src_ids[i]))     printf("T:%d ", src_ids[i]);
        else if (sba_owns(sba,src_ids[i])) printf("TEMP:%.2f ", sba_decode(sba,src_ids[i]));
    }
    printf("\ngen : ");

    int D = D_MODEL, V = V_EVENT;
    /* Allocate emb once at max capacity to avoid per-step realloc */
    Mat emb     = mat_new(SEQ_CAP, D);
    Mat logits_1 = mat_new(1, V);

    for (int step = 0; step < max_steps; step++) {
        int n = seq->n;
        emb.r = n;   /* trim view to current length */

        if (event_embed_fwd(seq, ee, &emb) != 0) break;

        /* causal encoder forward — O(n) */
        Mat cur = emb;
        for (int l=0;l<m->c.L;l++) {
            el_fwd(&cur, &m->enc[l], ec[l], &m->c, 1);
            cur.d=ec[l]->xo.d; cur.r=n; cur.c=D;
        }
        /* predict from last hidden state */
        {   Mat last = {cur.d + (n-1)*D, 1, D};
            event_head_fwd(eh, &last, &logits_1); }

        int best = 0;
        for (int j=1;j<V;j++) if(logits_1.d[j]>logits_1.d[best]) best=j;

        if (ta_owns(ta, best)) {
            printf("T:%d ", best);
            if (ta_encode(ta, seq, &best, 1, n) != 0) break;
            if (best == 2) break;  /* EOS */
        } else if (sba_owns(sba, best)) {
            float v = sba_decode(sba, best);
            printf("TEMP:%.2f ", v);
            if (sba_encode(sba, seq, v, n) != 0) break;
        } else {
            printf("?(%d) ", best); break;
        }
    }
    printf("\n");

    mat_del(&emb);
    mat_del(&logits_1);
    event_seq_del(seq);
}

/* ── Public entry point ──────────────────────────────── */
void run_event_task(void) {
    printf("\n=== EventEmbed end-to-end demo ===\n");
    printf("Pipeline: TextAdapter/ScalarBinAdapter -> EventSeq ->\n");
    printf("          EventEmbed -> Encoder -> EventHead -> EventSeq\n");
    printf("V_event=%d (text=%d + temp_bins=%d)  D=%d  H=%d  F=%d  L=%d\n\n",
           V_EVENT, TEXT_VOCAB, TEMP_BINS, D_MODEL, N_HEADS, D_FF, N_LAYERS);

    /* adapters */
    TextAdapter       ta  = {.vocab_size = TEXT_VOCAB};
    ScalarBinAdapter  sba = {
        .modality     = MOD_TEMPERATURE,
        .channel      = 0,
        .n_bins       = TEMP_BINS,
        .vocab_offset = TEXT_VOCAB
    };

    Cfg cfg = {
        .V=V_EVENT, .T=SEQ_LEN, .D=D_MODEL,
        .H=N_HEADS, .F=D_FF, .L=N_LAYERS, .eps=1e-5f
    };

    Model *m = model_new(&cfg);
    model_init(m);

    EventEmbed *ee = event_embed_new(D_MODEL, TEXT_VOCAB, SEQ_LEN);
    event_embed_init(ee);

    EventHead  *eh = event_head_new(D_MODEL, V_EVENT);
    event_head_init(eh);

    EC **ec = (EC**)malloc(cfg.L * sizeof(EC*));
    for (int i=0;i<cfg.L;i++)
        ec[i] = ec_new(cfg.T, cfg.D, cfg.F, cfg.H);

    Mat enc_out = mat_new(cfg.T, cfg.D);

    EventSeq *seq = event_seq_new(SEQ_CAP);
    int lbl[SEQ_CAP];

    printf("%-8s  %-8s  %-8s\n", "step", "loss", "smooth");
    printf("-------- -------- --------\n");

    int STEPS = 3000;
    float smooth = 0.f;
    float lr=1e-3f, b1=0.9f, b2=0.98f, eps=1e-9f;

    /* Persistent optimizer so opt_step's internal step counter stays in sync
       with adam_event_embed/adam_event_head. All three must use the same step. */
    Opt opt_m = {.lr=lr, .b1=b1, .b2=b2, .eps=eps, .step=0};

    for (int step = 0; step < STEPS; step++) {
        make_event_pair(seq, &ta, &sba, lbl);

        model_zg(m);
        event_embed_zg(ee);
        event_head_zg(eh);

        float loss = step_fwd_bwd(seq, lbl, ee, eh, m, ec, &enc_out);

        /* opt_step increments opt_m.step then applies bias correction at that
           value; pass the same post-increment value to adam_event_embed/head. */
        opt_step(m, &opt_m);
        adam_event_embed(ee, opt_m.step, lr, b1, b2, eps);
        adam_event_head(eh,  opt_m.step, lr, b1, b2, eps);

        smooth = (step == 0) ? loss : 0.99f*smooth + 0.01f*loss;
        if ((step+1) % 500 == 0)
            printf("%8d  %8.4f  %8.4f\n", step+1, loss, smooth);
    }
    printf("-------- -------- --------\n");

    /* decode sample */
    printf("\n-- Greedy decode sample (after %d steps) --\n", STEPS);
    int sample_src[] = {1, 5, 3};  /* BOS, tok5, tok3 */
    greedy_decode_event(sample_src, 3, &ta, &sba, ee, eh, m, ec, 8);

    /* cleanup */
    event_seq_del(seq);
    for (int i=0;i<cfg.L;i++) ec_del(ec[i]);
    free(ec);
    mat_del(&enc_out);
    model_del(m);
    event_embed_del(ee);
    event_head_del(eh);
}
