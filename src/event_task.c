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
#define ALPHA_N     8       /* alphabet tokens: ids 3..10 */
#define TEMP_BINS   8
#define TIME_BINS   6

/* Event vocabulary layout (no overlaps):
 *   [0 .. TEXT_VOCAB-1]                       text tokens
 *   [TEMP_OFFSET .. TEMP_OFFSET+TEMP_BINS-1]  temperature bins
 *   [TIME_OFFSET .. TIME_OFFSET+TIME_BINS-1]  time bins
 */
#define TEMP_OFFSET TEXT_VOCAB
#define TIME_OFFSET (TEXT_VOCAB + TEMP_BINS)
#define V_EVENT     (TEXT_VOCAB + TEMP_BINS + TIME_BINS)

#define SEQ_CAP     16
#define SEQ_LEN     8
#define D_MODEL     32
#define N_HEADS     4
#define D_FF        64
#define N_LAYERS    1

/* --- Adam step for a P--- */
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

/* --- Deterministic task rule (learnable)--- */
/*
 * The demo task is a deterministic 3-modality cycle (TEXT -> TEMP -> TIME),
 * so the model CAN learn it (unlike the earlier random task whose loss
 * stayed near the random baseline log(V_EVENT)).
 *
 * Sequence layout (length SEQ_LEN):
 *   pos 0 : BOS
 *   pos 1 : TEXT seed token (3..10), the only random element
 *   pos 2 : TEMP bin   = rule_text_to_temp(seed)
 *   pos 3 : TIME bin   = rule_temp_to_time(prev_temp)
 *   pos 4 : TEXT token = rule_time_to_text(prev_time)
 *   ...cycle repeats TEXT / TEMP / TIME...
 *   last  : EOS
 *
 * Every transition except BOS->seed is fully determined, so a trained model
 * should reach near-zero loss on all but the seed position.
 */
/* 3-modality deterministic cycle: TEXT -> TEMP -> TIME -> TEXT -> ...     */
static int rule_text_to_temp(int tok)  { return (tok  - 3) % TEMP_BINS; }   /* -> temp bin */
static int rule_temp_to_time(int tbin) { return (tbin + 1) % TIME_BINS; }   /* -> time bin */
static int rule_time_to_text(int mbin) { return 3 + (mbin % ALPHA_N); }     /* -> text tok */

/*
 * Compute the deterministic continuation (in event-vocab ids) that should
 * follow a [BOS, seed] prefix, mirroring make_event_pair exactly.
 * The cycle phase for position p (p>=1) is (p-1)%3: 0=TEXT,1=TEMP,2=TIME.
 * Writes up to `max` ids into exp[], returns the count.
 */
static int build_expected(int seed, int *exp, int max) {
    int k = 0, tok = seed, tbin = 0, mbin = 0;
    for (int pos = 2; pos < SEQ_LEN && k < max; pos++) {
        if (pos == SEQ_LEN - 1) { exp[k++] = 2; break; }     /* EOS at last pos */
        int phase = (pos - 1) % 3;
        if (phase == 1) {                                    /* TEMP follows TEXT */
            tbin = rule_text_to_temp(tok);
            exp[k++] = TEMP_OFFSET + tbin;
        } else if (phase == 2) {                             /* TIME follows TEMP */
            mbin = rule_temp_to_time(tbin);
            exp[k++] = TIME_OFFSET + mbin;
        } else {                                             /* TEXT follows TIME */
            tok = rule_time_to_text(mbin);
            exp[k++] = tok;
        }
    }
    return k;
}

/*
 * Build one deterministic event sequence and its next-event labels.
 * lbl[i] = event-vocab label that position i must predict (-1 = no loss).
 */
static void make_event_pair(EventSeq *s,
                            const TextAdapter *ta,
                            const ScalarBinAdapter *sba_temp,
                            const ScalarBinAdapter *sba_time,
                            int *lbl) {
    event_seq_clear(s);
    for (int i = 0; i < SEQ_CAP; i++) lbl[i] = -1;

    int cur = 0;
    int bos = 1;
    if (ta_encode(ta, s, &bos, 1, cur) != 0) return;
    cur++;

    int tok = 3 + rand() % ALPHA_N;  /* random seed token (only random element) */
    int tbin = 0, mbin = 0;
    int max_inner = SEQ_LEN - 2;     /* leave room for EOS */

    for (int i = 0; i < max_inner; i++) {
        int phase = i % 3;  /* 0=TEXT, 1=TEMP, 2=TIME */
        if (phase == 0) {
            if (ta_encode(ta, s, &tok, 1, cur) != 0) break;
            lbl[cur-1] = tok;
            cur++;
            tbin = rule_text_to_temp(tok);
        } else if (phase == 1) {
            float v = (float)tbin / (TEMP_BINS - 1);
            if (sba_encode(sba_temp, s, v, cur) != 0) break;
            lbl[cur-1] = sba_label(sba_temp, v);
            cur++;
            mbin = rule_temp_to_time(tbin);
        } else {
            float v = (float)mbin / (TIME_BINS - 1);
            if (sba_encode(sba_time, s, v, cur) != 0) break;
            lbl[cur-1] = sba_label(sba_time, v);
            cur++;
            tok = rule_time_to_text(mbin);
        }
    }

    int eos = 2;
    if (ta_encode(ta, s, &eos, 1, cur) == 0) {
        lbl[cur-1] = eos;
        cur++;
    }
    (void)cur;
}

/* --- Forward + loss + backward--- */
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

/* Human-readable modality tag for printing. */
static const char *mod_name(int modality) {
    switch (modality) {
        case MOD_TEMPERATURE: return "TEMP";
        case MOD_TIME:        return "TIME";
        case MOD_TOUCH:       return "TOUCH";
        default:              return "SCAL";
    }
}

/* --- Greedy decode: generate next event token by token--- */
/*
 * Re-runs the full causal encoder over the growing prefix at each step.
 * This is correct but NOT KV-cached: the self-attention cost grows with the
 * prefix length, so a single decode of length n costs O(n^2) attention work
 * overall (each step t attends over t positions). A future KV-cache pass
 * (Phase 5 style, see inference.c) would make each step O(prefix) instead.
 *
 * The `emb` buffer is allocated once at SEQ_CAP and its row count trimmed
 * each step to avoid per-step reallocation.
 */
static void greedy_decode_event(
        const int *src_ids, int src_len,
        const TextAdapter *ta,
        const ScalarBinAdapter *sbas, int n_sba,
        EventEmbed *ee, EventHead *eh,
        Model *m, EC **ec,
        int max_steps,
        int *gen_out, int *gen_n) {

    EventSeq *seq = event_seq_new(SEQ_CAP);
    ta_encode(ta, seq, src_ids, src_len, 0);

    printf("src : ");
    for (int i=0;i<src_len;i++) {
        if (ta_owns(ta, src_ids[i])) { printf("T:%d ", src_ids[i]); continue; }
        for (int k=0;k<n_sba;k++)
            if (sba_owns(&sbas[k], src_ids[i])) {
                printf("%s:%.2f ", mod_name(sbas[k].modality),
                       sba_decode(&sbas[k], src_ids[i]));
                break;
            }
    }
    printf("\ngen : ");

    int D = D_MODEL, V = V_EVENT;
    /* Allocate emb once at max capacity to avoid per-step realloc */
    Mat emb     = mat_new(SEQ_CAP, D);
    Mat logits_1 = mat_new(1, V);
    int g = 0;

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
        if (gen_out && g < SEQ_CAP) gen_out[g] = best;
        g++;

        if (ta_owns(ta, best)) {
            printf("T:%d ", best);
            if (ta_encode(ta, seq, &best, 1, n) != 0) break;
            if (best == 2) break;  /* EOS */
        } else {
            int handled = 0;
            for (int k = 0; k < n_sba; k++) {
                if (!sba_owns(&sbas[k], best)) continue;
                float v = sba_decode(&sbas[k], best);
                printf("%s:%.2f ", mod_name(sbas[k].modality), v);
                handled = (sba_encode(&sbas[k], seq, v, n) == 0) ? 1 : -1;
                break;
            }
            if (handled == 0) { printf("?(%d) ", best); break; }  /* unknown id */
            if (handled <  0) break;                              /* append failed */
        }
    }
    printf("\n");
    if (gen_n) *gen_n = g;

    mat_del(&emb);
    mat_del(&logits_1);
    event_seq_del(seq);
}

/* --- Public entry point--- */
void run_event_task(void) {
    printf("\n=== EventEmbed end-to-end demo (3 modalities) ===\n");
    printf("Pipeline: Text/Temp/Time Adapter -> EventSeq ->\n");
    printf("          EventEmbed -> causal Encoder -> EventHead -> EventSeq\n");
    printf("Task: deterministic cycle TEXT -> TEMP -> TIME -> TEXT ...\n");
    printf("V_event=%d (text=%d + temp=%d + time=%d)  D=%d H=%d F=%d L=%d\n\n",
           V_EVENT, TEXT_VOCAB, TEMP_BINS, TIME_BINS, D_MODEL, N_HEADS, D_FF, N_LAYERS);

    /* adapters */
    TextAdapter      ta = {.vocab_size = TEXT_VOCAB};
    ScalarBinAdapter sba_temp = {
        .modality = MOD_TEMPERATURE, .channel = 0,
        .n_bins = TEMP_BINS, .vocab_offset = TEMP_OFFSET
    };
    ScalarBinAdapter sba_time = {
        .modality = MOD_TIME, .channel = 0,
        .n_bins = TIME_BINS, .vocab_offset = TIME_OFFSET
    };
    const ScalarBinAdapter sbas[2] = { sba_temp, sba_time };

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
        make_event_pair(seq, &ta, &sba_temp, &sba_time, lbl);

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

    /* decode + rule verification: feed [BOS, seed], expect deterministic tail */
    printf("\n-- Greedy decode + rule check (after %d steps) --\n", STEPS);
    int seeds[] = {3, 5, 8, 10};
    int n_seed = (int)(sizeof(seeds)/sizeof(seeds[0]));
    int total_ok = 0, total_cmp = 0;
    for (int si = 0; si < n_seed; si++) {
        int seed = seeds[si];
        int src[2] = {1 /*BOS*/, seed};
        int gen[SEQ_CAP], gn = 0;
        greedy_decode_event(src, 2, &ta, sbas, 2, ee, eh, m, ec, SEQ_LEN, gen, &gn);

        int exp[SEQ_CAP];
        int en = build_expected(seed, exp, SEQ_CAP);

        int cmp = (gn < en) ? gn : en;
        int ok = 0;
        for (int k = 0; k < cmp; k++) if (gen[k] == exp[k]) ok++;
        total_ok += ok; total_cmp += en;

        printf("exp : ");
        for (int k = 0; k < en; k++) {
            if (exp[k] < TEXT_VOCAB) { printf("T:%d ", exp[k]); continue; }
            int shown = 0;
            for (int j = 0; j < 2; j++)
                if (sba_owns(&sbas[j], exp[k])) {
                    printf("%s:%.2f ", mod_name(sbas[j].modality),
                           sba_decode(&sbas[j], exp[k]));
                    shown = 1; break;
                }
            if (!shown) printf("?(%d) ", exp[k]);
        }
        printf("  [%d/%d match]\n", ok, en);
    }
    printf("Rule accuracy: %d/%d = %.1f%%  %s\n",
           total_ok, total_cmp, 100.0*total_ok/total_cmp,
           (total_ok == total_cmp) ? "PASS" : "(not perfect)");

    /* cleanup */
    event_seq_del(seq);
    for (int i=0;i<cfg.L;i++) ec_del(ec[i]);
    free(ec);
    mat_del(&enc_out);
    model_del(m);
    event_embed_del(ee);
    event_head_del(eh);
}
