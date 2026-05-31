#include "model.h"
#include "event.h"
#include <stdio.h>

void run_event_task(void);  /* defined in event_task.c */
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Demo task: sequence copy.
 * Input:  [BOS, a, b, c, EOS]
 * Target: [BOS, a, b, c, EOS]  (teacher forcing input)
 * Label:  [a, b, c, EOS, EOS]  (shifted right)
 *
 * Tokens: 0=PAD, 1=BOS, 2=EOS, 3..3+ALPHA-1 = alphabet
 */
#define PAD   0
#define BOS   1
#define EOS   2
#define ALPHA 8     /* alphabet size */
#define VOCAB (3 + ALPHA)

#define MAX_SEQ 12

static void make_copy_pair(int *src, int *tgt, int *lbl, int *SL, int *TL) {
    int len = 2 + rand() % 4;  /* sequence length 2..5 */
    src[0] = BOS;
    for (int i = 0; i < len; i++) src[1+i] = 3 + rand() % ALPHA;
    src[1+len] = EOS;
    *SL = len + 2;

    tgt[0] = BOS;
    for (int i = 0; i < len; i++) tgt[1+i] = src[1+i];
    *TL = len + 1;

    for (int i = 0; i < len; i++) lbl[i] = src[1+i];
    lbl[len] = EOS;
}

int main(void) {
    srand((unsigned)time(NULL));

    Cfg cfg = {
        .V   = VOCAB,
        .T   = MAX_SEQ,
        .D   = 64,
        .H   = 4,
        .F   = 256,
        .L   = 2,
        .eps = 1e-5f
    };

    Model *m = model_new(&cfg);
    model_init(m);

    /* allocate caches */
    EC **ec = (EC**)malloc(cfg.L * sizeof(EC*));
    DC **dc = (DC**)malloc(cfg.L * sizeof(DC*));
    for (int i = 0; i < cfg.L; i++) {
        ec[i] = ec_new(cfg.T, cfg.D, cfg.F, cfg.H);
        dc[i] = dc_new(cfg.T, cfg.D, cfg.F, cfg.H);
    }

    Mat enc_out = mat_new(cfg.T, cfg.D);
    Mat dec_out = mat_new(cfg.T, cfg.D);
    Mat logits  = mat_new(cfg.T, cfg.V);

    Opt opt = { .lr=1e-3f, .b1=0.9f, .b2=0.98f, .eps=1e-9f, .step=0 };

    int src[MAX_SEQ], tgt[MAX_SEQ], lbl[MAX_SEQ];
    int SL, TL;

    printf("Training sequence copy task  (D=%d H=%d F=%d L=%d)\n",
           cfg.D, cfg.H, cfg.F, cfg.L);
    printf("%-8s  %-8s  %-8s  %-8s  %s\n",
           "step", "loss", "smooth", "min", "elapsed");
    printf("-------- -------- -------- -------- --------\n");

    int STEPS = 5000;
    int LOG_INTERVAL = 500;
    float smooth = 0.f, win_min = 1e9f;
    clock_t train_start = clock(), win_start = clock();

    for (int step = 0; step < STEPS; step++) {
        make_copy_pair(src, tgt, lbl, &SL, &TL);

        if (model_fwd(m, src, SL, tgt, TL, ec, dc, &enc_out, &dec_out, &logits) != 0) {
            fprintf(stderr, "model_fwd failed at step %d, skipping\n", step);
            continue;
        }

        model_zg(m);
        float loss = model_loss_bwd(m, src, SL, tgt, TL, lbl,
                                    ec, dc, &enc_out, &dec_out, &logits);
        if (loss < 0.f) {
            fprintf(stderr, "model_loss_bwd failed at step %d, skipping\n", step);
            continue;
        }
        opt_step(m, &opt);

        smooth  = (step == 0) ? loss : 0.99f*smooth + 0.01f*loss;
        win_min = (loss < win_min) ? loss : win_min;

        if ((step+1) % LOG_INTERVAL == 0) {
            double elapsed = (double)(clock()-win_start)/CLOCKS_PER_SEC;
            printf("%8d  %8.4f  %8.4f  %8.4f  %6.2fs\n",
                   step+1, loss, smooth, win_min, elapsed);
            win_min   = 1e9f;
            win_start = clock();
        }
    }
    double total_s = (double)(clock()-train_start)/CLOCKS_PER_SEC;
    printf("-------- -------- -------- -------- --------\n");
    printf("Total training time: %.2f s\n", total_s);

    /* greedy decode — baseline (full prefix re-forward each step) */
    printf("\n-- Greedy decode (baseline) --\n");
    make_copy_pair(src, tgt, lbl, &SL, &TL);
    printf("src: "); for (int i=0;i<SL;i++) printf("%d ",src[i]); printf("\n");
    printf("ref: "); for (int i=0;i<TL;i++) printf("%d ",lbl[i]); printf("\n");

    int gen[MAX_SEQ]; gen[0] = BOS;
    int gl = 1;
    for (int step = 0; step < MAX_SEQ-1; step++) {
        if (model_fwd(m, src, SL, gen, gl, ec, dc, &enc_out, &dec_out, &logits) != 0) break;
        float *row = logits.d + (gl-1)*cfg.V;
        int best = 0;
        for (int j = 1; j < cfg.V; j++) if (row[j]>row[best]) best=j;
        gen[gl++] = best;
        if (best == EOS) break;
    }
    printf("gen: "); for (int i=1;i<gl;i++) printf("%d ",gen[i]); printf("\n");

    /* greedy decode — KV cache */
    printf("\n-- Greedy decode (KV cache) --\n");
    if (model_encode(m, src, SL, ec, &enc_out) != 0) {
        fprintf(stderr, "model_encode failed\n");
        return 1;
    }

    DecodeCache *dkv = decode_cache_new(&cfg);
    if (decode_cache_precompute_cross(dkv, m, &enc_out) != 0) {
        fprintf(stderr, "decode_cache_precompute_cross failed\n");
        decode_cache_del(dkv);
        return 1;
    }

    Mat logits1 = mat_new(1, cfg.V);
    int gen2[MAX_SEQ]; gen2[0] = BOS;
    int gl2 = 1;
    for (int step = 0; step < MAX_SEQ-1; step++) {
        if (model_decode_step(m, gen2[gl2-1], gl2-1, dkv, &logits1) != 0) break;
        int best = 0;
        for (int j = 1; j < cfg.V; j++) if (logits1.d[j]>logits1.d[best]) best=j;
        gen2[gl2++] = best;
        if (best == EOS) break;
    }
    printf("gen: "); for (int i=1;i<gl2;i++) printf("%d ",gen2[i]); printf("\n");

    /* verify outputs match */
    int match = (gl == gl2);
    for (int i = 0; match && i < gl; i++) match = (gen[i] == gen2[i]);
    printf("KV cache match: %s\n", match ? "OK" : "MISMATCH");

    decode_cache_del(dkv);
    mat_del(&logits1);

    /* --- Inference timing benchmark--- */
    printf("\n-- Inference timing benchmark (N=200 sequences) --\n");
    int N_BENCH = 200;
    clock_t t0, t1;

    /* baseline: full prefix re-forward */
    t0 = clock();
    for (int r = 0; r < N_BENCH; r++) {
        make_copy_pair(src, tgt, lbl, &SL, &TL);
        int g[MAX_SEQ]; g[0] = BOS; int gl_b = 1;
        for (int step = 0; step < MAX_SEQ-1; step++) {
            if (model_fwd(m, src, SL, g, gl_b, ec, dc, &enc_out, &dec_out, &logits) != 0) break;
            float *row = logits.d + (gl_b-1)*cfg.V;
            int best = 0;
            for (int j = 1; j < cfg.V; j++) if (row[j]>row[best]) best=j;
            g[gl_b++] = best;
            if (best == EOS) break;
        }
    }
    t1 = clock();
    double baseline_ms = (double)(t1-t0)/CLOCKS_PER_SEC*1000.0/N_BENCH;

    /* KV cache */
    DecodeCache *bench_kv = decode_cache_new(&cfg);
    Mat logits_b = mat_new(1, cfg.V);
    t0 = clock();
    for (int r = 0; r < N_BENCH; r++) {
        make_copy_pair(src, tgt, lbl, &SL, &TL);
        if (model_encode(m, src, SL, ec, &enc_out) != 0) continue;
        if (decode_cache_precompute_cross(bench_kv, m, &enc_out) != 0) continue;
        decode_cache_reset(bench_kv);
        int g[MAX_SEQ]; g[0] = BOS; int gl_k = 1;
        for (int step = 0; step < MAX_SEQ-1; step++) {
            if (model_decode_step(m, g[gl_k-1], gl_k-1, bench_kv, &logits_b) != 0) break;
            int best = 0;
            for (int j = 1; j < cfg.V; j++) if (logits_b.d[j]>logits_b.d[best]) best=j;
            g[gl_k++] = best;
            if (best == EOS) break;
        }
    }
    t1 = clock();
    double kv_ms = (double)(t1-t0)/CLOCKS_PER_SEC*1000.0/N_BENCH;

    printf("Baseline : %.3f ms/seq\n", baseline_ms);
    printf("KV cache : %.3f ms/seq\n", kv_ms);
    printf("Speedup  : %.2fx\n", baseline_ms / kv_ms);
    decode_cache_del(bench_kv);
    mat_del(&logits_b);

    /* --- EventEmbed mixed demo--- */
    printf("\n-- EventEmbed: text + scalar mixed sequence --\n");
    EventEmbed *ee = event_embed_new(cfg.D, cfg.V, cfg.T);
    event_embed_init(ee);

    EventSeq *ev = event_seq_new(cfg.T);
    int text_ids[] = {BOS, 5, 7};
    event_append_text(ev, text_ids, 3, 0);
    float temps[] = {0.42f, 0.65f};
    event_append_scalar(ev, temps, 2, MOD_TEMPERATURE, 0, 3);
    float times[] = {0.10f, 0.20f, 0.30f};
    event_append_scalar(ev, times, 3, MOD_TIME, 0, 5);

    Mat ev_emb = mat_new(ev->n, cfg.D);
    event_embed_fwd(ev, ee, &ev_emb);
    printf("EventSeq length: %d  ->  embedding shape: %d x %d\n",
           ev->n, ev_emb.r, ev_emb.c);
    printf("Modalities: ");
    const char *mod_names[] = {"TEXT","IMAGE","AUDIO","TOUCH","TEMP","TIME","GENERIC"};
    for (int i = 0; i < ev->n; i++) printf("%s ", mod_names[ev->modality[i]]);
    printf("\n");

    mat_del(&ev_emb);
    event_seq_del(ev);
    event_embed_del(ee);

    /* --- EventEmbed training demo--- */
    run_event_task();

    /* cleanup */
    for (int i = 0; i < cfg.L; i++) { ec_del(ec[i]); dc_del(dc[i]); }
    free(ec); free(dc);
    mat_del(&enc_out); mat_del(&dec_out); mat_del(&logits);
    model_del(m);
    return 0;
}
