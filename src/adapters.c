#include "adapters.h"
#include <stdio.h>

/* ── ScalarBinAdapter ────────────────────────────────── */

int sba_encode(const ScalarBinAdapter *a, EventSeq *s, float val, int time_index) {
    /* clamp to [0,1] */
    if (val < 0.f) val = 0.f;
    if (val > 1.f) val = 1.f;

    int bin = (int)(val * (a->n_bins - 1) + 0.5f);
    if (bin < 0)          bin = 0;
    if (bin >= a->n_bins) bin = a->n_bins - 1;

    /* store normalized value; token_id stays -1 (scalar event).
       The bin is recorded in value so the caller can read it back.
       The event vocabulary label is vocab_offset + bin, used only in loss. */
    float norm = (float)bin / (a->n_bins - 1);
    return event_append_scalar(s, &norm, 1, a->modality, a->channel, time_index);
}

float sba_decode(const ScalarBinAdapter *a, int token_id) {
    int bin = token_id - a->vocab_offset;
    if (bin < 0 || bin >= a->n_bins) return -1.f;
    return (float)bin / (float)(a->n_bins - 1);
}

int sba_owns(const ScalarBinAdapter *a, int token_id) {
    int bin = token_id - a->vocab_offset;
    return bin >= 0 && bin < a->n_bins;
}

/* Convenience: return the event-vocab label for a raw float value. */
int sba_label(const ScalarBinAdapter *a, float val) {
    if (val < 0.f) val = 0.f;
    if (val > 1.f) val = 1.f;
    int bin = (int)(val * (a->n_bins - 1) + 0.5f);
    if (bin < 0)          bin = 0;
    if (bin >= a->n_bins) bin = a->n_bins - 1;
    return a->vocab_offset + bin;
}

/* ── TextAdapter ─────────────────────────────────────── */

int ta_encode(const TextAdapter *a, EventSeq *s,
              const int *ids, int len, int start_time) {
    for (int i = 0; i < len; i++) {
        if (ids[i] < 0 || ids[i] >= a->vocab_size) {
            fprintf(stderr, "ta_encode: invalid token id %d (vocab=%d)\n",
                    ids[i], a->vocab_size);
            return -1;
        }
    }
    return event_append_text(s, ids, len, start_time);
}

int ta_owns(const TextAdapter *a, int token_id) {
    return token_id >= 0 && token_id < a->vocab_size;
}
