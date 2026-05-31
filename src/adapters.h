#ifndef ADAPTERS_H
#define ADAPTERS_H

/*
 * Adapter layer: bridges raw modality-specific data and the standard EventSeq.
 *
 * Input adapters : raw data -> EventSeq
 * Output adapters: EventSeq (predicted events) -> interpreted output
 *
 * The model core (Transformer) only sees EventSeq, never raw formats.
 */

#include "event.h"

/* --- ScalarBinAdapter--- */
/*
 * Discretizes a continuous scalar value in [0, 1] into one of n_bins bins
 * and appends it to an EventSeq as a scalar event.
 *
 * The bin token id used in an event vocabulary is:
 *   vocab_offset + bin_index      (0 <= bin_index < n_bins)
 *
 * This lets the model predict scalar values as classification over bins,
 * using the same cross-entropy loss as text token prediction.
 */
typedef struct {
    int modality;      /* e.g. MOD_TEMPERATURE */
    int channel;       /* sensor channel index */
    int n_bins;        /* number of discretization bins */
    int vocab_offset;  /* base token id for this modality's bins */
} ScalarBinAdapter;

/*
 * Encode: float value -> EventSeq entry.
 * Returns 0 on success, -1 on overflow.
 */
int  sba_encode(const ScalarBinAdapter *a, EventSeq *s, float val, int time_index);

/*
 * Decode: bin token id -> float value (center of the bin).
 * Returns -1.f if token_id is outside the adapter's range.
 */
float sba_decode(const ScalarBinAdapter *a, int token_id);

/* Check whether a token_id belongs to this adapter. */
int   sba_owns(const ScalarBinAdapter *a, int token_id);

/* Return the event-vocabulary label for a raw float value. */
int   sba_label(const ScalarBinAdapter *a, float val);

/* --- TextAdapter--- */
/*
 * Appends text token ids to EventSeq with bounds and vocab validation.
 * token ids are expected in [0, vocab_size).
 */
typedef struct {
    int vocab_size;
} TextAdapter;

/*
 * Encode: int token ids -> EventSeq entries.
 * Returns 0 on success, -1 on overflow or invalid token id.
 */
int ta_encode(const TextAdapter *a, EventSeq *s,
              const int *ids, int len, int start_time);

/*
 * Decode: checks whether a predicted token is a valid text token.
 */
int ta_owns(const TextAdapter *a, int token_id);

/* --- Output adapter: EventHead logits -> EventSeq--- */
/*
 * Restore an EventSeq from EventHead logits (n x V).
 * For each position, picks argmax and uses TextAdapter / ScalarBinAdapters
 * to interpret the token and append the corresponding event to out_seq.
 *
 * sba_list: array of n_sba ScalarBinAdapters, tried in order.
 * eos_token_id: stop appending after this token is emitted.
 *
 * Returns the number of events appended (>= 0), or -1 if an append failed
 * (e.g. out_seq capacity exceeded).
 */
int event_head_to_seq(const Mat *logits,
                      const TextAdapter *ta,
                      const ScalarBinAdapter *sba_list, int n_sba,
                      EventSeq *out_seq,
                      int eos_token_id);

#endif
