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

/* --- ScalarBinAdapter --- */
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

/* --- TextAdapter --- */
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

/* --- Output adapter: EventHead logits -> EventSeq --- */
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

/* --- AdapterSchema --- */
/*
 * Stores adapter configuration so that saved model weights can be
 * interpreted correctly. Without this, the meaning of EventHead output
 * token ids cannot be recovered from the weights file alone.
 *
 * AdapterEntry: one adapter's config (text or scalar).
 * AdapterSchema: collection of all adapters used by a model.
 */

typedef enum {
    ADAPTER_TEXT       = 0,
    ADAPTER_SCALAR_BIN = 1
} AdapterType;

typedef struct {
    AdapterType type;
    int         modality;      /* Modality enum (for scalar); 0 for text */
    int         channel;       /* sensor channel (for scalar); 0 for text */
    int         vocab_offset;  /* first token id in event vocab */
    int         vocab_count;   /* number of tokens (vocab_size or n_bins) */
    float       val_min;       /* min value before normalization (scalar) */
    float       val_max;       /* max value before normalization (scalar) */
} AdapterEntry;

#define SCHEMA_MAX_ADAPTERS 32
#define SCHEMA_MAX_TOTAL_VOCAB 10000000

typedef struct {
    AdapterEntry entries[SCHEMA_MAX_ADAPTERS];
    int          n;            /* number of adapters */
    int          total_vocab;  /* total event vocab size */
} AdapterSchema;

AdapterSchema *adapter_schema_new(void);
void           adapter_schema_del(AdapterSchema *s);

/* Add a text adapter. Returns index (>=0) or -1 on error. */
int adapter_schema_add_text(AdapterSchema *s, int vocab_size);

/* Add a scalar bin adapter. Returns index (>=0) or -1 on error. */
int adapter_schema_add_scalar(AdapterSchema *s,
                              int modality, int channel,
                              int n_bins,
                              float val_min, float val_max);

/* Create TextAdapter / ScalarBinAdapter from schema entry. */
TextAdapter       adapter_schema_get_text(const AdapterSchema *s, int idx);
ScalarBinAdapter  adapter_schema_get_scalar(const AdapterSchema *s, int idx);

/* Validate schema consistency. Returns 0 if valid, -1 if invalid. */
int adapter_schema_validate(const AdapterSchema *s);

/* Serialization: save/load schema. Returns 0/-1 or new object/NULL.
 * Both save and load call validate internally. */
int            adapter_schema_save(const AdapterSchema *s, const char *path);
AdapterSchema *adapter_schema_load(const char *path);

/* --- ModelBundle --- */
/*
 * Bundles all components needed for inference with Event I/O:
 *   - Model (Transformer weights)
 *   - EventEmbed (event -> hidden)
 *   - EventHead (hidden -> event logits)
 *   - AdapterSchema (interpretation of event vocab token ids)
 *
 * Saving a ModelBundle produces a single file containing all components.
 * Loading restores all components, allowing full inference without
 * additional configuration.
 */
typedef struct {
    Model         *model;
    EventEmbed    *embed;
    EventHead     *head;
    AdapterSchema *schema;
} ModelBundle;

ModelBundle *model_bundle_new(Model *m, EventEmbed *e, EventHead *h, AdapterSchema *s);
void         model_bundle_del(ModelBundle *b);

/* Validate bundle component consistency:
 * - D matches across model, embed, head
 * - V matches between schema->total_vocab, embed->V, head->V
 * - embed->max_time >= model->c.T
 * Returns 0 if valid, -1 if invalid. */
int model_bundle_validate(const ModelBundle *b);

/* Save all components to a single file. Returns 0 on success, -1 on error.
 * Calls validate internally. */
int model_bundle_save(const ModelBundle *b, const char *path);

/* Load all components from a bundle file. Returns new bundle or NULL on error.
 * Validates component consistency after loading. */
ModelBundle *model_bundle_load(const char *path);

/* --- Checkpoint --- */
/*
 * Checkpoint groups:
 *   - ModelBundle: model + EventEmbed + EventHead + AdapterSchema
 *   - optimizer state: Adam moments and Opt values for bundle->model
 *   - train_step: caller-level progress marker
 *
 * Given path "foo.ckpt" and train_step 10, checkpoint_save writes:
 *   - foo.ckpt.step10.bundle
 *   - foo.ckpt.step10.opt
 *   - foo.ckpt        (manifest written last)
 *
 * checkpoint_load reads the manifest, then loads the derived bundle and
 * optimizer state. Returns a new ModelBundle or NULL on error.
 */
int checkpoint_save(const ModelBundle *b, const Opt *o, int train_step, const char *path);
ModelBundle *checkpoint_load(const char *path, Opt *o, int *train_step);

#endif
