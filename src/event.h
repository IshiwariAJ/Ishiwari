#ifndef EVENT_H
#define EVENT_H

#include "model.h"

/*
 * EventToken: modality-agnostic input unit.
 *
 * All input sources (text, scalar sensor, future: vision/audio/touch)
 * are converted to EventToken sequences before entering the Transformer.
 * The model itself never sees raw modality-specific formats.
 */

typedef enum {
    MOD_TEXT        = 0,
    MOD_IMAGE       = 1,
    MOD_AUDIO       = 2,
    MOD_TOUCH       = 3,
    MOD_TEMPERATURE = 4,
    MOD_TIME        = 5,
    MOD_GENERIC     = 6
} Modality;

/*
 * EventSeq: a sequence of EventTokens stored as parallel arrays.
 *
 * Fields:
 *   modality   : Modality enum value for each token
 *   channel    : sub-channel within the modality (e.g. sensor index)
 *   time_index : discrete time step (0-based)
 *   value      : normalized scalar value in [0, 1] (text uses 0)
 *   token_id   : discrete token id (text tokens; -1 for continuous-only)
 *   n          : current number of tokens
 *   capacity   : allocated size of each array
 */
typedef struct {
    int   *modality;
    int   *channel;
    int   *time_index;
    float *value;
    int   *token_id;
    int    n;
    int    capacity;
} EventSeq;

EventSeq *event_seq_new(int capacity);
void      event_seq_del(EventSeq *s);
void      event_seq_clear(EventSeq *s);

/*
 * Append text token ids as MOD_TEXT events.
 * Returns 0 on success, -1 if capacity would be exceeded.
 */
int event_append_text(EventSeq *s, const int *ids, int len, int start_time);

/*
 * Append scalar values as events. Values should be in [0, 1].
 * Returns 0 on success, -1 if capacity would be exceeded.
 */
int event_append_scalar(EventSeq *s, const float *vals, int len,
                        int modality, int channel, int start_time);

/*
 * EventEmbed: learnable embedding layer for EventSeq.
 *
 * Each event token is embedded as:
 *   out[i] = mod_emb[modality[i]]
 *           + chan_emb[channel[i]]
 *           + pos[time_index[i]]          (sinusoidal, non-learnable)
 *           + tok_emb[token_id[i]]        (when token_id >= 0, i.e. text)
 *           + value[i] * val_w + val_b    (when token_id < 0, i.e. scalar)
 */
#define EVENT_MAX_MOD  7
#define EVENT_MAX_CHAN 16

typedef struct {
    P     mod_emb;   /* (EVENT_MAX_MOD  x D) */
    P     chan_emb;  /* (EVENT_MAX_CHAN x D) */
    P     tok_emb;   /* (V x D)  text token lookup */
    P     val_w;     /* (D,)     scalar value scale */
    P     val_b;     /* (D,)     scalar value bias */
    float *pos;      /* (max_time x D) sinusoidal */
    int   D, V, max_time;
} EventEmbed;

EventEmbed *event_embed_new(int D, int V, int max_time);
void        event_embed_del(EventEmbed *e);
void        event_embed_init(EventEmbed *e);

/* Convert EventSeq to (s->n x D) embedding matrix. out must be pre-allocated.
 * Returns 0 on success, -1 if any event field is out of range. */
int event_embed_fwd(const EventSeq *s, const EventEmbed *e, Mat *out);

/*
 * Backward: given dout (s->n x D), accumulate gradients into de.
 * de->*.g fields must be zeroed before the call (or accumulated over batch).
 */
void event_embed_bwd(const EventSeq *s, const EventEmbed *e,
                     const Mat *dout, EventEmbed *de);

/* Zero all gradients in EventEmbed. */
void event_embed_zg(EventEmbed *e);

/*
 * EventHead: output projection for event vocabulary.
 *
 * Maps hidden (n x D) -> event_logits (n x V_event).
 * V_event covers all event token types (text vocab + scalar bins).
 * Symmetric counterpart to EventEmbed on the output side.
 */
typedef struct {
    P   proj;    /* (D x V)  output projection */
    P   proj_b;  /* (V,)     output bias */
    int D, V;
} EventHead;

EventHead *event_head_new(int D, int V);
void       event_head_del(EventHead *h);
void       event_head_init(EventHead *h);
void       event_head_zg(EventHead *h);

/* logits must be pre-allocated (n x V). */
void event_head_fwd(const EventHead *h, const Mat *hidden, Mat *logits);

/* d_hidden accumulated; dh->proj.g and dh->proj_b.g accumulated. */
void event_head_bwd(const EventHead *h, const Mat *hidden,
                    const Mat *d_logits, Mat *d_hidden, EventHead *dh);

/* ── EventEmbed / EventHead serialization ────────────── */
/* save: 0 ok / -1 fail. load: new object (free with *_del) / NULL fail. */
int         event_embed_save(const EventEmbed *e, const char *path);
EventEmbed *event_embed_load(const char *path);
int         event_head_save(const EventHead *h, const char *path);
EventHead  *event_head_load(const char *path);

#endif
