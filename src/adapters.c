#include "adapters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* --- AdapterSchema --- */

#define SCHEMA_MAGIC   "ADSC"
#define SCHEMA_VERSION 1

AdapterSchema *adapter_schema_new(void) {
    AdapterSchema *s = (AdapterSchema*)calloc(1, sizeof(AdapterSchema));
    s->n = 0;
    s->total_vocab = 0;
    return s;
}

void adapter_schema_del(AdapterSchema *s) {
    free(s);
}

int adapter_schema_add_text(AdapterSchema *s, int vocab_size) {
    if (s->n >= SCHEMA_MAX_ADAPTERS) {
        fprintf(stderr, "adapter_schema_add_text: max adapters reached\n");
        return -1;
    }
    if (vocab_size <= 0) {
        fprintf(stderr, "adapter_schema_add_text: vocab_size=%d must be >0\n", vocab_size);
        return -1;
    }
    /* text adapter must be single and first (offset 0) */
    for (int i = 0; i < s->n; i++) {
        if (s->entries[i].type == ADAPTER_TEXT) {
            fprintf(stderr, "adapter_schema_add_text: only one text adapter allowed\n");
            return -1;
        }
    }
    if (s->n > 0) {
        fprintf(stderr, "adapter_schema_add_text: text adapter must be added first\n");
        return -1;
    }
    /* overflow check */
    if (vocab_size > SCHEMA_MAX_TOTAL_VOCAB - s->total_vocab) {
        fprintf(stderr, "adapter_schema_add_text: total_vocab overflow\n");
        return -1;
    }
    int idx = s->n;
    AdapterEntry *e = &s->entries[idx];
    e->type = ADAPTER_TEXT;
    e->modality = 0;
    e->channel = 0;
    e->vocab_offset = s->total_vocab;  /* always 0 for text */
    e->vocab_count = vocab_size;
    e->val_min = 0.f;
    e->val_max = 1.f;
    s->total_vocab += vocab_size;
    s->n++;
    return idx;
}

int adapter_schema_add_scalar(AdapterSchema *s,
                              int modality, int channel,
                              int n_bins,
                              float val_min, float val_max) {
    if (s->n >= SCHEMA_MAX_ADAPTERS) {
        fprintf(stderr, "adapter_schema_add_scalar: max adapters reached\n");
        return -1;
    }
    if (n_bins < 2) {
        fprintf(stderr, "adapter_schema_add_scalar: n_bins=%d must be >=2\n", n_bins);
        return -1;
    }
    if (modality < 0 || modality >= EVENT_MAX_MOD) {
        fprintf(stderr, "adapter_schema_add_scalar: modality=%d out of range [0,%d)\n",
                modality, EVENT_MAX_MOD);
        return -1;
    }
    if (channel < 0 || channel >= EVENT_MAX_CHAN) {
        fprintf(stderr, "adapter_schema_add_scalar: channel=%d out of range [0,%d)\n",
                channel, EVENT_MAX_CHAN);
        return -1;
    }
    if (val_min >= val_max) {
        fprintf(stderr, "adapter_schema_add_scalar: val_min=%f must be < val_max=%f\n",
                (double)val_min, (double)val_max);
        return -1;
    }
    /* overflow check */
    if (n_bins > SCHEMA_MAX_TOTAL_VOCAB - s->total_vocab) {
        fprintf(stderr, "adapter_schema_add_scalar: total_vocab overflow\n");
        return -1;
    }
    int idx = s->n;
    AdapterEntry *e = &s->entries[idx];
    e->type = ADAPTER_SCALAR_BIN;
    e->modality = modality;
    e->channel = channel;
    e->vocab_offset = s->total_vocab;
    e->vocab_count = n_bins;
    e->val_min = val_min;
    e->val_max = val_max;
    s->total_vocab += n_bins;
    s->n++;
    return idx;
}

/* --- AdapterSchema validation --- */

int adapter_schema_validate(const AdapterSchema *s) {
    if (s->n < 0 || s->n > SCHEMA_MAX_ADAPTERS) return -1;
    if (s->total_vocab < 0 || s->total_vocab > SCHEMA_MAX_TOTAL_VOCAB) return -1;

    int computed_vocab = 0;
    int text_count = 0;

    for (int i = 0; i < s->n; i++) {
        const AdapterEntry *e = &s->entries[i];

        /* type must be known */
        if (e->type != ADAPTER_TEXT && e->type != ADAPTER_SCALAR_BIN) return -1;

        /* vocab_offset must match computed position (contiguous) */
        if (e->vocab_offset != computed_vocab) return -1;

        /* vocab_count must be positive */
        if (e->vocab_count <= 0) return -1;

        if (e->type == ADAPTER_TEXT) {
            /* text adapter must be single and first */
            text_count++;
            if (text_count > 1) return -1;
            if (i != 0) return -1;
            if (e->vocab_offset != 0) return -1;
        } else {
            /* scalar adapter: vocab_count >= 2 */
            if (e->vocab_count < 2) return -1;
            /* modality / channel in range */
            if (e->modality < 0 || e->modality >= EVENT_MAX_MOD) return -1;
            if (e->channel < 0 || e->channel >= EVENT_MAX_CHAN) return -1;
            /* val_min < val_max */
            if (e->val_min >= e->val_max) return -1;
        }

        /* overflow check before addition */
        if (e->vocab_count > SCHEMA_MAX_TOTAL_VOCAB - computed_vocab) return -1;
        computed_vocab += e->vocab_count;
    }

    /* total_vocab must match computed sum */
    if (s->total_vocab != computed_vocab) return -1;

    return 0;
}

TextAdapter adapter_schema_get_text(const AdapterSchema *s, int idx) {
    TextAdapter ta = {0};
    if (idx < 0 || idx >= s->n) return ta;
    if (s->entries[idx].type != ADAPTER_TEXT) return ta;
    ta.vocab_size = s->entries[idx].vocab_count;
    return ta;
}

ScalarBinAdapter adapter_schema_get_scalar(const AdapterSchema *s, int idx) {
    ScalarBinAdapter sba = {0};
    if (idx < 0 || idx >= s->n) return sba;
    if (s->entries[idx].type != ADAPTER_SCALAR_BIN) return sba;
    const AdapterEntry *e = &s->entries[idx];
    sba.modality = e->modality;
    sba.channel = e->channel;
    sba.n_bins = e->vocab_count;
    sba.vocab_offset = e->vocab_offset;
    return sba;
}

/* --- AdapterSchema serialization --- */

static int wr(const void *p, size_t sz, size_t n, FILE *f) {
    return fwrite(p, sz, n, f) == n;
}
static int rd(void *p, size_t sz, size_t n, FILE *f) {
    return fread(p, sz, n, f) == n;
}

/* --- internal save/load to FILE* --- */

static int adapter_schema_save_to_fp(const AdapterSchema *s, FILE *f) {
    int ok = 1;
    int ver = SCHEMA_VERSION;
    ok &= wr(SCHEMA_MAGIC, 1, 4, f);
    ok &= wr(&ver, sizeof(int), 1, f);
    ok &= wr(&s->n, sizeof(int), 1, f);
    ok &= wr(&s->total_vocab, sizeof(int), 1, f);

    for (int i = 0; i < s->n && ok; i++) {
        const AdapterEntry *e = &s->entries[i];
        int type_i = (int)e->type;
        ok &= wr(&type_i, sizeof(int), 1, f);
        ok &= wr(&e->modality, sizeof(int), 1, f);
        ok &= wr(&e->channel, sizeof(int), 1, f);
        ok &= wr(&e->vocab_offset, sizeof(int), 1, f);
        ok &= wr(&e->vocab_count, sizeof(int), 1, f);
        ok &= wr(&e->val_min, sizeof(float), 1, f);
        ok &= wr(&e->val_max, sizeof(float), 1, f);
    }

    return ok ? 0 : -1;
}

static AdapterSchema *adapter_schema_load_from_fp(FILE *f) {
    char magic[4];
    int ver, n, total_vocab;
    if (!rd(magic, 1, 4, f) || memcmp(magic, SCHEMA_MAGIC, 4) != 0) return NULL;
    if (!rd(&ver, sizeof(int), 1, f) || ver != SCHEMA_VERSION) return NULL;
    if (!rd(&n, sizeof(int), 1, f) || n < 0 || n > SCHEMA_MAX_ADAPTERS) return NULL;
    if (!rd(&total_vocab, sizeof(int), 1, f) || total_vocab < 0) return NULL;

    AdapterSchema *s = adapter_schema_new();
    s->n = n;
    s->total_vocab = total_vocab;

    int ok = 1;
    for (int i = 0; i < n && ok; i++) {
        AdapterEntry *e = &s->entries[i];
        int type_i;
        ok &= rd(&type_i, sizeof(int), 1, f);
        ok &= rd(&e->modality, sizeof(int), 1, f);
        ok &= rd(&e->channel, sizeof(int), 1, f);
        ok &= rd(&e->vocab_offset, sizeof(int), 1, f);
        ok &= rd(&e->vocab_count, sizeof(int), 1, f);
        ok &= rd(&e->val_min, sizeof(float), 1, f);
        ok &= rd(&e->val_max, sizeof(float), 1, f);
        e->type = (AdapterType)type_i;
    }

    if (!ok) { adapter_schema_del(s); return NULL; }
    if (adapter_schema_validate(s) != 0) { adapter_schema_del(s); return NULL; }
    return s;
}

/* --- public API --- */

int adapter_schema_save(const AdapterSchema *s, const char *path) {
    if (adapter_schema_validate(s) != 0) {
        fprintf(stderr, "adapter_schema_save: schema validation failed\n");
        return -1;
    }
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int result = adapter_schema_save_to_fp(s, f);
    fclose(f);
    return result;
}

AdapterSchema *adapter_schema_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    AdapterSchema *s = adapter_schema_load_from_fp(f);
    fclose(f);
    return s;
}

/* --- ScalarBinAdapter --- */

int sba_encode(const ScalarBinAdapter *a, EventSeq *s, float val, int time_index) {
    if (a->n_bins < 2) {
        fprintf(stderr, "sba_encode: n_bins=%d must be >= 2\n", a->n_bins);
        return -1;
    }
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
    if (a->n_bins < 2) return -1.f;
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
    if (a->n_bins < 2) return a->vocab_offset;
    if (val < 0.f) val = 0.f;
    if (val > 1.f) val = 1.f;
    int bin = (int)(val * (a->n_bins - 1) + 0.5f);
    if (bin < 0)          bin = 0;
    if (bin >= a->n_bins) bin = a->n_bins - 1;
    return a->vocab_offset + bin;
}

/* --- TextAdapter--- */

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

/* --- Output adapter: EventHead logits -> EventSeq--- */

int event_head_to_seq(const Mat *logits,
                      const TextAdapter *ta,
                      const ScalarBinAdapter *sba_list, int n_sba,
                      EventSeq *out_seq,
                      int eos_token_id) {
    int n = logits->r, V = logits->c;
    int appended = 0;
    for (int t = 0; t < n; t++) {
        const float *row = logits->d + t * V;
        int best = 0;
        for (int j = 1; j < V; j++) if (row[j] > row[best]) best = j;

        if (ta_owns(ta, best)) {
            if (ta_encode(ta, out_seq, &best, 1, t) != 0) return -1;
        } else {
            int matched = 0;
            for (int k = 0; k < n_sba; k++) {
                if (sba_owns(&sba_list[k], best)) {
                    float v = sba_decode(&sba_list[k], best);
                    if (event_append_scalar(out_seq, &v, 1,
                                            sba_list[k].modality,
                                            sba_list[k].channel, t) != 0)
                        return -1;
                    matched = 1;
                    break;
                }
            }
            if (!matched) break;  /* unknown token: stop cleanly */
        }
        appended++;
        if (best == eos_token_id) break;
    }
    return appended;
}

/* --- ModelBundle --- */

#define BUNDLE_MAGIC   "MBDL"
#define BUNDLE_VERSION 1
#define BUNDLE_MAX_COMPONENT_SIZE ((int64_t)1 << 30)   /* 1GB per component */
#define BUNDLE_MAX_TOTAL_SIZE     ((int64_t)4 << 30)   /* 4GB total */
#define BUNDLE_HEADER_SIZE        (4 + 4 + 4*8)        /* magic + ver + 4 x int64 */

ModelBundle *model_bundle_new(Model *m, EventEmbed *e, EventHead *h, AdapterSchema *s) {
    ModelBundle *b = (ModelBundle*)calloc(1, sizeof(ModelBundle));
    b->model  = m;
    b->embed  = e;
    b->head   = h;
    b->schema = s;
    return b;
}

void model_bundle_del(ModelBundle *b) {
    if (!b) return;
    if (b->model)  model_del(b->model);
    if (b->embed)  event_embed_del(b->embed);
    if (b->head)   event_head_del(b->head);
    if (b->schema) adapter_schema_del(b->schema);
    free(b);
}

int model_bundle_validate(const ModelBundle *b) {
    if (!b) return -1;
    if (!b->model || !b->embed || !b->head || !b->schema) return -1;

    /* D must match across all components */
    int model_D = b->model->c.D;
    int embed_D = b->embed->D;
    int head_D  = b->head->D;
    if (model_D != embed_D || model_D != head_D) {
        fprintf(stderr, "model_bundle_validate: D mismatch (model=%d, embed=%d, head=%d)\n",
                model_D, embed_D, head_D);
        return -1;
    }

    /* V (event vocab size) must match between schema, embed, and head */
    int schema_V = b->schema->total_vocab;
    int embed_V  = b->embed->V;
    int head_V   = b->head->V;
    if (schema_V != embed_V || schema_V != head_V) {
        fprintf(stderr, "model_bundle_validate: V mismatch (schema=%d, embed=%d, head=%d)\n",
                schema_V, embed_V, head_V);
        return -1;
    }

    /* embed max_time must be >= model T */
    if (b->embed->max_time < b->model->c.T) {
        fprintf(stderr, "model_bundle_validate: embed max_time=%d < model T=%d\n",
                b->embed->max_time, b->model->c.T);
        return -1;
    }

    /* schema must be valid */
    if (adapter_schema_validate(b->schema) != 0) {
        fprintf(stderr, "model_bundle_validate: schema validation failed\n");
        return -1;
    }

    return 0;
}

/*
 * Bundle file layout:
 *   magic   : 4 bytes "MBDL"
 *   version : int32
 *   model_size   : int64 (byte count of embedded model data)
 *   embed_size   : int64 (byte count of embedded EventEmbed data)
 *   head_size    : int64 (byte count of embedded EventHead data)
 *   schema_size  : int64 (byte count of embedded AdapterSchema data)
 *   model_data   : raw bytes
 *   embed_data   : raw bytes
 *   head_data    : raw bytes
 *   schema_data  : raw bytes
 *
 * Each component's data is the same format as its standalone save.
 * Components are written directly without temp files; sizes are computed
 * after writing and patched back into the header.
 */

/*
 * Validate bundle component sizes.
 * file_sz: actual file size for load (-1 to skip file size check for save)
 * Returns 0 if valid, -1 if invalid.
 */
static int bundle_sizes_valid(int64_t sz_model, int64_t sz_embed,
                              int64_t sz_head, int64_t sz_schema,
                              int64_t file_sz) {
    /* each component must be positive and within limit */
    if (sz_model <= 0 || sz_model > BUNDLE_MAX_COMPONENT_SIZE) return -1;
    if (sz_embed <= 0 || sz_embed > BUNDLE_MAX_COMPONENT_SIZE) return -1;
    if (sz_head <= 0 || sz_head > BUNDLE_MAX_COMPONENT_SIZE) return -1;
    if (sz_schema <= 0 || sz_schema > BUNDLE_MAX_COMPONENT_SIZE) return -1;

    /* overflow check: add pairwise */
    int64_t sum_a = sz_model + sz_embed;
    if (sum_a < 0 || sum_a > BUNDLE_MAX_TOTAL_SIZE) return -1;
    int64_t sum_b = sz_head + sz_schema;
    if (sum_b < 0 || sum_b > BUNDLE_MAX_TOTAL_SIZE) return -1;
    int64_t total = sum_a + sum_b;
    if (total < 0 || total > BUNDLE_MAX_TOTAL_SIZE) return -1;

    /* file size check for load */
    if (file_sz >= 0) {
        int64_t expected = BUNDLE_HEADER_SIZE + total;
        if (file_sz != expected) return -1;
    }

    return 0;
}

#ifdef _WIN32
#include <process.h>
#define GET_PID() _getpid()
__declspec(dllimport) int __stdcall MoveFileExA(const char *existing_name,
                                                const char *new_name,
                                                unsigned long flags);
#define MOVEFILE_REPLACE_EXISTING 0x00000001UL
#define MOVEFILE_WRITE_THROUGH    0x00000008UL
#else
#include <unistd.h>
#define GET_PID() getpid()
#endif

static int g_bundle_counter = 0;
#define TEMP_NAME_SIZE 64

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

static int make_temp_bundle_name(char *out, size_t out_size) {
    int pid = GET_PID();
    const int max_attempts = 100;
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        int cnt = g_bundle_counter++;
        int n = snprintf(out, out_size, "_tmp_bdl_%d_%d_out.bin", pid, cnt);
        if (n <= 0 || (size_t)n >= out_size) return -1;
        if (!file_exists(out)) return 0;
    }
    return -1;
}

static int replace_file_atomic(const char *src, const char *dst) {
#ifdef _WIN32
    return MoveFileExA(src, dst, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
#else
    return rename(src, dst);
#endif
}

static int64_t stream_pos(FILE *f) {
    long pos = ftell(f);
    if (pos < 0) return -1;
    return (int64_t)pos;
}

/* forward declarations for internal *_to_fp / *_from_fp helpers */
int model_save_fp(const Model *m, FILE *f);
Model *model_load_fp(FILE *f);
int event_embed_save_fp(const EventEmbed *e, FILE *f);
EventEmbed *event_embed_load_fp(FILE *f);
int event_head_save_fp(const EventHead *h, FILE *f);
EventHead *event_head_load_fp(FILE *f);

int model_bundle_save(const ModelBundle *b, const char *path) {
    if (!path) return -1;
    if (model_bundle_validate(b) != 0) return -1;

    char tmp_bundle[TEMP_NAME_SIZE];
    if (make_temp_bundle_name(tmp_bundle, sizeof(tmp_bundle)) != 0) return -1;

    FILE *f = fopen(tmp_bundle, "w+b");  /* w+b for read/write/seek */
    if (!f) return -1;

    /* write header with placeholder sizes */
    int ver = BUNDLE_VERSION;
    int64_t zero = 0;
    int ok = 1;
    ok &= (fwrite(BUNDLE_MAGIC, 1, 4, f) == 4);
    ok &= (fwrite(&ver, sizeof(int), 1, f) == 1);
    int64_t size_offset = stream_pos(f);  /* remember position of size fields */
    ok &= (fwrite(&zero, sizeof(int64_t), 1, f) == 1);  /* model_size */
    ok &= (fwrite(&zero, sizeof(int64_t), 1, f) == 1);  /* embed_size */
    ok &= (fwrite(&zero, sizeof(int64_t), 1, f) == 1);  /* head_size */
    ok &= (fwrite(&zero, sizeof(int64_t), 1, f) == 1);  /* schema_size */
    if (!ok || size_offset < 0) { fclose(f); remove(tmp_bundle); return -1; }

    /* write components directly, tracking positions */
    int64_t pos_model_start = stream_pos(f);
    ok &= (model_save_fp(b->model, f) == 0);
    int64_t pos_model_end = stream_pos(f);

    int64_t pos_embed_start = pos_model_end;
    ok &= (event_embed_save_fp(b->embed, f) == 0);
    int64_t pos_embed_end = stream_pos(f);

    int64_t pos_head_start = pos_embed_end;
    ok &= (event_head_save_fp(b->head, f) == 0);
    int64_t pos_head_end = stream_pos(f);

    int64_t pos_schema_start = pos_head_end;
    ok &= (adapter_schema_save_to_fp(b->schema, f) == 0);
    int64_t pos_schema_end = stream_pos(f);

    if (!ok || pos_model_start < 0 || pos_model_end < 0 ||
        pos_embed_end < 0 || pos_head_end < 0 || pos_schema_end < 0) {
        fclose(f); remove(tmp_bundle); return -1;
    }

    /* compute sizes */
    int64_t sz_model  = pos_model_end - pos_model_start;
    int64_t sz_embed  = pos_embed_end - pos_embed_start;
    int64_t sz_head   = pos_head_end - pos_head_start;
    int64_t sz_schema = pos_schema_end - pos_schema_start;

    /* validate sizes */
    if (bundle_sizes_valid(sz_model, sz_embed, sz_head, sz_schema, -1) != 0) {
        fclose(f); remove(tmp_bundle); return -1;
    }

    /* seek back and write actual sizes */
    if (fseek(f, size_offset, SEEK_SET) != 0) { fclose(f); remove(tmp_bundle); return -1; }
    ok &= (fwrite(&sz_model, sizeof(int64_t), 1, f) == 1);
    ok &= (fwrite(&sz_embed, sizeof(int64_t), 1, f) == 1);
    ok &= (fwrite(&sz_head, sizeof(int64_t), 1, f) == 1);
    ok &= (fwrite(&sz_schema, sizeof(int64_t), 1, f) == 1);

    fclose(f);
    if (!ok) { remove(tmp_bundle); return -1; }

    /* atomic rename */
    if (replace_file_atomic(tmp_bundle, path) != 0) { remove(tmp_bundle); return -1; }

    return 0;
}

ModelBundle *model_bundle_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    /* get file size for consistency check */
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long pos = ftell(f);
    if (pos < 0) { fclose(f); return NULL; }
    int64_t file_sz = (int64_t)pos;
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }

    char magic[4];
    int ver;
    int64_t sz_model, sz_embed, sz_head, sz_schema;

    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, BUNDLE_MAGIC, 4) != 0) {
        fclose(f); return NULL;
    }
    if (fread(&ver, sizeof(int), 1, f) != 1 || ver != BUNDLE_VERSION) {
        fclose(f); return NULL;
    }
    if (fread(&sz_model, sizeof(int64_t), 1, f) != 1 ||
        fread(&sz_embed, sizeof(int64_t), 1, f) != 1 ||
        fread(&sz_head, sizeof(int64_t), 1, f) != 1 ||
        fread(&sz_schema, sizeof(int64_t), 1, f) != 1) {
        fclose(f); return NULL;
    }

    /* validate sizes */
    if (bundle_sizes_valid(sz_model, sz_embed, sz_head, sz_schema, file_sz) != 0) {
        fclose(f); return NULL;
    }

    /* load components directly and verify each consumes its declared byte count. */
    int64_t before = stream_pos(f);
    Model *m = model_load_fp(f);
    int64_t after = stream_pos(f);
    if (!m || before < 0 || after < 0 || after - before != sz_model) {
        if (m) model_del(m);
        fclose(f);
        return NULL;
    }

    before = stream_pos(f);
    EventEmbed *e = event_embed_load_fp(f);
    after = stream_pos(f);
    if (!e || before < 0 || after < 0 || after - before != sz_embed) {
        model_del(m);
        if (e) event_embed_del(e);
        fclose(f);
        return NULL;
    }

    before = stream_pos(f);
    EventHead *h = event_head_load_fp(f);
    after = stream_pos(f);
    if (!h || before < 0 || after < 0 || after - before != sz_head) {
        model_del(m);
        event_embed_del(e);
        if (h) event_head_del(h);
        fclose(f);
        return NULL;
    }

    before = stream_pos(f);
    AdapterSchema *s = adapter_schema_load_from_fp(f);
    after = stream_pos(f);
    if (!s || before < 0 || after < 0 || after - before != sz_schema) {
        model_del(m);
        event_embed_del(e);
        event_head_del(h);
        if (s) adapter_schema_del(s);
        fclose(f);
        return NULL;
    }

    fclose(f);

    ModelBundle *b = model_bundle_new(m, e, h, s);

    /* validate component consistency */
    if (model_bundle_validate(b) != 0) {
        model_bundle_del(b);
        return NULL;
    }

    return b;
}
