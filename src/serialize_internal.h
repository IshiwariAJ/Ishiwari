#ifndef SERIALIZE_INTERNAL_H
#define SERIALIZE_INTERNAL_H

#include "event.h"
#include "model.h"
#include <stdio.h>

/*
 * Internal FILE* serialization helpers used by ModelBundle.
 * Public callers should keep using model_save/model_load and
 * event_embed_save/event_embed_load unless they need to write into an
 * already-open container file.
 */

int    model_save_fp(const Model *m, FILE *f);
Model *model_load_fp(FILE *f);

int         event_embed_save_fp(const EventEmbed *e, FILE *f);
EventEmbed *event_embed_load_fp(FILE *f);

int        event_head_save_fp(const EventHead *h, FILE *f);
EventHead *event_head_load_fp(FILE *f);

#endif
