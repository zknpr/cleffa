#include <stdio.h>
#include <string.h>
#include "clef_model.h"
#include "clef_audio.h"

bool clef_model_info_load(const gguf_file *f, clef_model_info *out, char *err, size_t errlen) {
    clef_model_info m = { .backbone = CLEF_QWEN35, .capabilities = CLEF_CAP_TEXT, .context_default = 16384 };
    const char *keys[] = { "clef.backbone", "clef.format_version", "clef.capabilities", "clef.context_default", "clef.context_length" };
    int present = 0;
    for (size_t i = 0; i < sizeof(keys)/sizeof(keys[0]); i++) present += gguf_find_kv(f, keys[i]) != NULL;
    if (!present) {
        if (gguf_find_kv(f, "clef.moe.expert_count") || gguf_find_tensor(f, "blk.0.ffn_router.weight")) {
            snprintf(err, errlen, "model: Omni weights require family/format metadata");
            return false;
        }
        if (gguf_find_kv(f, "clef.vision.block_count")) m.capabilities |= CLEF_CAP_VISION;
        *out = m;
        return true;
    }
    gguf_str family;
    uint32_t version, caps, def, limit;
    if (present != 5 || !gguf_get_str(f, "clef.backbone", &family) || family.len != 14 ||
        memcmp(family.ptr, "qwen3_omni_moe", 14) ||
        !gguf_get_u32(f, "clef.format_version", &version) || version != 1 ||
        !gguf_get_u32(f, "clef.capabilities", &caps) || (caps != 1 && caps != 3 && caps != 7) ||
        !gguf_get_u32(f, "clef.context_default", &def) || !def ||
        !gguf_get_u32(f, "clef.context_length", &limit) || limit != 65536 || def > limit) {
        snprintf(err, errlen, "model: incomplete or unsupported family, format, capabilities or context metadata");
        return false;
    }
    m = (clef_model_info){ .backbone = CLEF_OMNI, .capabilities = caps,
                         .context_default = (int)def, .context_length = (int)limit };
    *out = m;
    return true;
}

static bool audio_integer(const jval *object, const char *key, int expected) {
    const jval *v = json_get(object,key);
    char text[24];
    int length = snprintf(text,sizeof(text),"%d",expected);
    return v && v->type == J_INT && v->len == (size_t)length && !memcmp(v->s,text,v->len);
}

static bool audio_zero(const jval *object, const char *key) {
    const jval *v = json_get(object,key);
    return audio_integer(object,key,0) || (v && v->type == J_FLOAT && v->f == 0);
}

bool clef_audio_config_load(const gguf_file *f, clef_audio_config *out, char *err, size_t errlen) {
    gguf_str config,processor;
    if (!gguf_get_str(f,"clef.audio.config",&config) || !gguf_get_str(f,"clef.audio.processor",&processor) ||
        config.len > 65536 || processor.len > 65536) {
        snprintf(err,errlen,"model: missing or oversized Omni audio metadata");return false;
    }
    jarena *arena = jarena_new();
    if (!arena) { snprintf(err,errlen,"model: cannot allocate audio metadata parser");return false; }
    jval *a = json_parse(arena,config.ptr,config.len,err,errlen);
    jval *p = json_parse(arena,processor.ptr,processor.len,err,errlen);
    bool ok = a && p && a->type == J_OBJECT && p->type == J_OBJECT;
    const struct { const char *name; int value; } geometry[] = {
        {"d_model",1280},{"encoder_layers",32},{"encoder_attention_heads",20},{"encoder_ffn_dim",5120},
        {"num_mel_bins",128},{"downsample_hidden_size",480},{"output_dim",2048},
        {"n_window",50},{"n_window_infer",800},{"conv_chunksize",500},{"max_source_positions",1500}
    }, frontend[] = {{"sampling_rate",16000},{"feature_size",128},{"n_fft",400},{"hop_length",160},
                     {"chunk_length",30},{"n_samples",480000},{"nb_max_frames",3000}};
    for (size_t i = 0; ok && i < sizeof(geometry)/sizeof(geometry[0]); i++) ok = audio_integer(a,geometry[i].name,geometry[i].value);
    for (size_t i = 0; ok && i < sizeof(frontend)/sizeof(frontend[0]); i++) ok = audio_integer(p,frontend[i].name,frontend[i].value);
    const jval *scale = json_get(a,"scale_embedding"), *mask = json_get(p,"return_attention_mask");
    uint32_t temporal,chunk;
    ok = ok && json_str_eq(json_get(a,"model_type"),"qwen3_omni_moe_audio_encoder") &&
         json_str_eq(json_get(a,"activation_function"),"gelu") && (!scale || scale->type == J_FALSE) &&
         json_str_eq(json_get(p,"feature_extractor_type"),"WhisperFeatureExtractor") &&
         json_str_eq(json_get(p,"padding_side"),"right") && audio_zero(p,"padding_value") && audio_zero(p,"dither") &&
         mask && mask->type == J_TRUE && gguf_get_u32(f,"clef.audio.position_id_per_seconds",&temporal) && temporal == 13 &&
         gguf_get_u32(f,"clef.audio.seconds_per_chunk",&chunk) && chunk == 2;
    jarena_free(arena);
    if (!ok) { snprintf(err,errlen,"model: unsupported Omni audio geometry or frontend configuration");return false; }
    *out = (clef_audio_config){.layers=32,.H=1280,.ffn=5120,.heads=20,.hd=64,.mel_bins=128,.conv_channels=480,
        .output=2048,.chunk_frames=100,.window_frames=800,.conv_chunks=500,.max_positions=1500,.eps=1e-5f};
    return true;
}

