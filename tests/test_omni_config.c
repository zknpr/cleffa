/* Host-only validation: no tensor data, model loading, or GPU work. */
#include <assert.h>
#include "../clef.c"

int main(int argc, char **argv) {
    char err[512] = "";
    if (argc == 2 && !strcmp(argv[1], "rank")) {
        gguf_tensor tensor = { .name = {"matrix", 6}, .n_dims = 3, .ne = {8, 2, 2, 1}, .type = GGML_BF16 };
        gguf_file f = { .tensors = &tensor, .n_tensors = 1 };
        assert(!bind(&f, "matrix", GGML_BF16, 8, 2, err, sizeof(err)));
        assert(bind_experts(&f, "matrix", 8, 2, 2, err, sizeof(err)));
        tensor.ne[2] = 3;
        assert(!bind_experts(&f, "matrix", 8, 2, 2, err, sizeof(err)));
        tensor.ne[2] = 2;
        tensor.n_dims = 4;
        assert(!bind_experts(&f, "matrix", 8, 2, 2, err, sizeof(err)));
        tensor.ne[0]=3;tensor.ne[1]=3;tensor.ne[2]=480;tensor.ne[3]=480;
        assert(bind_audio_conv(&f,"matrix",480,err,sizeof(err)));
        tensor.n_dims=3;
        assert(!bind_audio_conv(&f,"matrix",480,err,sizeof(err)));
        clef_model_info m;
        f.n_tensors = 0;
        assert(clef_model_info_load(&f, &m, err, sizeof(err)));
        assert(m.backbone == CLEF_QWEN35 && m.context_default == 16384 && m.context_length == 0);
        return 0;
    }
    if (argc == 2 && !strcmp(argv[1], "audio")) {
        const char *json = "{\"model\":\"m\",\"state\":\"s\",\"audio\":[0],\"questions\":{\"q\":{\"type\":\"unknown\"}}}";
        jarena *a = jarena_new();
        jval *req = json_parse(a, json, strlen(json), err, sizeof(err));
        clef_record rec;
        clef_encode_opts opts = CLEF_ENCODE_DEFAULTS;
        assert(!clef_encode_request(NULL, req, opts, &rec, err, sizeof(err)));
        assert(strstr(err, "audio"));
        jarena_free(a);
        return 0;
    }
    if (argc != 3) return 2;
    gguf_file f;
    if (!gguf_open(&f, argv[1], err, sizeof(err))) { fprintf(stderr, "%s\n", err); return 1; }
    clef_config c = {0};
    const bool audio_only = !strncmp(argv[2],"audio:",6);
    const char *expected = audio_only ? argv[2]+6 : argv[2];
    bool ok = audio_only ? clef_audio_config_load(&f,&c.audio,err,sizeof(err)) : load_config(&f, &c, err, sizeof(err));
    clef_vision_opts v = {0};
    if (ok && !audio_only) ok = clef_vision_opts_load(&f,&v,err,sizeof(err));
    if (audio_only && ok && !strcmp(expected,"ok")) {
        assert(c.audio.layers==32 && c.audio.H==1280 && c.audio.hd==64 && c.audio.eps==1e-5f);
    } else if (ok && !strcmp(expected, "ok")) {
        assert(c.hd == 128 && c.n_rot == 128 && c.H == 2048);
        assert(c.model.backbone == CLEF_OMNI && c.model.context_default == 64000);
        assert(c.model.context_length == 65536 && c.n_expert == 128 && c.n_expert_used == 8);
        assert(!!v.image_token_id == !!c.has_vision);
        if (c.has_vision) assert(c.v_deepstack[0]==8 && c.v_deepstack[1]==16 && c.v_deepstack[2]==24);
    } else if (ok || !strcmp(expected, "ok") || !strstr(err, expected)) {
        fprintf(stderr, "%s: %s; expected %s\n", argv[1], ok ? "accepted" : err, expected);
        gguf_close(&f);
        return 1;
    }
    gguf_close(&f);
    return 0;
}
