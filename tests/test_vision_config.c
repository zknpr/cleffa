/* Vision geometry from an untrusted GGUF is validated before it is multiplied, the M-RoPE
 * layout must be the one the kernels implement, and vision token ids must be in the vocabulary
 * (review #3; load_config, then clef_vision_opts_load).
 * cfg_u32 admits each field up to 2^24, and 3 * temporal * patch * patch overflowed int before
 * the fixed-shape check rejected the file. Built with UBSan and -fno-sanitize-recover, so a
 * signed overflow aborts instead of passing. No Metal device or tensor data is touched.
 * Usage: test-vision-config FILE.gguf EXPECT ...  (EXPECT: "ok", or a substring of the error)
 * Fixtures: tests/vision_config_fixture.py MODEL.gguf DIR */
#include "../clef.c"

int main(int argc, char **argv) {
    if (argc < 3 || argc % 2 == 0) { fprintf(stderr, "usage: test-vision-config FILE.gguf EXPECT ...\n"); return 2; }
    for (int i = 1; i + 1 < argc; i += 2) {
        gguf_file f;
        char err[256] = "";
        if (!gguf_open(&f, argv[i], err, sizeof(err))) { fprintf(stderr, "%s: %s\n", argv[i], err); return 1; }
        clef_config c;
        memset(&c, 0, sizeof(c));
        clef_vision_opts vo;
        memset(&vo, 0, sizeof(vo));
        const bool ok = load_config(&f, &c, err, sizeof(err)) && clef_vision_opts_load(&f, &vo, err, sizeof(err));
        gguf_close(&f);
        const bool want_ok = !strcmp(argv[i + 1], "ok");
        if (ok != want_ok || (!ok && !strstr(err, argv[i + 1]))) {
            fprintf(stderr, "vision config: %s: %s, expected %s\n", argv[i], ok ? "loaded" : err, argv[i + 1]);
            return 1;
        }
        if (ok && (!c.has_vision || c.v_in != 3 * 2 * 16 * 16)) { fprintf(stderr, "vision config: %s: v_in %d\n", argv[i], c.v_in); return 1; }
    }
    printf("vision config: %d GGUF headers, geometry validated before use (UBSan)\n", (argc - 1) / 2);
    return 0;
}
