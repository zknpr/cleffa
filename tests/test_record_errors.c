/* Error buffers of every small capacity must contain complete UTF-8, including
 * when truncation happens before the encoder's usual 200-byte question-id cap. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../clef_record.h"

int main(void) {
    const char *chars[] = { "é", "界", "🚀" };
    int cases = 0;
    for (size_t c = 0; c < sizeof(chars) / sizeof(chars[0]); c++) {
        char id[512] = "x", request[640];
        for (int i = 0; i < 100; i++) strcat(id, chars[c]);
        snprintf(request, sizeof(request), "{\"model\":\"m\",\"state\":\"s\",\"questions\":{\"%s\":{\"type\":\"unknown\"}}}", id);
        jarena *a = jarena_new();
        char err[256];
        jval *req = json_parse(a, request, strlen(request), err, sizeof(err));
        assert(req);
        for (size_t cap = 0; cap <= sizeof(err); cap++) {
            memset(err, 0x7f, sizeof(err));
            clef_record rec;
            clef_encode_opts opts = CLEF_ENCODE_DEFAULTS;
            /* Validation precedes tokenization, so this needs no model. */
            assert(!clef_encode_request(NULL, req, opts, &rec, err, cap));
            for (size_t i = cap; i < sizeof(err); i++) assert(err[i] == 0x7f);
            if (cap) {
                assert(memchr(err, 0, cap));
                jbuf b = {0};
                json_put_string(&b, err, strlen(err));
                jarena *check = jarena_new();
                char parse_err[128];
                assert(json_parse(check, b.p, b.len, parse_err, sizeof(parse_err)));
                jarena_free(check);
                jbuf_free(&b);
            }
            cases++;
        }
        jarena_free(a);
    }
    printf("record errors: %d UTF-8/buffer-boundary cases passed\n", cases);
    return 0;
}
