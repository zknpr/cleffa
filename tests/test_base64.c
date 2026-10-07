#include "../clef_image.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void valid(const char *encoded, const uint8_t *expected, size_t length) {
    uint8_t *out = NULL;
    size_t n = 0;
    char err[256];
    assert(clef_base64_decode(encoded, strlen(encoded), &out, &n, err, sizeof(err)));
    assert(n == length && !memcmp(out, expected, n));
    free(out);
}

static void invalid(const char *encoded, size_t length) {
    uint8_t *out = (uint8_t *)1;
    size_t n = 123;
    char err[256] = {0};
    assert(!clef_base64_decode(encoded, length, &out, &n, err, sizeof(err)));
    assert(out == NULL && n == 0 && err[0]);
}

int main(void) {
    const char *vectors[] = {"", "Zg==", "Zm8=", "Zm9v", "Zm9vYg==", "Zm9vYmE=", "Zm9vYmFy"};
    for (size_t i = 0; i < sizeof(vectors) / sizeof(*vectors); i++)
        valid(vectors[i], (const uint8_t *)"foobar", i);
    valid("data:image/png;base64,Zm9vYmFy", (const uint8_t *)"foobar", 6);
    valid("DATA:image/png;base64,Zm9vYmFy", (const uint8_t *)"foobar", 6);
    const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    /* Every alphabet symbol is checked in all four bit positions, in both the bulk loop
     * and final quartet. Repeated sextets have independently known output bytes. */
    for (unsigned i = 0; i < 64; i++) {
        char encoded[9];
        memset(encoded, alphabet[i], 8); encoded[8] = 0;
        const uint8_t expected[6] = {i * 4 + i / 16, i * 16 + i / 4, i * 64 + i,
                                     i * 4 + i / 16, i * 16 + i / 4, i * 64 + i};
        valid(encoded, expected, sizeof(expected));
    }
    for (unsigned value = 0; value < 256; value++) {
        if (value && strchr(alphabet, (int)value)) continue;
        for (int pos = 0; pos < 8; pos++) {
            if (value == '=' && pos >= 6) continue;   /* valid final padding */
            char encoded[8];
            memset(encoded, 'A', sizeof(encoded));
            encoded[pos] = (char)value;
            invalid(encoded, sizeof(encoded));
        }
    }
    const char *bad[] = {"A", "AA", "AAA", "A===", "AA=A", "=AAA", "AAAA====",
                         "AAAAAA=Z", "AAAA\nAAA", "data:image/png,AAAA", "data:image/png;base64"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) invalid(bad[i], strlen(bad[i]));
    puts("base64: RFC vectors, all alphabet sextets, every invalid byte/position and strict padding passed");
    return 0;
}
