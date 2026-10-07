/* Crafted JPEGs that drive the vendored decoder (third_party/iris/jpeg.h) into undefined
 * behavior (review #3). Each is small and structurally plausible; before the fixes listed in
 * jpeg.h's header comment every case aborted under UBSan:
 *   1 undefined-dc-table  SOS selects a DC table no DHT defined: the zeroed table read values[-1]
 *   2 dc-symbol-255       a DHT maps a DC code to 255, then shifts by 255 bits
 *   3 idct-overflow       1x1 baseline, DC difference 2047 at quantizer 255: pass 2 overflowed int
 *   4 dc-accumulation     1024x512, the same difference in every block: DC prediction times the
 *                         quantizer overflowed int long before the IDCT
 *   5 progressive-dc      128x128 DC-only progressive scan, Al = 13: prediction << Al overflowed
 *   6 duplicate-scan      a 3-component baseline SOS naming component 1 three times, only table 1
 *                         defined: components 2 and 3 decoded with undefined table 0 (values[-1])
 * Cases 1, 2 and 6 must be refused, 3-5 must decode (the content is out of range, the syntax is not).
 * Built with UBSan and ASan and -fno-sanitize-recover; includes clef_image.c so the decoders are
 * instrumented. Usage: test-jpeg-ub [CASE...]  (all cases without arguments) */
#include "../clef_image.c"

#include <stdio.h>
#include <stdlib.h>

typedef struct { uint8_t *p; size_t n, cap; uint32_t acc; int nbits; } jbuf_t;

static void put(jbuf_t *j, const void *src, size_t n) {
    if (j->n + n > j->cap) {
        j->cap = (j->n + n) * 2 + 256;
        j->p = realloc(j->p, j->cap);
        if (!j->p) { fprintf(stderr, "jpeg ub: out of memory\n"); exit(1); }
    }
    memcpy(j->p + j->n, src, n);
    j->n += n;
}
static void byte(jbuf_t *j, uint8_t b) { put(j, &b, 1); }
static void seg(jbuf_t *j, uint8_t marker, const uint8_t *payload, size_t n) {
    const uint8_t h[4] = { 0xff, marker, (uint8_t)((n + 2) >> 8), (uint8_t)(n + 2) };
    put(j, h, 4);
    put(j, payload, n);
}
/* entropy-coded bits, MSB first, with 0xFF byte stuffing */
static void bits(jbuf_t *j, uint32_t v, int n) {
    for (int i = n - 1; i >= 0; i--) {
        j->acc = j->acc << 1 | ((v >> i) & 1);
        if (++j->nbits == 8) {
            byte(j, (uint8_t)j->acc);
            if ((uint8_t)j->acc == 0xff) byte(j, 0x00);
            j->acc = 0;
            j->nbits = 0;
        }
    }
}
static void flush(jbuf_t *j) { while (j->nbits) bits(j, 1, 1); }

/* One grayscale frame: quantizer 1 everywhere but qt0 at DC; DC table 0 has one 1-bit code
 * ('0') for dc_symbol; AC table 0 one 1-bit code ('0') for EOB; the scan selects DC table
 * dc_select. Every block codes DC difference +2047 (category 11) unless zero_scan, and ends
 * with EOB in a baseline scan. */
static jbuf_t frame(int w, int h, uint8_t qt0, uint8_t dc_symbol, int dc_select, bool progressive, bool zero_scan) {
    jbuf_t j = {0};
    const uint8_t soi[2] = { 0xff, 0xd8 };
    put(&j, soi, 2);
    uint8_t dqt[65] = { 0x00 };
    for (int i = 1; i < 65; i++) dqt[i] = 1;
    dqt[1] = qt0;
    seg(&j, 0xdb, dqt, sizeof(dqt));
    const uint8_t sof[9] = { 8, (uint8_t)(h >> 8), (uint8_t)h, (uint8_t)(w >> 8), (uint8_t)w, 1, 1, 0x11, 0 };
    seg(&j, progressive ? 0xc2 : 0xc0, sof, sizeof(sof));
    uint8_t dht_dc[18] = { 0x00, 1 };
    dht_dc[17] = dc_symbol;
    seg(&j, 0xc4, dht_dc, sizeof(dht_dc));
    uint8_t dht_ac[18] = { 0x10, 1 };
    dht_ac[17] = 0x00;
    seg(&j, 0xc4, dht_ac, sizeof(dht_ac));
    const uint8_t sos[6] = { 1, 1, (uint8_t)(dc_select << 4), 0, (uint8_t)(progressive ? 0 : 63), (uint8_t)(progressive ? 13 : 0) };
    seg(&j, 0xda, sos, sizeof(sos));
    const long blocks = (long)((w + 7) / 8) * ((h + 7) / 8);
    if (zero_scan) {
        for (int i = 0; i < 8; i++) byte(&j, 0x00);
    } else {
        for (long b = 0; b < blocks; b++) {
            bits(&j, 0, 1);              /* DC code */
            bits(&j, 0x7ff, 11);         /* +2047 */
            if (!progressive) bits(&j, 0, 1);   /* EOB */
        }
        flush(&j);
    }
    const uint8_t eoi[2] = { 0xff, 0xd9 };
    put(&j, eoi, 2);
    return j;
}

/* Three-component baseline frame whose SOS names component 1 three times, with only table 1
 * defined: the scan's own table check passes, but baseline decoding walks every frame component
 * and components 2 and 3 kept table 0, which was never defined (review #3, round 4). */
static jbuf_t duplicate_scan_components(void) {
    jbuf_t j = {0};
    const uint8_t soi[2] = { 0xff, 0xd8 };
    put(&j, soi, 2);
    uint8_t dqt[65] = { 0x00 };
    for (int i = 1; i < 65; i++) dqt[i] = 1;
    seg(&j, 0xdb, dqt, sizeof(dqt));
    const uint8_t sof[15] = { 8, 0, 8, 0, 8, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0 };
    seg(&j, 0xc0, sof, sizeof(sof));
    uint8_t dht_dc[18] = { 0x01, 1 };
    seg(&j, 0xc4, dht_dc, sizeof(dht_dc));
    uint8_t dht_ac[18] = { 0x11, 1 };
    seg(&j, 0xc4, dht_ac, sizeof(dht_ac));
    const uint8_t sos[10] = { 3, 1, 0x11, 1, 0x11, 1, 0x11, 0, 63, 0 };
    seg(&j, 0xda, sos, sizeof(sos));
    for (int i = 0; i < 8; i++) byte(&j, 0x00);
    const uint8_t eoi[2] = { 0xff, 0xd9 };
    put(&j, eoi, 2);
    return j;
}

static int run(int c) {
    static const char *names[] = { "", "undefined-dc-table", "dc-symbol-255", "idct-overflow", "dc-accumulation", "progressive-dc", "duplicate-scan-components" };
    jbuf_t j;
    switch (c) {
    case 1: j = frame(8, 8, 1, 0, 1, false, true); break;
    case 2: j = frame(8, 8, 1, 255, 0, false, true); break;
    case 3: j = frame(1, 1, 255, 11, 0, false, false); break;
    case 4: j = frame(1024, 512, 255, 11, 0, false, false); break;
    case 5: j = frame(128, 128, 1, 11, 0, true, false); break;
    case 6: j = duplicate_scan_components(); break;
    default: fprintf(stderr, "jpeg ub: no case %d\n", c); return 1;
    }
    clef_rgb rgb;
    char err[256] = "";
    const bool ok = clef_image_decode(j.p, j.n, &rgb, err, sizeof(err));
    free(j.p);
    const bool want_ok = c >= 3 && c <= 5;
    if (ok != want_ok) {
        fprintf(stderr, "jpeg ub: case %d %s: %s\n", c, names[c], ok ? "decoded, expected refusal" : err);
        if (ok) clef_rgb_free(&rgb);
        return 1;
    }
    if (ok && c == 3 && (rgb.width != 1 || rgb.height != 1 || rgb.rgb[0] != 255)) {
        /* a DC level far above white clamps to white once the IDCT no longer wraps */
        fprintf(stderr, "jpeg ub: case 3 pixel %u, expected 255\n", rgb.rgb[0]);
        clef_rgb_free(&rgb);
        return 1;
    }
    if (ok) clef_rgb_free(&rgb);
    printf("jpeg ub: case %d %s: %s\n", c, names[c], ok ? "decoded" : "refused");
    return 0;
}

int main(int argc, char **argv) {
    int failures = 0;
    if (argc > 1) for (int i = 1; i < argc; i++) failures += run(atoi(argv[i]));
    else for (int c = 1; c <= 6; c++) failures += run(c);
    return failures ? 1 : 0;
}
