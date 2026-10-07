/* libFuzzer target for the untrusted-image path: decode (third_party/iris PNG and JPEG, through
 * clef_image.c with its zlib hooks), smart_resize, the antialiased bicubic resize, patches and the
 * position-table interpolation. Built with ASan and UBSan and no recovery, so undefined behavior
 * is a crash. The source limit is 4 Mpx here so mutated headers cannot spend the run allocating;
 * the decoders compare it at the same point they compare the production limit. PNG inputs are
 * also decoded with repaired checksums (png_fixup), or the fuzzer would rarely get past them.
 *   make fuzz-image
 *   .venv/bin/python -B tests/fuzz_image_seeds.py fuzz/seeds
 *   ./fuzz-image -dict=tests/fuzz_image.dict -fork=14 -ignore_crashes=1 -max_total_time=1800 fuzz/corpus fuzz/seeds
 * Not part of make test: it needs Homebrew LLVM (Apple's clang ships without libFuzzer). */
#include "../clef_image.c"

static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static void put_be32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }

/* A mutated PNG almost always fails a chunk CRC or the zlib trailer, so without help the fuzzer
 * never gets past the checksums into inflate output, filtering and color conversion. This copy has
 * them repaired: the zlib header check bits (dictionary flag cleared), the Adler-32 trailer from a
 * raw inflate of the concatenated IDAT data (when the deflate stream ends exactly before it), and
 * every chunk's CRC. Production code is untouched; the original bytes are decoded as well. */
static uint8_t *png_fixup(const uint8_t *d, size_t n) {
    if (n < 8 || memcmp(d, "\x89PNG\r\n\x1a\n", 8)) return NULL;
    uint8_t *p = malloc(n);
    if (!p) return NULL;
    memcpy(p, d, n);
    size_t at[64], len[64], total = 0;
    int nr = 0;
    for (size_t pos = 8; pos + 12 <= n;) {
        const uint32_t l = be32(p + pos);
        if (l > n - pos - 12) break;
        if (!memcmp(p + pos + 4, "IDAT", 4) && nr < 64) { at[nr] = pos + 8; len[nr] = l; nr++; total += l; }
        pos += 12 + (size_t)l;
    }
    if (total >= 7) {
        uint8_t *z = malloc(total);
        size_t off = 0;
        for (int i = 0; i < nr; i++) { memcpy(z + off, p + at[i], len[i]); off += len[i]; }
        z[1] = (uint8_t)(z[1] & 0xc0);
        z[1] = (uint8_t)(z[1] | (31 - (z[0] * 256u + z[1]) % 31) % 31);
        z_stream s;
        memset(&s, 0, sizeof(s));
        if (inflateInit2(&s, -15) == Z_OK) {
            size_t cap = 1u << 16, got = 0;
            uint8_t *out = malloc(cap);
            int r = Z_OK;
            s.next_in = z + 2;
            s.avail_in = (uInt)(total - 2);
            while (out && r == Z_OK) {
                if (got == cap) {
                    if (cap >= (32u << 20)) break;
                    uint8_t *grown = realloc(out, cap * 2);
                    if (!grown) break;
                    out = grown;
                    cap *= 2;
                }
                s.next_out = out + got;
                s.avail_out = (uInt)(cap - got);
                r = inflate(&s, Z_NO_FLUSH);
                got = cap - s.avail_out;
            }
            const size_t used = (size_t)(total - 2) - s.avail_in;
            if (r == Z_STREAM_END && 2 + used + 4 == total) put_be32(z + 2 + used, (uint32_t)adler32(1, out, (uInt)got));
            inflateEnd(&s);
            free(out);
        }
        off = 0;
        for (int i = 0; i < nr; i++) { memcpy(p + at[i], z + off, len[i]); off += len[i]; }
        free(z);
    }
    for (size_t pos = 8; pos + 12 <= n;) {
        const uint32_t l = be32(p + pos);
        if (l > n - pos - 12) break;
        put_be32(p + pos + 8 + l, (uint32_t)crc32(0, p + pos + 4, 4 + l));
        pos += 12 + (size_t)l;
    }
    return p;
}

static void decode_and_preprocess(const uint8_t *data, size_t size);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    decode_and_preprocess(data, size);
    uint8_t *fixed = png_fixup(data, size);
    if (fixed) {
        decode_and_preprocess(fixed, size);
        free(fixed);
    }
    return 0;
}

static void decode_and_preprocess(const uint8_t *data, size_t size) {
    clef_rgb rgb;
    char err[256];
    if (!clef_image_decode_limited(data, size, 1L << 22, &rgb, err, sizeof(err))) return;
    /* a small pixel budget with a floor above some decoded sizes: both down- and upscaling */
    const clef_image_params p = { .min_pixels = 4096, .max_pixels = 1L << 18, .patch = 16, .merge = 2, .temporal = 2 };
    clef_image_patches pt;
    if (clef_image_preprocess(&rgb, &p, &pt, err, sizeof(err))) {
        int32_t *idx = malloc((size_t)pt.n_patch * 4 * sizeof(*idx));
        float *w = malloc((size_t)pt.n_patch * 4 * sizeof(*w));
        if (idx && w) clef_image_pos_interp(pt.grid_h, pt.grid_w, 48, p.merge, idx, w);
        free(idx);
        free(w);
        clef_image_patches_free(&pt);
    }
    clef_rgb_free(&rgb);
}
