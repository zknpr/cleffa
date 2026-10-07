/*
 * JPEG Decoder - Single-header implementation
 *
 * A dependency-free C implementation for reading JPEG images.
 * Supports baseline and progressive DCT with various chroma subsampling.
 *
 * Usage:
 *   jpeg_image *img = jpeg_load("image.jpg");
 *   if (!img) { handle error }
 *
 *   // Access pixel data (RGB)
 *   uint8_t *pixel = img->data + (y * img->width + x) * img->channels;
 *
 *   jpeg_free(img);
 *
 * To use as header-only, define JPEG_IMPLEMENTATION before including:
 *   #define JPEG_IMPLEMENTATION
 *   #include "jpeg.h"
 *
 * Features:
 *   - Baseline DCT (SOF0)
 *   - Progressive DCT (SOF2)
 *   - 4:4:4, 4:2:2, 4:2:0 chroma subsampling
 *   - Restart markers
 *   - Multi-scan progressive images
 */

#ifndef JPEG_H
#define JPEG_H

#include <stddef.h>
#include <stdint.h>

/* DS4 imports IRIS at commit 9873887d4aa0646c650adc5b86b986d2f653b7e0.
 * Local changes include decode limits, centered 2x chroma interpolation,
 * rounded YCbCr conversion, and retaining buffered progressive scan bits.
 * These limits make the memory decoder suitable for untrusted server input.
 * cleffa (2026-10-07) adds two input checks on top of ds4's copy:
 *   - the scan header's Ss/Se/Ah/Al are validated (spectral range within 0..63,
 *     DC scans alone at Ss = 0, AC scans of one component, baseline 0..63 with no
 *     successive approximation): a progressive scan with Se > 63 drove the
 *     coefficient index past the 64-entry zigzag table and wrote outside the block;
 *   - in a multi-component image the first (luma) component must carry the largest
 *     sampling factors: its plane is allocated from its own factors while the
 *     conversion loop reads it at full image resolution, so a legal but unusual
 *     layout such as Y 1x1 under Cb 2x2 read past the plane. libjpeg decodes such
 *     files; here they are refused (tests/test_image.py).
 * and, after a second review (tests/test_jpeg_ub.c, each case undefined behavior under
 * UBSan before the change):
 *   - a scan may only select Huffman tables a DHT defined, as libjpeg requires: a zeroed
 *     table matched the all-zero code and indexed values[-1]; and it may name each component
 *     once, as libjpeg requires, since a baseline scan naming component 1 three times decoded
 *     the other two with tables the check had not seen;
 *   - DC table symbols above 15 are refused when the table is defined (libjpeg's
 *     jpeg_make_d_derived_tbl bound): they are bit counts, and 255 shifted by 255;
 *   - the DC predictor wraps in unsigned arithmetic and the coefficient is stored as
 *     16 bits before dequantization, as libjpeg stores it in a JCOEF: a repeated
 *     difference of 2047 overflowed int in the prediction, the dequantization and the
 *     progressive shift by Al;
 *   - the IDCT computes in 64-bit, as libjpeg's JLONG does on this platform: a legal
 *     DC difference of 2047 at quantizer 255 overflowed int in the second pass and
 *     wrapped a far-above-white block to black. In-range blocks compute identically;
 *   - a progressive DC scan of one component walks that component's blocks in raster
 *     order, as AC scans already did: walked as MCUs, a 4:2:0 file with a DC scan per
 *     component decoded wrong or was refused, where libjpeg decodes it;
 * and, found by decoding files from other encoders against libjpeg (tests/test_image_diff.py):
 *   - vertical-only 2:1 chroma (4:4:0) uses libjpeg's h1v2 triangle filter instead of row
 *     replication, which was up to 69 levels off;
 *   - SOF1 (extended sequential, 8-bit) decodes as baseline: libjpeg writes it whenever a
 *     quantizer exceeds 255, and it was refused;
 * and, found by tests/fuzz_jpeg_diff.c against libjpeg:
 *   - a refinement run overshooting Se writes its new coefficient at Se + 1 (or 63), and a
 *     first-scan or baseline run overshooting Se (or 63) writes its coefficient at k (or 63),
 *     as libjpeg does through its padded natural order;
 *   - an incomplete progressive image, which libjpeg would reconstruct with block smoothing,
 *     is refused (libjpeg's smoothing_ok() condition);
 *   - a one-component frame decodes as 1x1 whatever sampling factors it declares, as libjpeg
 *     does;
 *   - three components are RGB-coded, and copied, under libjpeg's rule (no JFIF marker, and an
 *     Adobe transform of 0 or, without one, component ids 'R','G','B'); a progressive frame
 *     with no scan is refused, as libjpeg's JERR_SOF_NO_SOS;
 *   - each component's quantization table must be defined by its first scan and is latched
 *     there, as libjpeg does; a block with a dequantized coefficient beyond 16 bits or an IDCT
 *     output beyond [-512, 511] is refused, since libjpeg wraps such outputs where this decoder
 *     clamps;
 *   - a sequential frame may split its components across scans: its coefficients are buffered
 *     and finished like a progressive frame's, as libjpeg does; a scan before the frame header,
 *     or a second frame header, is refused (JERR_SOS_NO_SOF, JERR_SOF_DUPLICATE);
 *   - 0xFF fill bytes before a marker are skipped one at a time, so "FF FF D9" is an EOI;
 *     inside entropy data "FF FF 00" (not standard) is refused, since libjpeg-turbo's own
 *     result for it depends on how its input is buffered. */
#ifndef JPEG_MAX_INPUT_BYTES
#define JPEG_MAX_INPUT_BYTES (64u * 1024u * 1024u)
#endif
#ifndef JPEG_MAX_DIMENSION
#define JPEG_MAX_DIMENSION 16384
#endif
#ifndef JPEG_MAX_PIXELS
#define JPEG_MAX_PIXELS (64u * 1024u * 1024u)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Image Structure
 * ======================================================================== */

typedef struct {
    int width;
    int height;
    int channels;       /* 1=Grayscale, 3=RGB */
    uint8_t *data;      /* Row-major, channel-interleaved */
} jpeg_image;

/* ========================================================================
 * Public API
 * ======================================================================== */

/*
 * Load JPEG image from file.
 * Returns NULL on error.
 */
jpeg_image *jpeg_load(const char *path);

/*
 * Load JPEG image from memory buffer.
 * Returns NULL on error.
 */
jpeg_image *jpeg_load_mem(const uint8_t *data, size_t len);

/*
 * Create a new image with given dimensions.
 * Allocates zeroed pixel data.
 */
jpeg_image *jpeg_create(int width, int height, int channels);

/*
 * Free image and pixel data.
 */
void jpeg_free(jpeg_image *img);

/*
 * Clone an image (deep copy).
 */
jpeg_image *jpeg_clone(const jpeg_image *img);

#ifdef __cplusplus
}
#endif

#endif /* JPEG_H */

/* ========================================================================
 * Implementation
 * ======================================================================== */

#ifdef JPEG_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Image Creation and Management
 * ======================================================================== */

jpeg_image *jpeg_create(int width, int height, int channels) {
    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
        return NULL;
    }
    size_t pixels = (size_t)width * (size_t)height;
    if (pixels > SIZE_MAX / (size_t)channels) return NULL;
    size_t bytes = pixels * (size_t)channels;
    jpeg_image *img = (jpeg_image *)malloc(sizeof(jpeg_image));
    if (!img) return NULL;

    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = (uint8_t *)calloc(bytes, sizeof(uint8_t));

    if (!img->data) {
        free(img);
        return NULL;
    }

    return img;
}

void jpeg_free(jpeg_image *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

jpeg_image *jpeg_clone(const jpeg_image *img) {
    if (!img) return NULL;

    jpeg_image *clone = jpeg_create(img->width, img->height, img->channels);
    if (!clone) return NULL;

    size_t bytes = (size_t)img->width * (size_t)img->height *
                   (size_t)img->channels;
    memcpy(clone->data, img->data, bytes);
    return clone;
}

/* ========================================================================
 * JPEG Decoder Internals
 * ======================================================================== */

/* JPEG markers */
#define JPEG_SOI  0xD8  /* Start of image */
#define JPEG_EOI  0xD9  /* End of image */
#define JPEG_SOF0 0xC0  /* Baseline DCT */
#define JPEG_SOF1 0xC1  /* Extended sequential DCT, Huffman (cleffa: 8-bit only, decoded as baseline) */
#define JPEG_SOF2 0xC2  /* Progressive DCT */
#define JPEG_DHT  0xC4  /* Define Huffman table */
#define JPEG_DQT  0xDB  /* Define quantization table */
#define JPEG_DRI  0xDD  /* Define restart interval */
#define JPEG_SOS  0xDA  /* Start of scan */
#define JPEG_RST0 0xD0  /* Restart marker 0 */

/* Clamp value to 0-255 range */
#define JPEG_CLAMP(x) ((x) < 0 ? 0 : ((x) > 255 ? 255 : (x)))

typedef struct {
    uint8_t bits[16];      /* Number of codes of each length 1-16 */
    uint8_t values[256];   /* Symbol values */
    /* Derived lookup tables */
    int maxcode[17];       /* Max code for each length, -1 if none */
    int valptr[17];        /* Index into values for codes of length i */
    int lookup[256];       /* Fast lookup for short codes: (length << 8) | symbol */
} jpeg_huff_table;

typedef struct {
    const uint8_t *data;
    size_t len;
    size_t pos;
    uint64_t bitbuf;  /* 64-bit to prevent overflow when refilling */
    int bitcount;
    int eof;  /* Set when we've padded past end of data */
} jpeg_bitstream;

typedef struct {
    int width, height;
    int num_components;
    int restart_interval;
    int is_progressive;

    /* Component info */
    struct {
        int id;
        int h_samp, v_samp;  /* Sampling factors */
        int qt_idx;          /* Quantization table index */
        int dc_idx, ac_idx;  /* Huffman table indices */
        int blocks_x, blocks_y;  /* Number of 8x8 blocks */
        int16_t *coefs;      /* Coefficient buffer for progressive */
    } comp[4];

    int max_h_samp, max_v_samp;

    /* Quantization tables (up to 4) */
    uint16_t qt[4][64];
    unsigned qt_defined;     /* cleffa: bit t set once a DQT defined qt[t] */
    uint16_t comp_qt[4][64]; /* cleffa: each component's table, latched at its first scan */
    unsigned qt_latched;     /* cleffa: bit c set once comp_qt[c] is latched */

    /* Huffman tables (DC: 0-1, AC: 2-3) */
    jpeg_huff_table huff[4];
    unsigned huff_defined;   /* cleffa: bit t set once a DHT defined huff[t] */
    int8_t coef_bits[4][64]; /* cleffa: progressive precision per coefficient, -1 = never coded */
    int scans_seen;          /* cleffa: SOS segments processed */
    int sof_seen;            /* cleffa: frame headers met by the decoding pass */
    int seq_buffered;        /* cleffa: a sequential frame whose components are split across scans */
    int saw_jfif, saw_adobe, adobe_transform;   /* cleffa: markers before the first SOS */

    /* DC prediction for each component */
    int dc_pred[4];

    /* Bitstream */
    jpeg_bitstream bs;

    /* MCU dimensions */
    int mcu_width, mcu_height;
    int mcus_x, mcus_y;

    /* Progressive scan parameters */
    int ss, se;    /* Spectral selection: start and end coefficient */
    int ah, al;    /* Successive approximation: high and low bit */
    int eobrun;    /* End of block run counter */
} jpeg_decoder;

/* Zigzag order for 8x8 block */
static const uint8_t jpeg_zigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

/* Read next byte, handling FF00 stuffing */
static int jpeg_read_byte(jpeg_bitstream *bs) {
    if (bs->pos >= bs->len) return -1;
    uint8_t b = bs->data[bs->pos++];
    if (b == 0xFF) {
        /* Skip any fill bytes (extra 0xFF) */
        while (bs->pos < bs->len && bs->data[bs->pos] == 0xFF) {
            bs->pos++;
        }
        if (bs->pos >= bs->len) return -1;

        uint8_t next = bs->data[bs->pos];
        if (next == 0x00) {
            /* Stuffed 0x00 means literal 0xFF data byte */
            bs->pos++;
            return 0xFF;
        }

        /* Marker found (including restart markers).
         *
         * IMPORTANT: Don't consume the marker here. If we skip restart markers
         * inside the bitreader, the 8-bit Huffman fast path (peek_bits(8))
         * can read past a restart boundary and accidentally consume bytes from
         * the next interval. Instead, signal the marker and let the scan-level
         * logic consume restart markers at the correct MCU boundary.
         */
        bs->pos--;  /* Back up so bs->data[bs->pos] points at 0xFF */
        if (next >= JPEG_RST0 && next <= JPEG_RST0 + 7) return -2; /* Restart marker */
        return -1;  /* Other marker (end of scan data) */
    }
    return b;
}

/* Get n bits from bitstream */
static int jpeg_get_bits(jpeg_bitstream *bs, int n) {
    while (bs->bitcount < n) {
        int b = jpeg_read_byte(bs);
        if (b == -2) {
            /* Restart marker encountered: treat as fill bits without setting EOF. */
            b = 0xFF;
        } else if (b < 0) {
            /* At EOF - pad with 1s (JPEG convention for fill bits) */
            b = 0xFF;
            bs->eof = 1;
        }
        bs->bitbuf = (bs->bitbuf << 8) | b;
        bs->bitcount += 8;
    }
    bs->bitcount -= n;
    return (bs->bitbuf >> bs->bitcount) & ((1 << n) - 1);
}

/* Peek at n bits without consuming */
static int jpeg_peek_bits(jpeg_bitstream *bs, int n) {
    while (bs->bitcount < n) {
        int b = jpeg_read_byte(bs);
        if (b == -2) {
            /* Restart marker encountered: treat as fill bits without setting EOF. */
            b = 0xFF;
        } else if (b < 0) {
            /* At EOF - pad with 1s (JPEG convention for fill bits) */
            b = 0xFF;
            bs->eof = 1;
        }
        bs->bitbuf = (bs->bitbuf << 8) | b;
        bs->bitcount += 8;
    }
    return (bs->bitbuf >> (bs->bitcount - n)) & ((1 << n) - 1);
}

/* Skip n bits */
static void jpeg_skip_bits(jpeg_bitstream *bs, int n) {
    bs->bitcount -= n;
}

/* Build Huffman table */
static int jpeg_build_huffman(jpeg_huff_table *h, const uint8_t *bits, const uint8_t *values) {
    memcpy(h->bits, bits, 16);
    memset(h->lookup, 0, sizeof(h->lookup));

    int total = 0;
    for (int i = 0; i < 16; i++) {
        total += bits[i];
    }
    if (total > 256) return 0;
    memcpy(h->values, values, total);

    /* Build maxcode and valptr tables */
    int code = 0;
    int idx = 0;
    for (int len = 1; len <= 16; len++) {
        if (bits[len - 1] == 0) {
            h->maxcode[len] = -1;
            h->valptr[len] = 0;
        } else {
            h->valptr[len] = idx;
            for (int i = 0; i < bits[len - 1]; i++) {
                /* Build fast lookup for codes up to 8 bits */
                if (len <= 8) {
                    int pad = 8 - len;
                    int base_code = code << pad;
                    /* Validate lookup index is in bounds */
                    if (base_code + (1 << pad) > 256) return 0;
                    for (int p = 0; p < (1 << pad); p++) {
                        h->lookup[base_code + p] = (len << 8) | values[idx + i];
                    }
                }
                code++;
            }
            h->maxcode[len] = code - 1;
            idx += bits[len - 1];
        }
        code <<= 1;
    }

    return 1;
}

/* Decode one Huffman symbol */
static int jpeg_decode_huffman(jpeg_bitstream *bs, jpeg_huff_table *h) {
    /* Try fast lookup first (8 bits) */
    int peek = jpeg_peek_bits(bs, 8);
    if (peek < 0) return -1;

    int lookup = h->lookup[peek];
    if (lookup != 0) {
        int len = lookup >> 8;
        jpeg_skip_bits(bs, len);
        return lookup & 0xFF;
    }

    /* Slow path for longer codes */
    int code = jpeg_get_bits(bs, 8);
    if (code < 0) return -1;

    for (int len = 9; len <= 16; len++) {
        int bit = jpeg_get_bits(bs, 1);
        if (bit < 0) return -1;
        code = (code << 1) | bit;

        if (h->maxcode[len] >= 0 && code <= h->maxcode[len]) {
            int idx = h->valptr[len] + code - (h->maxcode[len] - h->bits[len - 1] + 1);
            return h->values[idx];
        }
    }

    return -1;  /* Invalid code */
}

/* Extend sign bit */
static int jpeg_extend(int v, int bits) {
    if (bits == 0) return 0;
    int vt = 1 << (bits - 1);
    if (v < vt) {
        v = v - (1 << bits) + 1;
    }
    return v;
}

/* ========================================================================
 * Inverse DCT
 * ======================================================================== */

/* cleffa: jpeg_idct reports a block whose dequantized coefficients leave 16 bits or whose output
 * leaves [-512, 511] before the level shift, and the file is refused. This decoder clamps; libjpeg
 * clamps only inside that range and its builds disagree outside it: measured with libjpeg-turbo
 * 3.2, the C path (JSIMD_FORCENONE=1) wraps an output of 512 to black, while the NEON path that
 * Pillow uses on this Mac clamps to about +-1024 and then wraps differently. An AC coefficient of
 * category 15 decoded 128 levels from Pillow (review #3, Codex on 3f3eb03). Encoder output stays
 * well inside: the largest over 3,692 files from cjpeg, jpegtran, sips, ffmpeg and Pillow,
 * photographs among them at quality 1 to 100, was 397 (docs/vision.md). */
#define JPEG_IDCT_OUT(v) (out_of_range |= ((v) < -512 || (v) > 511), (uint8_t)JPEG_CLAMP((v) + 128))

/* Fast integer IDCT using AAN algorithm (Arai, Agui, Nakajima 1988) */
static int jpeg_idct(int *block, uint8_t *out, int stride) {
    int out_of_range = 0;
    for (int i = 0; i < 64; i++) if (block[i] < -32768 || block[i] > 32767) out_of_range = 1;
    /* cleffa: 64-bit like libjpeg's JLONG here; a dequantized coefficient may be near 2^31 */
    int64_t tmp0, tmp1, tmp2, tmp3;
    int64_t tmp10, tmp11, tmp12, tmp13;
    int64_t z1, z2, z3, z4, z5;
    int *blkptr;
    uint8_t *outptr;
    int64_t workspace[64];

    /* Constants for IDCT */
    #define FIX_0_298 2446
    #define FIX_0_390 3196
    #define FIX_0_541 4433
    #define FIX_0_765 6270
    #define FIX_0_899 7373
    #define FIX_1_175 9633
    #define FIX_1_501 12299
    #define FIX_1_847 15137
    #define FIX_1_961 16069
    #define FIX_2_053 16819
    #define FIX_2_562 20995
    #define FIX_3_072 25172

    /* Pass 1: process columns */
    blkptr = block;
    int64_t *wsptr = workspace;
    for (int col = 0; col < 8; col++) {
        /* Check for all-zero AC terms */
        if (blkptr[8] == 0 && blkptr[16] == 0 && blkptr[24] == 0 &&
            blkptr[32] == 0 && blkptr[40] == 0 && blkptr[48] == 0 && blkptr[56] == 0) {
            int64_t dc = (int64_t)blkptr[0] * 4;
            wsptr[0] = wsptr[8] = wsptr[16] = wsptr[24] =
            wsptr[32] = wsptr[40] = wsptr[48] = wsptr[56] = dc;
            blkptr++;
            wsptr++;
            continue;
        }

        z2 = blkptr[16];
        z3 = blkptr[48];
        z1 = (z2 + z3) * FIX_0_541;
        tmp2 = z1 + z3 * (-FIX_1_847);
        tmp3 = z1 + z2 * FIX_0_765;

        z2 = blkptr[0];
        z3 = blkptr[32];
        tmp0 = (z2 + z3) * 8192;
        tmp1 = (z2 - z3) * 8192;

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        tmp0 = blkptr[56];
        tmp1 = blkptr[40];
        tmp2 = blkptr[24];
        tmp3 = blkptr[8];

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        z4 = tmp1 + tmp3;
        z5 = (z3 + z4) * FIX_1_175;

        tmp0 = tmp0 * FIX_0_298;
        tmp1 = tmp1 * FIX_2_053;
        tmp2 = tmp2 * FIX_3_072;
        tmp3 = tmp3 * FIX_1_501;
        z1 = z1 * (-FIX_0_899);
        z2 = z2 * (-FIX_2_562);
        z3 = z3 * (-FIX_1_961);
        z4 = z4 * (-FIX_0_390);

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        wsptr[0]  = (tmp10 + tmp3 + (1 << 10)) >> 11;
        wsptr[56] = (tmp10 - tmp3 + (1 << 10)) >> 11;
        wsptr[8]  = (tmp11 + tmp2 + (1 << 10)) >> 11;
        wsptr[48] = (tmp11 - tmp2 + (1 << 10)) >> 11;
        wsptr[16] = (tmp12 + tmp1 + (1 << 10)) >> 11;
        wsptr[40] = (tmp12 - tmp1 + (1 << 10)) >> 11;
        wsptr[24] = (tmp13 + tmp0 + (1 << 10)) >> 11;
        wsptr[32] = (tmp13 - tmp0 + (1 << 10)) >> 11;

        blkptr++;
        wsptr++;
    }

    /* Pass 2: process rows */
    wsptr = workspace;
    outptr = out;
    for (int row = 0; row < 8; row++) {
        z2 = wsptr[2];
        z3 = wsptr[6];
        z1 = (z2 + z3) * FIX_0_541;
        tmp2 = z1 + z3 * (-FIX_1_847);
        tmp3 = z1 + z2 * FIX_0_765;

        tmp0 = (wsptr[0] + wsptr[4]) * 8192;
        tmp1 = (wsptr[0] - wsptr[4]) * 8192;

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        tmp0 = wsptr[7];
        tmp1 = wsptr[5];
        tmp2 = wsptr[3];
        tmp3 = wsptr[1];

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        z4 = tmp1 + tmp3;
        z5 = (z3 + z4) * FIX_1_175;

        tmp0 = tmp0 * FIX_0_298;
        tmp1 = tmp1 * FIX_2_053;
        tmp2 = tmp2 * FIX_3_072;
        tmp3 = tmp3 * FIX_1_501;
        z1 = z1 * (-FIX_0_899);
        z2 = z2 * (-FIX_2_562);
        z3 = z3 * (-FIX_1_961);
        z4 = z4 * (-FIX_0_390);

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        outptr[0] = JPEG_IDCT_OUT((tmp10 + tmp3 + (1 << 17)) >> 18);
        outptr[7] = JPEG_IDCT_OUT((tmp10 - tmp3 + (1 << 17)) >> 18);
        outptr[1] = JPEG_IDCT_OUT((tmp11 + tmp2 + (1 << 17)) >> 18);
        outptr[6] = JPEG_IDCT_OUT((tmp11 - tmp2 + (1 << 17)) >> 18);
        outptr[2] = JPEG_IDCT_OUT((tmp12 + tmp1 + (1 << 17)) >> 18);
        outptr[5] = JPEG_IDCT_OUT((tmp12 - tmp1 + (1 << 17)) >> 18);
        outptr[3] = JPEG_IDCT_OUT((tmp13 + tmp0 + (1 << 17)) >> 18);
        outptr[4] = JPEG_IDCT_OUT((tmp13 - tmp0 + (1 << 17)) >> 18);

        wsptr += 8;
        outptr += stride;
    }

    #undef FIX_0_298
    #undef FIX_0_390
    #undef FIX_0_541
    #undef FIX_0_765
    #undef FIX_0_899
    #undef FIX_1_175
    #undef FIX_1_501
    #undef FIX_1_847
    #undef FIX_1_961
    #undef FIX_2_053
    #undef FIX_2_562
    #undef FIX_3_072
    return out_of_range;
}

/* ========================================================================
 * Baseline Decoding
 * ======================================================================== */

/* Decode one 8x8 block */
static int jpeg_decode_block(jpeg_decoder *dec, int comp_idx, int *block) {
    int dc_idx = dec->comp[comp_idx].dc_idx;
    int ac_idx = dec->comp[comp_idx].ac_idx;
    jpeg_huff_table *dc_huff = &dec->huff[dc_idx];
    jpeg_huff_table *ac_huff = &dec->huff[ac_idx + 2];
    uint16_t *qt = dec->comp_qt[comp_idx];

    memset(block, 0, 64 * sizeof(int));

    /* Decode DC coefficient */
    int dc_len = jpeg_decode_huffman(&dec->bs, dc_huff);
    if (dc_len < 0) return -1;

    int dc_val = 0;
    if (dc_len > 0) {
        dc_val = jpeg_get_bits(&dec->bs, dc_len);
        if (dc_val < 0) return -1;
        dc_val = jpeg_extend(dc_val, dc_len);
    }

    /* cleffa: unsigned wrap, then a 16-bit coefficient as libjpeg's JCOEF, so the product
     * with a quantizer of at most 65535 fits int (tests/test_jpeg_ub.c) */
    dec->dc_pred[comp_idx] = (int)((unsigned)dec->dc_pred[comp_idx] + (unsigned)dc_val);
    block[0] = (int16_t)dec->dc_pred[comp_idx] * qt[0];

    /* Decode AC coefficients */
    int k = 1;
    while (k < 64) {
        int rs = jpeg_decode_huffman(&dec->bs, ac_huff);
        if (rs < 0) return -1;

        int run = rs >> 4;
        int size = rs & 0x0F;

        if (size == 0) {
            if (run == 15) {
                k += 16;  /* ZRL: skip 16 zeros */
            } else {
                break;  /* EOB */
            }
        } else {
            k += run;
            /* cleffa: past 63, libjpeg's decode_mcu writes through the padding of its natural
             * order, at 63, and the block ends; this was refused. */

            int ac_val = jpeg_get_bits(&dec->bs, size);
            if (ac_val < 0) return -1;
            ac_val = jpeg_extend(ac_val, size);
            if (k > 63) k = 63;

            block[jpeg_zigzag[k]] = ac_val * qt[k];
            k++;
        }
    }

    return 0;
}

/* cleffa: a sequential block stored as quantized coefficients, for a frame whose components are
 * split across scans. libjpeg decodes such a frame into zero-initialized coefficient arrays and
 * dequantizes at output (jdcoefct.c consume_data); it writes the DC coefficient and each decoded
 * AC coefficient and nothing else, so a component scanned twice keeps the first scan's values
 * where the second codes zeros, and a component never scanned stays zero. Both decode without a
 * warning, as here. Otherwise the arithmetic is jpeg_decode_block's. */
static int jpeg_seq_decode_coefs(jpeg_decoder *dec, int comp_idx, int16_t *coef) {
    jpeg_huff_table *dc_huff = &dec->huff[dec->comp[comp_idx].dc_idx];
    jpeg_huff_table *ac_huff = &dec->huff[dec->comp[comp_idx].ac_idx + 2];

    int dc_len = jpeg_decode_huffman(&dec->bs, dc_huff);
    if (dc_len < 0) return -1;
    int dc_val = 0;
    if (dc_len > 0) {
        dc_val = jpeg_get_bits(&dec->bs, dc_len);
        if (dc_val < 0) return -1;
        dc_val = jpeg_extend(dc_val, dc_len);
    }
    dec->dc_pred[comp_idx] = (int)((unsigned)dec->dc_pred[comp_idx] + (unsigned)dc_val);
    coef[0] = (int16_t)dec->dc_pred[comp_idx];

    for (int k = 1; k < 64; ) {
        int rs = jpeg_decode_huffman(&dec->bs, ac_huff);
        if (rs < 0) return -1;
        int run = rs >> 4, size = rs & 0x0F;
        if (size == 0) {
            if (run != 15) break;   /* EOB */
            k += 16;                /* ZRL */
            continue;
        }
        k += run;
        int ac_val = jpeg_get_bits(&dec->bs, size);
        if (ac_val < 0) return -1;
        ac_val = jpeg_extend(ac_val, size);
        if (k > 63) k = 63;         /* libjpeg's padded natural order, as in jpeg_decode_block */
        coef[jpeg_zigzag[k]] = (int16_t)ac_val;
        k++;
    }
    return 0;
}

/* cleffa: libjpeg's colour-space choice for three components (jdapimin.c default_decompress_parms,
 * which Pillow leaves in charge): a JFIF marker means YCbCr; otherwise an Adobe marker decides
 * (transform 0 = RGB, anything else YCbCr); otherwise component ids 'R','G','B' mean RGB. Converting
 * RGB-coded planes as YCbCr gave wrong colours where Pillow copies them. */
static int jpeg_rgb_coded(const jpeg_decoder *dec) {
    if (dec->num_components != 3 || dec->saw_jfif) return 0;
    if (dec->saw_adobe) return dec->adobe_transform == 0;
    return dec->comp[0].id == 82 && dec->comp[1].id == 71 && dec->comp[2].id == 66;
}

/* YCbCr to RGB conversion */
static void jpeg_ycbcr_to_rgb(uint8_t y, uint8_t cb, uint8_t cr, uint8_t *rgb) {
    int yy = y;
    int cbb = cb - 128;
    int crr = cr - 128;

    /* Round the full-precision color transform, avoiding a half-level bias
     * from truncating each channel's contribution. */
    int r = yy + ((crr * 91881 + 32768) >> 16);
    int g = yy + ((-cbb * 22554 - crr * 46802 + 32768) >> 16);
    int b = yy + ((cbb * 116130 + 32768) >> 16);

    rgb[0] = JPEG_CLAMP(r);
    rgb[1] = JPEG_CLAMP(g);
    rgb[2] = JPEG_CLAMP(b);
}

/* DS4: interpolate common subsampled chroma at pixel centers, matching the
 * triangle filter used by libjpeg/Pillow. Clamp to the real component extent,
 * not its padded MCU storage. Unusual sampling ratios retain replication. */
static uint8_t jpeg_sample_chroma(const jpeg_decoder *dec, int component,
                                  const uint8_t *plane, int stride, int x, int y) {
    const int hs = dec->comp[component].h_samp;
    const int vs = dec->comp[component].v_samp;
    const int sx = x * hs / dec->max_h_samp;
    const int sy = y * vs / dec->max_v_samp;
    const int width = (dec->width * hs + dec->max_h_samp - 1) / dec->max_h_samp;
    const int height = (dec->height * vs + dec->max_v_samp - 1) / dec->max_v_samp;
    /* cleffa: vertical-only 2:1 (4:4:0), libjpeg's h1v2_fancy_upsample, the transpose of the
     * h2v1 case below: three parts the nearest row, one part the next nearest, bias 1 above and
     * 2 below. Replicating rows here put 4:4:0 files up to 69 levels from libjpeg
     * (tests/test_image_diff.py). */
    if (dec->max_h_samp == hs && dec->max_v_samp == 2 * vs) {
        int ny = sy + ((y & 1) ? 1 : -1);
        if (ny < 0) ny = 0;
        if (ny >= height) ny = height - 1;
        return (uint8_t)((3 * plane[sy * stride + sx] + plane[ny * stride + sx] + ((y & 1) ? 2 : 1)) >> 2);
    }
    if (dec->max_h_samp != 2 * hs || width <= 2 ||
        (dec->max_v_samp != vs && dec->max_v_samp != 2 * vs)) {
        return plane[sy * stride + sx];
    }
    int nx = sx + ((x & 1) ? 1 : -1);
    if (nx < 0) nx = 0;
    if (nx >= width) nx = width - 1;
    if (dec->max_v_samp == vs) {
        return (uint8_t)((3 * plane[sy * stride + sx] + plane[sy * stride + nx] +
                          ((x & 1) ? 2 : 1)) >> 2);
    }
    int ny = sy + ((y & 1) ? 1 : -1);
    if (ny < 0) ny = 0;
    if (ny >= height) ny = height - 1;
    int near = 3 * plane[sy * stride + sx] + plane[ny * stride + sx];
    int far = 3 * plane[sy * stride + nx] + plane[ny * stride + nx];
    return (uint8_t)((3 * near + far + ((x & 1) ? 7 : 8)) >> 4);
}

/* Consume a restart marker at the current bitstream position.
 * Caller must ensure the bitstream is byte-aligned (bitcount == 0). */
static int jpeg_skip_restart_marker(jpeg_decoder *dec) {
    jpeg_bitstream *bs = &dec->bs;

    /* Skip any fill bytes (extra 0xFF) */
    while (bs->pos + 1 < bs->len &&
           bs->data[bs->pos] == 0xFF &&
           bs->data[bs->pos + 1] == 0xFF) {
        bs->pos++;
    }

    if (bs->pos + 1 >= bs->len) return -1;
    if (bs->data[bs->pos] != 0xFF) return -1;
    uint8_t marker = bs->data[bs->pos + 1];
    if (marker < JPEG_RST0 || marker > JPEG_RST0 + 7) return -1;

    bs->pos += 2;
    return 0;
}

/* Decode scan data for baseline JPEG */
static int jpeg_decode_scan(jpeg_decoder *dec, uint8_t *y_data, uint8_t *cb_data, uint8_t *cr_data) {
    int block[64];
    uint8_t block_out[64];

    int restart_count = dec->restart_interval;

    /* Reset DC predictors */
    for (int i = 0; i < 4; i++) {
        dec->dc_pred[i] = 0;
    }

    for (int mcu_y = 0; mcu_y < dec->mcus_y; mcu_y++) {
        for (int mcu_x = 0; mcu_x < dec->mcus_x; mcu_x++) {
            /* Handle restart interval */
            if (dec->restart_interval > 0 && restart_count == 0) {
                /* Align to byte boundary */
                dec->bs.bitcount = 0;
                dec->bs.bitbuf = 0;

                /* Reset DC predictors */
                for (int i = 0; i < 4; i++) {
                    dec->dc_pred[i] = 0;
                }
                if (jpeg_skip_restart_marker(dec) < 0) return -1;
                restart_count = dec->restart_interval;
            }

            /* Decode Y blocks */
            for (int v = 0; v < dec->comp[0].v_samp; v++) {
                for (int h = 0; h < dec->comp[0].h_samp; h++) {
                    if (jpeg_decode_block(dec, 0, block) < 0) return -1;
                    if (jpeg_idct(block, block_out, 8)) return -1;

                    /* Copy to Y plane */
                    int bx = mcu_x * dec->comp[0].h_samp * 8 + h * 8;
                    int by = mcu_y * dec->comp[0].v_samp * 8 + v * 8;
                    int y_stride = dec->mcus_x * dec->comp[0].h_samp * 8;

                    for (int row = 0; row < 8; row++) {
                        int dst_y = by + row;
                        if (dst_y < dec->height) {
                            for (int col = 0; col < 8; col++) {
                                int dst_x = bx + col;
                                if (dst_x < dec->width) {
                                    y_data[dst_y * y_stride + dst_x] = block_out[row * 8 + col];
                                }
                            }
                        }
                    }
                }
            }

            /* Decode Cb block(s) */
            if (dec->num_components >= 3) {
                for (int v = 0; v < dec->comp[1].v_samp; v++) {
                    for (int h = 0; h < dec->comp[1].h_samp; h++) {
                        if (jpeg_decode_block(dec, 1, block) < 0) return -1;
                        if (jpeg_idct(block, block_out, 8)) return -1;

                        int bx = mcu_x * dec->comp[1].h_samp * 8 + h * 8;
                        int by = mcu_y * dec->comp[1].v_samp * 8 + v * 8;
                        int cb_stride = dec->mcus_x * dec->comp[1].h_samp * 8;

                        for (int row = 0; row < 8; row++) {
                            for (int col = 0; col < 8; col++) {
                                int dst_y = by + row;
                                int dst_x = bx + col;
                                if (dst_y < (dec->mcus_y * dec->comp[1].v_samp * 8) &&
                                    dst_x < cb_stride) {
                                    cb_data[dst_y * cb_stride + dst_x] = block_out[row * 8 + col];
                                }
                            }
                        }
                    }
                }

                /* Decode Cr block(s) */
                for (int v = 0; v < dec->comp[2].v_samp; v++) {
                    for (int h = 0; h < dec->comp[2].h_samp; h++) {
                        if (jpeg_decode_block(dec, 2, block) < 0) return -1;
                        if (jpeg_idct(block, block_out, 8)) return -1;

                        int bx = mcu_x * dec->comp[2].h_samp * 8 + h * 8;
                        int by = mcu_y * dec->comp[2].v_samp * 8 + v * 8;
                        int cr_stride = dec->mcus_x * dec->comp[2].h_samp * 8;

                        for (int row = 0; row < 8; row++) {
                            for (int col = 0; col < 8; col++) {
                                int dst_y = by + row;
                                int dst_x = bx + col;
                                if (dst_y < (dec->mcus_y * dec->comp[2].v_samp * 8) &&
                                    dst_x < cr_stride) {
                                    cr_data[dst_y * cr_stride + dst_x] = block_out[row * 8 + col];
                                }
                            }
                        }
                    }
                }
            }

            if (dec->restart_interval > 0) {
                restart_count--;
            }
        }
    }

    return 0;
}

/* ========================================================================
 * Progressive Decoding
 * ======================================================================== */

/* Decode DC coefficient for progressive first scan (Ah == 0) */
static int jpeg_prog_decode_dc_first(jpeg_decoder *dec, int comp_idx, int16_t *coef) {
    jpeg_huff_table *dc_huff = &dec->huff[dec->comp[comp_idx].dc_idx];

    int dc_len = jpeg_decode_huffman(&dec->bs, dc_huff);
    if (dc_len < 0) return -1;

    int dc_val = 0;
    if (dc_len > 0) {
        dc_val = jpeg_get_bits(&dec->bs, dc_len);
        if (dc_val < 0) return -1;
        dc_val = jpeg_extend(dc_val, dc_len);
    }

    /* cleffa: as baseline, and the shift by Al in unsigned (libjpeg's LEFT_SHIFT) */
    dec->dc_pred[comp_idx] = (int)((unsigned)dec->dc_pred[comp_idx] + (unsigned)dc_val);
    coef[0] = (int16_t)(int)((unsigned)dec->dc_pred[comp_idx] << dec->al);

    return 0;
}

/* Decode DC coefficient refinement for progressive (Ah != 0) */
static int jpeg_prog_decode_dc_refine(jpeg_decoder *dec, int16_t *coef) {
    int bit = jpeg_get_bits(&dec->bs, 1);
    if (bit < 0) return -1;

    if (bit) {
        coef[0] |= (1 << dec->al);
    }

    return 0;
}

/* Decode AC coefficients for progressive first scan (Ah == 0) */
static int jpeg_prog_decode_ac_first(jpeg_decoder *dec, int comp_idx, int16_t *coef) {
    jpeg_huff_table *ac_huff = &dec->huff[dec->comp[comp_idx].ac_idx + 2];

    if (dec->eobrun > 0) {
        dec->eobrun--;
        return 0;
    }

    int k = dec->ss;
    while (k <= dec->se && k < 64) {
        int rs = jpeg_decode_huffman(&dec->bs, ac_huff);
        if (rs < 0) {
            /* At EOF, treat as implicit EOB for remaining blocks */
            if (dec->bs.eof) return 0;
            return -1;
        }

        int run = rs >> 4;
        int size = rs & 0x0F;

        if (size == 0) {
            if (run == 15) {
                k += 16;  /* ZRL: skip 16 zeros */
            } else {
                /* EOBn: end of block run */
                dec->eobrun = (1 << run);
                if (run > 0) {
                    int extra = jpeg_get_bits(&dec->bs, run);
                    if (extra < 0) {
                        if (dec->bs.eof) break;
                        return -1;
                    }
                    dec->eobrun += extra;
                }
                dec->eobrun--;
                break;
            }
        } else {
            k += run;
            /* cleffa: a run that overshoots Se still writes its coefficient, at zigzag position
             * k, or 63 once k passes the table, and ends the band: libjpeg's
             * decode_mcu_AC_first indexes its padded natural order without a check or a
             * warning. The decoder dropped the coefficient at the end of the data and refused
             * the file elsewhere (tests/fuzz_jpeg_diff.c; one Cr coefficient of 6 was lost). */

            int ac_val = jpeg_get_bits(&dec->bs, size);
            if (ac_val < 0) {
                if (dec->bs.eof) return 0;
                return -1;
            }
            ac_val = jpeg_extend(ac_val, size);
            if (k > 63) k = 63;

            coef[jpeg_zigzag[k]] = (int16_t)(ac_val * (1 << dec->al));
            k++;
        }
    }

    return 0;
}

/* Decode AC coefficient refinement for progressive (Ah != 0) */
static int jpeg_prog_decode_ac_refine(jpeg_decoder *dec, int comp_idx, int16_t *coef) {
    jpeg_huff_table *ac_huff = &dec->huff[dec->comp[comp_idx].ac_idx + 2];
    int p1 = 1 << dec->al;    /* Bit to add for positive refinement */
    int m1 = -p1;             /* Bit to add for negative refinement */

    int k = dec->ss;

    if (dec->eobrun == 0) {
        while (k <= dec->se && k < 64) {
            int rs = jpeg_decode_huffman(&dec->bs, ac_huff);
            if (rs < 0) {
                /* At EOF, treat as implicit EOB for remaining blocks */
                if (dec->bs.eof) break;
                return -1;
            }

            int run = rs >> 4;
            int size = rs & 0x0F;

            int new_val = 0;
            if (size == 0) {
                if (run != 15) {
                    /* EOBn */
                    dec->eobrun = (1 << run);
                    if (run > 0) {
                        int extra = jpeg_get_bits(&dec->bs, run);
                        if (extra < 0) {
                            if (dec->bs.eof) break;
                            return -1;
                        }
                        dec->eobrun += extra;
                    }
                    break;
                }
                /* ZRL: skip 15 zeros, then write a zero (16 zeros total).
                 * Do not keep refining trailing non-zeros before the next
                 * Huffman symbol; those correction bits come after it. */
            } else if (size != 1) {
                if (dec->bs.eof) break;
                return -1;  /* Invalid: size must be 1 for refinement */
            } else {
                int bit = jpeg_get_bits(&dec->bs, 1);
                if (bit < 0) {
                    if (dec->bs.eof) break;
                    return -1;
                }
                new_val = bit ? p1 : m1;
            }

            /* Skip 'run' zeros, refining non-zeros along the way, then
             * place new_val (0 for ZRL) at the next zero and stop. */
            int placed = 0;
            while (k <= dec->se && k < 64) {
                int zk = jpeg_zigzag[k];
                if (coef[zk] != 0) {
                    /* Refine existing non-zero coefficient */
                    int bit = jpeg_get_bits(&dec->bs, 1);
                    if (bit < 0) {
                        if (dec->bs.eof) goto refine_done;
                        return -1;
                    }
                    if (bit && (coef[zk] & p1) == 0) {
                        if (coef[zk] > 0) {
                            coef[zk] += p1;
                        } else {
                            coef[zk] += m1;
                        }
                    }
                    k++;
                } else if (run > 0) {
                    run--;
                    k++;
                } else {
                    coef[zk] = (int16_t)new_val;
                    placed = 1;
                    k++;
                    break;
                }
            }
            /* cleffa: a run that overshoots Se still writes the new coefficient, at position
             * Se + 1 (or 63), as libjpeg's decode_mcu_AC_refine does through its padded natural
             * order, without a warning; skipping it left one coefficient different. */
            if (new_val && !placed) coef[jpeg_zigzag[k < 64 ? k : 63]] = (int16_t)new_val;

            /* Huffman lookahead may have reached the next marker while valid
             * bits remain buffered. The new coefficient was already written
             * in the skip loop; keep decoding this block. */
        }
    }

refine_done:
    /* Process remaining coefficients if in EOBRUN */
    if (dec->eobrun > 0) {
        while (k <= dec->se && k < 64) {
            int zk = jpeg_zigzag[k];
            if (coef[zk] != 0) {
                int bit = jpeg_get_bits(&dec->bs, 1);
                if (bit < 0) {
                    if (dec->bs.eof) break;
                    return -1;
                }
                if (bit && (coef[zk] & p1) == 0) {
                    if (coef[zk] > 0) {
                        coef[zk] += p1;
                    } else {
                        coef[zk] += m1;
                    }
                }
            }
            k++;
        }
        dec->eobrun--;
    }

    return 0;
}

/* Decode one progressive scan */
static int jpeg_decode_progressive_scan(jpeg_decoder *dec, int *scan_comps, int num_scan_comps) {
    int restart_count = dec->restart_interval;

    /* Reset state */
    for (int i = 0; i < 4; i++) {
        dec->dc_pred[i] = 0;
    }
    dec->eobrun = 0;

    /* Interleaved DC scans walk MCUs; a scan of one component, DC (cleffa) or AC, walks that
     * component's own blocks in raster order (A.2.2/A.2.3). A single-component DC scan walked
     * as MCUs read a subsampled luma's blocks out of order (silently wrong pixels) or past the
     * end of the scan (refused), where libjpeg decodes the file. */
    if (dec->ss == 0 && num_scan_comps > 1) {
        /* DC scan - interleaved MCUs */
        for (int mcu_y = 0; mcu_y < dec->mcus_y; mcu_y++) {
            for (int mcu_x = 0; mcu_x < dec->mcus_x; mcu_x++) {
                /* Handle restart interval */
                if (dec->restart_interval > 0 && restart_count == 0) {
                    dec->bs.bitcount = 0;
                    dec->bs.bitbuf = 0;
                    for (int i = 0; i < 4; i++) {
                        dec->dc_pred[i] = 0;
                    }
                    dec->eobrun = 0;
                    if (jpeg_skip_restart_marker(dec) < 0) return -1;
                    restart_count = dec->restart_interval;
                }

                /* Process each component in this MCU */
                for (int ci = 0; ci < num_scan_comps; ci++) {
                    int comp_idx = scan_comps[ci];
                    int h_samp = dec->comp[comp_idx].h_samp;
                    int v_samp = dec->comp[comp_idx].v_samp;
                    int blocks_x = dec->comp[comp_idx].blocks_x;

                    for (int v = 0; v < v_samp; v++) {
                        for (int h = 0; h < h_samp; h++) {
                            int bx = mcu_x * h_samp + h;
                            int by = mcu_y * v_samp + v;
                            int16_t *coef = dec->comp[comp_idx].coefs + (by * blocks_x + bx) * 64;

                            if (!dec->is_progressive) {
                                if (jpeg_seq_decode_coefs(dec, comp_idx, coef) < 0) return -1;
                            } else if (dec->ah == 0) {
                                if (jpeg_prog_decode_dc_first(dec, comp_idx, coef) < 0) return -1;
                            } else {
                                if (jpeg_prog_decode_dc_refine(dec, coef) < 0) return -1;
                            }
                        }
                    }
                }

                if (dec->restart_interval > 0) {
                    restart_count--;
                }
            }
        }
    } else {
        /* One component (AC, or DC in its own scan) - non-interleaved.
         * Per JPEG spec section A.2.3, non-interleaved scans process data units
         * in raster order. For components with sampling factors > 1, the number
         * of data units is based on the COMPONENT dimensions (scaled from image
         * dimensions), not MCU-aligned dimensions.
         *
         * Component dimensions: ceil(image_dim * samp_factor / max_samp_factor)
         * Blocks per row: ceil(comp_width / 8)
         *
         * However, coefficients are STORED using MCU-aligned indexing to match
         * how DC scans store coefficients. So we iterate over image-based block
         * positions but map to MCU-aligned storage positions. */
        int comp_idx = scan_comps[0];
        int h_samp = dec->comp[comp_idx].h_samp;
        int v_samp = dec->comp[comp_idx].v_samp;

        /* Image-based block dimensions (for bitstream iteration) */
        int comp_width = (dec->width * h_samp + dec->max_h_samp - 1) / dec->max_h_samp;
        int comp_height = (dec->height * v_samp + dec->max_v_samp - 1) / dec->max_v_samp;
        int scan_blocks_x = (comp_width + 7) / 8;
        int scan_blocks_y = (comp_height + 7) / 8;

        /* MCU-aligned block dimensions (for storage indexing) */
        int store_blocks_x = dec->comp[comp_idx].blocks_x;

        for (int by = 0; by < scan_blocks_y; by++) {
            for (int bx = 0; bx < scan_blocks_x; bx++) {
                /* Handle restart interval */
                if (dec->restart_interval > 0 && restart_count == 0) {
                    dec->bs.bitcount = 0;
                    dec->bs.bitbuf = 0;
                    dec->eobrun = 0;
                    for (int i = 0; i < 4; i++) {
                        dec->dc_pred[i] = 0;
                    }
                    if (jpeg_skip_restart_marker(dec) < 0) return -1;
                    restart_count = dec->restart_interval;
                }

                /* Map image-based block position to MCU-aligned storage index */
                int16_t *coef = dec->comp[comp_idx].coefs + (by * store_blocks_x + bx) * 64;

                if (!dec->is_progressive) {
                    if (jpeg_seq_decode_coefs(dec, comp_idx, coef) < 0) return -1;
                } else if (dec->ss == 0) {
                    if (dec->ah == 0) {
                        if (jpeg_prog_decode_dc_first(dec, comp_idx, coef) < 0) return -1;
                    } else {
                        if (jpeg_prog_decode_dc_refine(dec, coef) < 0) return -1;
                    }
                } else if (dec->ah == 0) {
                    if (jpeg_prog_decode_ac_first(dec, comp_idx, coef) < 0) return -1;
                } else {
                    if (jpeg_prog_decode_ac_refine(dec, comp_idx, coef) < 0) return -1;
                }

                if (dec->restart_interval > 0) {
                    restart_count--;
                }
            }
        }
    }

    return 0;
}

/* Convert progressive coefficients to pixels */
static int jpeg_prog_finish(jpeg_decoder *dec, uint8_t **planes, int *strides) {
    int block[64];
    uint8_t block_out[64];

    for (int comp_idx = 0; comp_idx < dec->num_components; comp_idx++) {
        int blocks_x = dec->comp[comp_idx].blocks_x;
        int blocks_y = dec->comp[comp_idx].blocks_y;
        uint16_t *qt = dec->comp_qt[comp_idx];
        int stride = strides[comp_idx];

        for (int by = 0; by < blocks_y; by++) {
            for (int bx = 0; bx < blocks_x; bx++) {
                int16_t *coef = dec->comp[comp_idx].coefs + (by * blocks_x + bx) * 64;

                /* Dequantize - coefficients are stored at zigzag positions,
                 * we need to put them at raster positions for IDCT */
                memset(block, 0, sizeof(block));
                for (int i = 0; i < 64; i++) {
                    block[jpeg_zigzag[i]] = coef[jpeg_zigzag[i]] * qt[i];
                }

                /* IDCT */
                if (jpeg_idct(block, block_out, 8)) return -1;

                /* Copy to output plane */
                int px = bx * 8;
                int py = by * 8;
                for (int row = 0; row < 8; row++) {
                    for (int col = 0; col < 8; col++) {
                        int x = px + col;
                        int y = py + row;
                        if (x < stride && y < blocks_y * 8) {
                            planes[comp_idx][y * stride + x] = block_out[row * 8 + col];
                        }
                    }
                }
            }
        }
    }
    return 0;
}

/* ========================================================================
 * JPEG Loading
 * ======================================================================== */

jpeg_image *jpeg_load_mem(const uint8_t *file_data, size_t file_size) {
    /* Check SOI marker */
    if (!file_data || file_size < 2 || file_size > JPEG_MAX_INPUT_BYTES ||
        file_data[0] != 0xFF || file_data[1] != JPEG_SOI) {
        return NULL;
    }

    jpeg_decoder dec;
    memset(&dec, 0, sizeof(dec));
    memset(dec.coef_bits, -1, sizeof(dec.coef_bits));

    size_t pos = 2;
    jpeg_image *img = NULL;

    /* Parse markers - first pass to get frame info */
    while (pos < file_size - 1) {
        if (file_data[pos] != 0xFF) {
            pos++;
            continue;
        }

        uint8_t marker = file_data[pos + 1];
        pos += 2;

        /* cleffa: 0xFF fill bytes may precede any marker; step over one at a time, as libjpeg's
         * next_marker does. Skipping "FF FF" as a pair swallowed the real marker's FF in
         * "FF FF D9", so the decoder read past an EOI that libjpeg stops at. */
        if (marker == 0xFF) { pos -= 1; continue; }
        if (marker == 0x00) continue;
        if (marker == JPEG_EOI) break;

        /* Markers without length */
        if (marker >= JPEG_RST0 && marker <= JPEG_RST0 + 7) continue;
        if (marker == JPEG_SOI) continue;

        /* Read segment length */
        if (pos + 2 > file_size) break;
        uint16_t seg_len = (file_data[pos] << 8) | file_data[pos + 1];
        if (pos + seg_len > file_size) break;

        /* cleffa: SOF1 is sequential like SOF0; libjpeg writes it when a quantizer exceeds 255
         * (cjpeg -quality below about 25). 12-bit precision and Huffman tables 2-3, which it also
         * allows, are still refused by the checks below and in DHT/SOS. */
        if (marker == JPEG_SOF0 || marker == JPEG_SOF1 || marker == JPEG_SOF2) {
            /* Start of frame */
            dec.is_progressive = (marker == JPEG_SOF2);

            if (seg_len < 8) goto fail;

            dec.height = (file_data[pos + 3] << 8) | file_data[pos + 4];
            dec.width = (file_data[pos + 5] << 8) | file_data[pos + 6];
            dec.num_components = file_data[pos + 7];

            if (file_data[pos + 2] != 8) goto fail;
            if (dec.width <= 0 || dec.height <= 0 ||
                dec.width > JPEG_MAX_DIMENSION ||
                dec.height > JPEG_MAX_DIMENSION ||
                (size_t)dec.width * (size_t)dec.height > JPEG_MAX_PIXELS) {
                goto fail;
            }
            if (dec.num_components != 1 && dec.num_components != 3) goto fail;
            if (seg_len < 8 + dec.num_components * 3) goto fail;

            dec.max_h_samp = dec.max_v_samp = 1;

            for (int i = 0; i < dec.num_components; i++) {
                int offset = pos + 8 + i * 3;
                dec.comp[i].id = file_data[offset];
                dec.comp[i].h_samp = file_data[offset + 1] >> 4;
                dec.comp[i].v_samp = file_data[offset + 1] & 0x0F;
                dec.comp[i].qt_idx = file_data[offset + 2];

                /* Validate sampling factors and table index */
                if (dec.comp[i].h_samp == 0 || dec.comp[i].h_samp > 4) goto fail;
                if (dec.comp[i].v_samp == 0 || dec.comp[i].v_samp > 4) goto fail;
                if (dec.comp[i].qt_idx > 3) goto fail;

                if (dec.comp[i].h_samp > dec.max_h_samp) dec.max_h_samp = dec.comp[i].h_samp;
                if (dec.comp[i].v_samp > dec.max_v_samp) dec.max_v_samp = dec.comp[i].v_samp;
            }
            /* cleffa: a one-component frame's sampling factors have no effect: its scans are
             * non-interleaved, one block per MCU in raster order over the component's real size,
             * as libjpeg decodes them. Walked as h x v MCUs, a one-component frame declaring 2x1
             * (found by tests/fuzz_jpeg_diff.c) put its blocks in the wrong places. */
            if (dec.num_components == 1) dec.comp[0].h_samp = dec.comp[0].v_samp = dec.max_h_samp = dec.max_v_samp = 1;

            /* Calculate MCU dimensions */
            /* cleffa: the luma plane is sized from component 0's own factors but read at
             * image resolution (the conversion loops and jpeg_sample_chroma), so component
             * 0 must be the most sampled one; grayscale trivially is. */
            if (dec.comp[0].h_samp != dec.max_h_samp || dec.comp[0].v_samp != dec.max_v_samp) goto fail;
            dec.mcu_width = dec.max_h_samp * 8;
            dec.mcu_height = dec.max_v_samp * 8;
            dec.mcus_x = (dec.width + dec.mcu_width - 1) / dec.mcu_width;
            dec.mcus_y = (dec.height + dec.mcu_height - 1) / dec.mcu_height;

            /* Calculate block dimensions for each component */
            for (int i = 0; i < dec.num_components; i++) {
                dec.comp[i].blocks_x = dec.mcus_x * dec.comp[i].h_samp;
                dec.comp[i].blocks_y = dec.mcus_y * dec.comp[i].v_samp;
            }

            /* For progressive, allocate coefficient buffers */
            if (dec.is_progressive) {
                for (int i = 0; i < dec.num_components; i++) {
                    size_t num_blocks = (size_t)dec.comp[i].blocks_x * dec.comp[i].blocks_y;
                    dec.comp[i].coefs = (int16_t *)calloc(num_blocks * 64, sizeof(int16_t));
                    if (!dec.comp[i].coefs) goto fail;
                }
            }
            break;  /* Found SOF, stop first pass */
        }

        pos += seg_len;
    }

    if (dec.width == 0 || dec.height == 0) goto fail;

    /* Second pass - process DHT, DQT, DRI, and SOS markers */
    pos = 2;
    while (pos < file_size - 1) {
        if (file_data[pos] != 0xFF) {
            pos++;
            continue;
        }

        uint8_t marker = file_data[pos + 1];
        pos += 2;

        /* cleffa: 0xFF fill bytes may precede any marker; step over one at a time, as libjpeg's
         * next_marker does. Skipping "FF FF" as a pair swallowed the real marker's FF in
         * "FF FF D9", so the decoder read past an EOI that libjpeg stops at. */
        if (marker == 0xFF) { pos -= 1; continue; }
        if (marker == 0x00) continue;
        if (marker == JPEG_EOI) break;

        if (marker >= JPEG_RST0 && marker <= JPEG_RST0 + 7) continue;
        if (marker == JPEG_SOI) continue;

        if (pos + 2 > file_size) break;
        uint16_t seg_len = (file_data[pos] << 8) | file_data[pos + 1];
        if (pos + seg_len > file_size) break;

        /* cleffa: libjpeg reads markers in order, so a scan before any frame header fails
         * (JERR_SOS_NO_SOF) and so does a second frame header of any SOFn type
         * (JERR_SOF_DUPLICATE). The first pass looks ahead for the frame, and this pass skipped
         * frame headers, so a scan was decoded with the geometry of a later frame and a file with
         * two frames was decoded (Codex on a82292d). */
        if (marker >= 0xC0 && marker <= 0xCF && marker != JPEG_DHT && marker != 0xC8 && marker != 0xCC) {
            if (dec.sof_seen++) goto fail;
        }

        if (marker == JPEG_DHT) {
            /* Define Huffman table */
            size_t off = pos + 2;
            size_t end = pos + seg_len;

            while (off < end) {
                uint8_t th = file_data[off++];
                int tc = th >> 4;
                int idx = th & 0x0F;

                if (tc > 1 || idx > 1) goto fail;
                int table_idx = tc * 2 + idx;

                if (off + 16 > end) goto fail;
                uint8_t bits[16];
                memcpy(bits, file_data + off, 16);
                off += 16;

                int total = 0;
                for (int i = 0; i < 16; i++) total += bits[i];
                if (off + total > end) goto fail;

                /* cleffa: a DC symbol is the bit count of the difference that follows;
                 * libjpeg refuses DC tables with symbols above 15. */
                if (tc == 0) for (int i = 0; i < total; i++) if (file_data[off + i] > 15) goto fail;
                if (!jpeg_build_huffman(&dec.huff[table_idx], bits, file_data + off)) {
                    goto fail;
                }
                dec.huff_defined |= 1u << table_idx;

                off += total;
            }

        } else if (marker == JPEG_DQT) {
            /* Define quantization table */
            size_t off = pos + 2;
            size_t end = pos + seg_len;

            while (off < end) {
                uint8_t pq_tq = file_data[off++];
                int precision = pq_tq >> 4;
                int tq = pq_tq & 0x0F;

                if (tq > 3) goto fail;

                if (precision == 0) {
                    if (off + 64 > end) goto fail;
                    for (int i = 0; i < 64; i++) {
                        dec.qt[tq][i] = file_data[off + i];
                    }
                    off += 64;
                } else {
                    if (off + 128 > end) goto fail;
                    for (int i = 0; i < 64; i++) {
                        dec.qt[tq][i] = (file_data[off + i * 2] << 8) | file_data[off + i * 2 + 1];
                    }
                    off += 128;
                }
                dec.qt_defined |= 1u << tq;
            }

        } else if ((marker == 0xE0 || marker == 0xEE) && !dec.scans_seen) {
            /* cleffa: libjpeg's examine_app0 / examine_app14. The colour space is decided at
             * the first SOS (jdapimin.c default_decompress_parms), so later markers do not count. */
            const uint8_t *d = file_data + pos + 2;
            const size_t n = seg_len >= 2 ? (size_t)seg_len - 2 : 0;
            if (marker == 0xE0 && n >= 14 && !memcmp(d, "JFIF\0", 5)) dec.saw_jfif = 1;
            if (marker == 0xEE && n >= 12 && !memcmp(d, "Adobe", 5)) { dec.saw_adobe = 1; dec.adobe_transform = d[11]; }

        } else if (marker == JPEG_DRI) {
            if (seg_len < 4) goto fail;
            dec.restart_interval = (file_data[pos + 2] << 8) | file_data[pos + 3];

        } else if (marker == JPEG_SOS) {
            /* Start of scan */
            if (!dec.sof_seen) goto fail;
            dec.scans_seen++;
            if (seg_len < 6) goto fail;

            int ns = file_data[pos + 2];
            if (ns < 1 || ns > 4) goto fail;
            if (seg_len < 6 + ns * 2) goto fail;

            int scan_comps[4];
            for (int i = 0; i < ns; i++) {
                int cs = file_data[pos + 3 + i * 2];
                int td_ta = file_data[pos + 4 + i * 2];

                /* Find component index */
                int comp_idx = -1;
                for (int j = 0; j < dec.num_components; j++) {
                    if (dec.comp[j].id == cs) {
                        comp_idx = j;
                        dec.comp[j].dc_idx = td_ta >> 4;
                        dec.comp[j].ac_idx = td_ta & 0x0F;
                        /* Validate Huffman table indices (DC: 0-1, AC: 0-1) */
                        if (dec.comp[j].dc_idx > 1) goto fail;
                        if (dec.comp[j].ac_idx > 1) goto fail;
                        break;
                    }
                }
                if (comp_idx < 0) goto fail;
                /* cleffa: each component at most once per scan, so the table check below covers
                 * every component the scan decodes. */
                for (int k = 0; k < i; k++) if (scan_comps[k] == comp_idx) goto fail;
                scan_comps[i] = comp_idx;
            }

            /* Parse Ss, Se, Ah, Al for progressive */
            size_t sos_offset = pos + 3 + ns * 2;
            dec.ss = file_data[sos_offset];
            dec.se = file_data[sos_offset + 1];
            dec.ah = file_data[sos_offset + 2] >> 4;
            dec.al = file_data[sos_offset + 2] & 0x0F;
            /* cleffa: the spectral selection indexes the 64-entry zigzag table and the
             * block's coefficients (jpeg_prog_decode_ac_first/_refine); bound it as
             * libjpeg's jdinput.c does before any scan data is read. */
            if (dec.is_progressive) {
                if (dec.ss > 63 || dec.se > 63 || dec.ss > dec.se || dec.ah > 13 || dec.al > 13) goto fail;
                if (dec.ss == 0 && dec.se != 0) goto fail;      /* a DC scan carries DC only */
                if (dec.ss != 0 && ns != 1) goto fail;          /* AC scans are one component */
            } else if (dec.ss != 0 || dec.se != 63 || dec.ah != 0 || dec.al != 0) {
                goto fail;
            }
            /* cleffa: every table this scan decodes with must have been defined, as libjpeg
             * requires (JERR_NO_HUFF_TABLE). Baseline scans use both; progressive DC first
             * scans the DC table, AC scans the AC table, DC refinement neither. */
            for (int i = 0; i < ns; i++) {
                const int ci = scan_comps[i];
                const bool dc = !dec.is_progressive || (dec.ss == 0 && dec.ah == 0);
                const bool ac = !dec.is_progressive || dec.ss != 0;
                if (dc && !(dec.huff_defined & (1u << dec.comp[ci].dc_idx))) goto fail;
                if (ac && !(dec.huff_defined & (1u << (dec.comp[ci].ac_idx + 2)))) goto fail;
            }
            /* cleffa: libjpeg copies a component's quantization table at the component's first
             * scan and fails if it is not defined by then (jdinput.c latch_quant_tables,
             * JERR_NO_QUANT_TABLE); a later DQT for the same slot does not change it. A file
             * with no table decoded from the zero-filled slot as flat gray, and a progressive
             * file was dequantized with whatever table the last DQT left (review #3, Codex on
             * 3f3eb03). */
            for (int i = 0; i < ns; i++) {
                const int ci = scan_comps[i];
                if (dec.qt_latched & (1u << ci)) continue;
                if (!(dec.qt_defined & (1u << dec.comp[ci].qt_idx))) goto fail;
                memcpy(dec.comp_qt[ci], dec.qt[dec.comp[ci].qt_idx], sizeof(dec.comp_qt[ci]));
                dec.qt_latched |= 1u << ci;
            }

            /* Setup bitstream for scan data */
            size_t scan_data_start = pos + seg_len;
            dec.bs.data = file_data + scan_data_start;
            dec.bs.pos = 0;
            dec.bs.bitbuf = 0;
            dec.bs.bitcount = 0;
            dec.bs.eof = 0;

            /* Find end of scan data (next marker). cleffa: fill FFs before a restart marker stay
             * in the scan; after any other run of FFs the scan ends at the run's first FF (a marker
             * with fill bytes). "FF FF 00", fill before a stuffed zero, is not standard JPEG and is
             * refused: libjpeg's slow path reads it as one FF data byte, but libjpeg-turbo first
             * tries a fast Huffman path whenever enough input is buffered (jdhuff.c decode_mcu),
             * and the measured result then depends on how the input is buffered: one 3.2 build
             * decoded the same file differently from memory and from 4 KiB stdio reads, without
             * a warning (docs/vision.md). The reference has no stable answer for it. Stopping at
             * "FF FF" had also ended such scans early (tests/fuzz_jpeg_diff.c, tests/test_image.py). */
            size_t scan_end = scan_data_start;
            while (scan_end < file_size - 1) {
                if (file_data[scan_end] == 0xFF) {
                    size_t q = scan_end + 1;
                    while (q < file_size && file_data[q] == 0xFF) q++;
                    if (q >= file_size) break;
                    if (file_data[q] == 0x00 && q > scan_end + 1) goto fail;
                    if (file_data[q] == 0x00 || (file_data[q] >= JPEG_RST0 && file_data[q] <= JPEG_RST0 + 7)) {
                        scan_end = q + 1;
                        continue;
                    }
                    break;
                }
                scan_end++;
            }
            dec.bs.len = scan_end - scan_data_start;

            /* cleffa: a sequential frame may split its components across scans (one each, or Y
             * then Cb and Cr). libjpeg buffers coefficients for the whole frame when its first
             * scan lacks a component (jdinput.c has_multiple_scans), and so does this decoder from
             * then on, through the progressive path's walk and finish; such files were refused,
             * though Pillow decodes them (Codex on a82292d). */
            if (!dec.is_progressive && !dec.seq_buffered && ns != dec.num_components) {
                for (int i = 0; i < dec.num_components; i++) {
                    size_t num_blocks = (size_t)dec.comp[i].blocks_x * dec.comp[i].blocks_y;
                    dec.comp[i].coefs = (int16_t *)calloc(num_blocks * 64, sizeof(int16_t));
                    if (!dec.comp[i].coefs) goto fail;
                }
                dec.seq_buffered = 1;
            }
            if (dec.is_progressive || dec.seq_buffered) {
                /* Decode progressive scan, or one sequential scan into the coefficient buffers */
                if (dec.is_progressive)
                    for (int i = 0; i < ns; i++)
                        for (int k = dec.ss; k <= dec.se; k++) dec.coef_bits[scan_comps[i]][k] = (int8_t)dec.al;
                if (jpeg_decode_progressive_scan(&dec, scan_comps, ns) < 0) goto fail;
                pos = scan_end;
                continue;  /* Continue to next scan */
            } else {
                /* Baseline: single scan with all components */

                /* Allocate component planes */
                int y_stride = dec.mcus_x * dec.comp[0].h_samp * 8;
                int y_height = dec.mcus_y * dec.comp[0].v_samp * 8;
                uint8_t *y_data = (uint8_t *)calloc(y_stride * y_height, 1);
                if (!y_data) goto fail;

                uint8_t *cb_data = NULL;
                uint8_t *cr_data = NULL;
                int cb_stride = 0, cr_stride = 0;

                if (dec.num_components >= 3) {
                    cb_stride = dec.mcus_x * dec.comp[1].h_samp * 8;
                    int cb_height = dec.mcus_y * dec.comp[1].v_samp * 8;
                    cb_data = (uint8_t *)calloc(cb_stride * cb_height, 1);

                    cr_stride = dec.mcus_x * dec.comp[2].h_samp * 8;
                    int cr_height = dec.mcus_y * dec.comp[2].v_samp * 8;
                    cr_data = (uint8_t *)calloc(cr_stride * cr_height, 1);

                    if (!cb_data || !cr_data) {
                        free(y_data);
                        free(cb_data);
                        free(cr_data);
                        goto fail;
                    }
                }

                /* Decode baseline scan */
                if (jpeg_decode_scan(&dec, y_data, cb_data, cr_data) < 0) {
                    free(y_data);
                    free(cb_data);
                    free(cr_data);
                    goto fail;
                }

                /* Create output image */
                int out_channels = (dec.num_components == 1) ? 1 : 3;
                img = jpeg_create(dec.width, dec.height, out_channels);
                if (!img) {
                    free(y_data);
                    free(cb_data);
                    free(cr_data);
                    goto fail;
                }

                /* Convert to RGB */
                if (dec.num_components == 1) {
                    for (int y = 0; y < dec.height; y++) {
                        for (int x = 0; x < dec.width; x++) {
                            img->data[y * dec.width + x] = y_data[y * y_stride + x];
                        }
                    }
                } else {
                    const int rgb_coded = jpeg_rgb_coded(&dec);
                    for (int y = 0; y < dec.height; y++) {
                        for (int x = 0; x < dec.width; x++) {
                            uint8_t yy = y_data[y * y_stride + x];
                            uint8_t cb = jpeg_sample_chroma(&dec, 1, cb_data, cb_stride, x, y);
                            uint8_t cr = jpeg_sample_chroma(&dec, 2, cr_data, cr_stride, x, y);

                            uint8_t *o = img->data + (y * dec.width + x) * 3;
                            if (rgb_coded) { o[0] = yy; o[1] = cb; o[2] = cr; }
                            else jpeg_ycbcr_to_rgb(yy, cb, cr, o);
                        }
                    }
                }

                free(y_data);
                free(cb_data);
                free(cr_data);
                return img;
            }
        }

        pos += seg_len;
    }

    /* For progressive, or a sequential frame split across scans, finish decoding after all scans */
    if (dec.is_progressive || dec.seq_buffered) {
        /* cleffa: a progressive frame that reached EOI before any scan is not an image; libjpeg
         * fails with JERR_SOF_NO_SOS ("missing SOS marker"), and finishing it here returned a
         * blank picture. */
        if (!dec.scans_seen) goto fail;
        /* Allocate pixel planes */
        uint8_t *planes[4] = {NULL, NULL, NULL, NULL};
        int strides[4] = {0, 0, 0, 0};

        for (int i = 0; i < dec.num_components; i++) {
            strides[i] = dec.comp[i].blocks_x * 8;
            planes[i] = (uint8_t *)calloc(strides[i] * dec.comp[i].blocks_y * 8, 1);
            if (!planes[i]) {
                for (int j = 0; j < i; j++) free(planes[j]);
                goto fail;
            }
        }

        /* Convert coefficients to pixels */
        /* cleffa: libjpeg smooths an incomplete progressive image (jdcoefct.c smoothing_ok):
         * when every component's DC is known and its DC and first nine AC quantizers are
         * nonzero, and some component's first nine AC coefficients are not at full precision.
         * That reconstruction is not implemented here, so such files are refused rather than
         * decoded differently from the reference. */
        {
            int smooth = 0, possible = dec.is_progressive;   /* libjpeg smooths progressive frames only */
            for (int ci = 0; ci < dec.num_components && possible; ci++) {
                const uint16_t *q = dec.comp_qt[ci];
                if (dec.coef_bits[ci][0] < 0) possible = 0;
                for (int k = 0; k < 10 && possible; k++) if (q[k] == 0) possible = 0;
                for (int k = 1; k < 10; k++) if (dec.coef_bits[ci][k] != 0) smooth = 1;
            }
            if (possible && smooth) {
                for (int i = 0; i < dec.num_components; i++) free(planes[i]);
                goto fail;
            }
        }
        if (jpeg_prog_finish(&dec, planes, strides) < 0) {
            for (int i = 0; i < dec.num_components; i++) free(planes[i]);
            goto fail;
        }

        /* Create output image */
        int out_channels = (dec.num_components == 1) ? 1 : 3;
        img = jpeg_create(dec.width, dec.height, out_channels);
        if (!img) {
            for (int i = 0; i < dec.num_components; i++) free(planes[i]);
            goto fail;
        }

        /* Convert to RGB */
        if (dec.num_components == 1) {
            for (int y = 0; y < dec.height; y++) {
                for (int x = 0; x < dec.width; x++) {
                    img->data[y * dec.width + x] = planes[0][y * strides[0] + x];
                }
            }
        } else {
            const int rgb_coded = jpeg_rgb_coded(&dec);
            for (int y = 0; y < dec.height; y++) {
                for (int x = 0; x < dec.width; x++) {
                    uint8_t yy = planes[0][y * strides[0] + x];
                    uint8_t cb = jpeg_sample_chroma(&dec, 1, planes[1], strides[1], x, y);
                    uint8_t cr = jpeg_sample_chroma(&dec, 2, planes[2], strides[2], x, y);

                    uint8_t *o = img->data + (y * dec.width + x) * 3;
                    if (rgb_coded) { o[0] = yy; o[1] = cb; o[2] = cr; }
                    else jpeg_ycbcr_to_rgb(yy, cb, cr, o);
                }
            }
        }

        for (int i = 0; i < dec.num_components; i++) free(planes[i]);
        for (int i = 0; i < dec.num_components; i++) free(dec.comp[i].coefs);
        return img;
    }

fail:
    for (int i = 0; i < 4; i++) {
        if (dec.comp[i].coefs) free(dec.comp[i].coefs);
    }
    return NULL;
}

jpeg_image *jpeg_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long end = ftell(f);
    if (end < 0 || (unsigned long)end > JPEG_MAX_INPUT_BYTES ||
        fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    size_t file_size = (size_t)end;

    uint8_t *file_data = (uint8_t *)malloc(file_size);
    if (!file_data) {
        fclose(f);
        return NULL;
    }

    if (fread(file_data, 1, file_size, f) != file_size) {
        free(file_data);
        fclose(f);
        return NULL;
    }
    fclose(f);

    jpeg_image *img = jpeg_load_mem(file_data, file_size);
    free(file_data);
    return img;
}

/* Clean up internal macros */
#undef JPEG_CLAMP
#undef JPEG_SOI
#undef JPEG_EOI
#undef JPEG_SOF0
#undef JPEG_SOF1
#undef JPEG_SOF2
#undef JPEG_DHT
#undef JPEG_DQT
#undef JPEG_DRI
#undef JPEG_SOS
#undef JPEG_RST0

#endif /* JPEG_IMPLEMENTATION */
