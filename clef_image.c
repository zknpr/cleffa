#include "clef_image.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <zlib.h>

/* PNG dimensions bound the allocation before inflate. Require exactly one complete,
 * checksummed stream of the expected size; never grow output from compressed input. */
static uint8_t *clef_png_inflate(const uint8_t *data, size_t len, size_t expected_len) {
    uint8_t *out = malloc(expected_len);
    if (!out) return NULL;
    uLongf written = expected_len;
    uLong consumed = len;
    if (uncompress2(out, &written, data, &consumed) != Z_OK || written != expected_len || consumed != len) {
        free(out);
        return NULL;
    }
    return out;
}

static uint32_t clef_png_crc(uint32_t crc, const uint8_t *data, size_t len) {
    /* iris carries an uncomplemented CRC between chunks; zlib complements it at each call. */
    return len ? (uint32_t)crc32_z(crc ^ 0xffffffffu, data, len) ^ 0xffffffffu : crc;
}

/* Single-header decoders imported from ds4's copy of iris (third_party/iris, MIT), which
 * carries ds4's decode limits and its JPEG fixes for libjpeg agreement (centered chroma
 * upsampling, rounded YCbCr). Limits are set here so the decoders refuse oversized input
 * before allocating. */
#define PNG_MAX_INPUT_BYTES CLEF_IMAGE_MAX_ENCODED
#define PNG_MAX_DIMENSION CLEF_IMAGE_MAX_DIMENSION
#define PNG_MAX_PIXELS CLEF_IMAGE_MAX_PIXELS
#define PNG_INFLATE clef_png_inflate
#define PNG_UPDATE_CRC clef_png_crc
#define JPEG_MAX_INPUT_BYTES CLEF_IMAGE_MAX_ENCODED
#define JPEG_MAX_DIMENSION CLEF_IMAGE_MAX_DIMENSION
#define JPEG_MAX_PIXELS CLEF_IMAGE_MAX_PIXELS
#define PNG_IMPLEMENTATION
#define JPEG_IMPLEMENTATION
#include "third_party/iris/jpeg.h"
#include "third_party/iris/png.h"

static bool fail(char *err, size_t errlen, const char *msg) {
    snprintf(err, errlen, "%s", msg);
    return false;
}

/* ---- base64 ---- */

/* 255 marks non-alphabet bytes, including padding handled only in the final quartet. */
static const uint8_t base64_value[256] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  62, 255, 255, 255,  63,
     52,  53,  54,  55,  56,  57,  58,  59,  60,  61, 255, 255, 255, 255, 255, 255,
    255,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
     15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 255, 255, 255, 255, 255,
    255,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
     41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

bool clef_base64_decode(const char *s, size_t len, uint8_t **out, size_t *out_len, char *err, size_t errlen) {
    *out = NULL;
    *out_len = 0;
    /* data URL: "data:image/png;base64,<payload>" (the hosted API matches the prefix case-
     * insensitively); the media type is not trusted, the payload's signature decides the format */
    if (len >= 5 && !strncasecmp(s, "data:", 5)) {
        const char *comma = memchr(s, ',', len);
        if (!comma) return fail(err, errlen, "image: data URL without a comma");
        const size_t hdr = (size_t)(comma - s);
        if (hdr < 7 || memcmp(comma - 7, ";base64", 7)) return fail(err, errlen, "image: data URL is not base64");
        s = comma + 1;
        len -= hdr + 1;
    }
    if (len % 4) return fail(err, errlen, "image: base64 length is not a multiple of 4 (padding required)");
    if (len / 4 * 3 > CLEF_IMAGE_MAX_ENCODED) return fail(err, errlen, "image: encoded image exceeds 64 MiB");
    uint8_t *buf = malloc(len / 4 * 3 + 1);
    if (!buf) return fail(err, errlen, "out of memory (image)");
    size_t n = 0;
    size_t i = 0;
    for (; i + 4 < len; i += 4) {
        const uint32_t a = base64_value[(uint8_t)s[i]], b = base64_value[(uint8_t)s[i + 1]];
        const uint32_t c = base64_value[(uint8_t)s[i + 2]], d = base64_value[(uint8_t)s[i + 3]];
        if ((a | b | c | d) & 0x80) { free(buf); return fail(err, errlen, "image: invalid base64 character"); }
        const uint32_t v = a << 18 | b << 12 | c << 6 | d;
        buf[n++] = (uint8_t)(v >> 16);
        buf[n++] = (uint8_t)(v >> 8);
        buf[n++] = (uint8_t)v;
    }
    if (i < len) {
        uint32_t v = 0;
        int pad = 0;
        for (int k = 0; k < 4; k++) {
            const char c = s[i + k];
            int d;
            if (c >= 'A' && c <= 'Z') d = c - 'A';
            else if (c >= 'a' && c <= 'z') d = c - 'a' + 26;
            else if (c >= '0' && c <= '9') d = c - '0' + 52;
            else if (c == '+') d = 62;
            else if (c == '/') d = 63;
            else if (c == '=' && i + 4 == len && k >= 2) { d = 0; pad++; }
            else { free(buf); return fail(err, errlen, "image: invalid base64 character"); }
            if (pad && c != '=') { free(buf); return fail(err, errlen, "image: invalid base64 padding"); }
            v = v << 6 | (uint32_t)d;
        }
        buf[n++] = (uint8_t)(v >> 16);
        if (pad < 2) buf[n++] = (uint8_t)(v >> 8);
        if (pad < 1) buf[n++] = (uint8_t)v;
    }
    *out = buf;
    *out_len = n;
    return true;
}

/* ---- decoding ---- */

/* PIL convert("RGB"): gray replicated, alpha dropped (no compositing), palette already applied. */
static bool to_rgb(clef_rgb *out, const uint8_t *px, int w, int h, int channels, char *err, size_t errlen) {
    const size_t n = (size_t)w * (size_t)h;
    uint8_t *rgb = malloc(n * 3);
    if (!rgb) return fail(err, errlen, "out of memory (image)");
    for (size_t i = 0; i < n; i++) {
        const uint8_t *s = px + i * (size_t)channels;
        if (channels < 3) rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = s[0];
        else { rgb[i * 3] = s[0]; rgb[i * 3 + 1] = s[1]; rgb[i * 3 + 2] = s[2]; }
    }
    out->width = w;
    out->height = h;
    out->rgb = rgb;
    return true;
}

bool clef_image_decode(const uint8_t *data, size_t len, clef_rgb *out, char *err, size_t errlen) {
    memset(out, 0, sizeof(*out));
    if (!data || len == 0) return fail(err, errlen, "image: empty");
    if (len > CLEF_IMAGE_MAX_ENCODED) return fail(err, errlen, "image: encoded image exceeds 64 MiB");
    if (len >= 8 && !memcmp(data, "\x89PNG\r\n\x1a\n", 8)) {
        png_image *img = png_load_mem(data, len);
        if (!img) return fail(err, errlen, "image: invalid or unsupported PNG (8-bit, non-interlaced; palette 1-8 bit)");
        if (img->channels == 3) {
            *out = (clef_rgb){ .width = img->width, .height = img->height, .rgb = img->data };
            img->data = NULL;   /* transfer the malloc-owned RGB bytes, without another copy */
            png_free(img);
            return true;
        }
        const bool ok = to_rgb(out, img->data, img->width, img->height, img->channels, err, errlen);
        png_free(img);
        return ok;
    }
    if (len >= 2 && data[0] == 0xff && data[1] == 0xd8) {
        jpeg_image *img = jpeg_load_mem(data, len);
        if (!img) return fail(err, errlen, "image: invalid or unsupported JPEG (baseline or progressive, grayscale or YCbCr)");
        if (img->channels == 3) {
            *out = (clef_rgb){ .width = img->width, .height = img->height, .rgb = img->data };
            img->data = NULL;
            jpeg_free(img);
            return true;
        }
        const bool ok = to_rgb(out, img->data, img->width, img->height, img->channels, err, errlen);
        jpeg_free(img);
        return ok;
    }
    if (len >= 12 && !memcmp(data, "RIFF", 4) && !memcmp(data + 8, "WEBP", 4))
        return fail(err, errlen, "image: WebP is not supported (PNG or JPEG only)");
    return fail(err, errlen, "image: not a PNG or JPEG");
}

void clef_rgb_free(clef_rgb *img) {
    if (!img) return;
    free(img->rgb);
    memset(img, 0, sizeof(*img));
}

/* ---- smart_resize ---- */

bool clef_smart_resize(int height, int width, int factor, long min_pixels, long max_pixels,
                       int *out_height, int *out_width, char *err, size_t errlen) {
    const double h = height, w = width;
    if (height <= 0 || width <= 0 || factor <= 0) return fail(err, errlen, "image: empty");
    if (fmax(h, w) / fmin(h, w) > 200.0) return fail(err, errlen, "image: absolute aspect ratio must be smaller than 200");
    /* Python round() is round-half-to-even; h / factor is an exact double, so rint matches. */
    double h_bar = rint(h / factor) * factor;
    double w_bar = rint(w / factor) * factor;
    if (h_bar * w_bar > (double)max_pixels) {
        const double beta = sqrt(h * w / (double)max_pixels);
        h_bar = fmax(factor, floor(h / beta / factor) * factor);
        w_bar = fmax(factor, floor(w / beta / factor) * factor);
    } else if (h_bar * w_bar < (double)min_pixels) {
        const double beta = sqrt((double)min_pixels / (h * w));
        h_bar = ceil(h * beta / factor) * factor;
        w_bar = ceil(w * beta / factor) * factor;
    }
    if (h_bar < factor || w_bar < factor || h_bar > CLEF_IMAGE_MAX_DIMENSION || w_bar > CLEF_IMAGE_MAX_DIMENSION)
        return fail(err, errlen, "image: resized dimensions out of range");
    *out_height = (int)h_bar;
    *out_width = (int)w_bar;
    return true;
}

/* ---- PyTorch uint8 antialiased bicubic (aten/src/ATen/native/cpu/UpSampleKernel.cpp) ----
 * Weights are computed in double exactly as HelperInterpBase::_compute_indices_min_size_weights_aa
 * with the Keys cubic (a = -0.5), normalized, then rescaled to int16 with the precision chosen
 * from the largest weight (_compute_index_ranges_int16_weights). Each output sample is an integer
 * dot product plus half an LSB, shifted down and clamped (basic_loop_aa_horizontal<uint8_t>). */

static double keys_cubic(double x) {
    const double a = -0.5;
    x = fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

typedef struct {
    int max_interp;        /* weights per output sample */
    unsigned precision;    /* bits of the int16 weights */
    int64_t *xmin, *xsize; /* [out] first source sample and count */
    int16_t *w;            /* [out][max_interp] */
} aa_plan;

static bool aa_plan_build(aa_plan *p, int in_size, int out_size, char *err, size_t errlen) {
    memset(p, 0, sizeof(*p));
    const double scale = (double)in_size / (double)out_size;   /* area_pixel_compute_scale, align_corners=false */
    const double support = scale >= 1.0 ? 2.0 * scale : 2.0;    /* interp_size 4 */
    const int max_interp = (int)ceil(support) * 2 + 1;
    double *wd = malloc((size_t)out_size * (size_t)max_interp * sizeof(double));
    p->xmin = malloc((size_t)out_size * sizeof(int64_t));
    p->xsize = malloc((size_t)out_size * sizeof(int64_t));
    p->w = malloc((size_t)out_size * (size_t)max_interp * sizeof(int16_t));
    if (!wd || !p->xmin || !p->xsize || !p->w) {
        free(wd); free(p->xmin); free(p->xsize); free(p->w);
        memset(p, 0, sizeof(*p));
        return fail(err, errlen, "out of memory (resize)");
    }
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    double wt_max = 0.0;
    for (int i = 0; i < out_size; i++) {
        const double center = scale * (i + 0.5);
        int64_t xmin = (int64_t)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int64_t xsize = (int64_t)(center + support + 0.5);
        if (xsize > in_size) xsize = in_size;
        xsize -= xmin;
        if (xsize < 0) xsize = 0;
        if (xsize > max_interp) xsize = max_interp;   /* "rare precision cases" clamp */
        double *wt = wd + (size_t)i * max_interp;
        double total = 0.0;
        int64_t j = 0;
        for (; j < xsize; j++) {
            const double v = keys_cubic(((double)(j + xmin) - center + 0.5) * invscale);
            wt[j] = v;
            total += v;
        }
        if (total != 0.0) {
            for (j = 0; j < xsize; j++) {
                wt[j] /= total;
                if (wt[j] > wt_max) wt_max = wt[j];
            }
        }
        for (; j < max_interp; j++) wt[j] = 0.0;
        p->xmin[i] = xmin;
        p->xsize[i] = xsize;
    }
    unsigned precision = 0;
    for (precision = 0; precision < 22; precision++) {
        const int next = (int)(0.5 + wt_max * (double)(1 << (precision + 1)));
        if (next >= (1 << 15)) break;
    }
    for (size_t k = 0; k < (size_t)out_size * (size_t)max_interp; k++) {
        const double v = wd[k] * (double)(1 << precision);
        p->w[k] = (int16_t)(v < 0 ? (int)(-0.5 + v) : (int)(0.5 + v));
    }
    free(wd);
    p->max_interp = max_interp;
    p->precision = precision;
    return true;
}

static void aa_plan_free(aa_plan *p) {
    free(p->xmin); free(p->xsize); free(p->w);
    memset(p, 0, sizeof(*p));
}

static void aa_sample_rgb(const uint8_t *src, size_t stride, const aa_plan *p, int i, uint8_t *dst) {
    const int16_t *w = p->w + (size_t)i * p->max_interp;
    const uint8_t *s = src + (size_t)p->xmin[i] * stride;
    const int round = 1 << (p->precision - 1);
    int r = round, g = round, b = round;
    /* Share weights and addresses across RGB, retaining each channel's accumulation order. */
    for (int64_t j = 0; j < p->xsize[i]; j++) {
        const uint8_t *pixel = s + (size_t)j * stride;
        const int weight = w[j];
        r += (int)pixel[0] * weight;
        g += (int)pixel[1] * weight;
        b += (int)pixel[2] * weight;
    }
    /* Negative bicubic lobes clamp to zero; positive shifts are exact floor division. */
    const unsigned qr = r > 0 ? (unsigned)r >> p->precision : 0;
    const unsigned qg = g > 0 ? (unsigned)g >> p->precision : 0;
    const unsigned qb = b > 0 ? (unsigned)b >> p->precision : 0;
    dst[0] = (uint8_t)(qr > 255 ? 255 : qr);
    dst[1] = (uint8_t)(qg > 255 ? 255 : qg);
    dst[2] = (uint8_t)(qb > 255 ? 255 : qb);
}

bool clef_resize_bicubic_aa(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int dh, char *err, size_t errlen) {
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return fail(err, errlen, "image: empty resize");
    if (sw == dw && sh == dh) {
        memcpy(dst, src, (size_t)sw * (size_t)sh * 3);
        return true;
    }
    /* Width pass into a uint8 temporary, then the height pass, as separable_upsample_generic_Nd_kernel_impl
     * orders them; a pass whose size does not change is skipped, not run as an identity. */
    const uint8_t *cur = src;
    uint8_t *tmp = NULL;
    int cw = sw;
    if (dw != sw) {
        aa_plan p;
        if (!aa_plan_build(&p, sw, dw, err, errlen)) return false;
        uint8_t *out = dh != sh ? (tmp = malloc((size_t)dw * (size_t)sh * 3)) : dst;
        if (!out) { aa_plan_free(&p); return fail(err, errlen, "out of memory (resize)"); }
        for (int y = 0; y < sh; y++)
            for (int x = 0; x < dw; x++)
                aa_sample_rgb(cur + (size_t)y * sw * 3, 3, &p, x, out + ((size_t)y * dw + x) * 3);
        aa_plan_free(&p);
        cur = out;
        cw = dw;
    }
    if (dh != sh) {
        aa_plan p;
        if (!aa_plan_build(&p, sh, dh, err, errlen)) { free(tmp); return false; }
        for (int y = 0; y < dh; y++)
            for (int x = 0; x < cw; x++)
                aa_sample_rgb(cur + (size_t)x * 3, (size_t)cw * 3, &p, y, dst + ((size_t)y * cw + x) * 3);
        aa_plan_free(&p);
    }
    free(tmp);
    return true;
}

/* ---- normalize + patches ---- */

bool clef_image_preprocess(const clef_rgb *img, const clef_image_params *p, clef_image_patches *out, char *err, size_t errlen) {
    memset(out, 0, sizeof(*out));
    const int factor = p->patch * p->merge;
    int rh, rw;
    if (!clef_smart_resize(img->height, img->width, factor, p->min_pixels, p->max_pixels, &rh, &rw, err, errlen)) return false;
    uint8_t *owned = NULL;
    const uint8_t *resized = img->rgb;
    if (rw != img->width || rh != img->height) {
        owned = malloc((size_t)rh * (size_t)rw * 3);
        if (!owned) return fail(err, errlen, "out of memory (image)");
        if (!clef_resize_bicubic_aa(img->rgb, img->width, img->height, owned, rw, rh, err, errlen)) { free(owned); return false; }
        resized = owned;
    }
    const int gh = rh / p->patch, gw = rw / p->patch, m = p->merge, P = p->patch;
    const int n_patch = gh * gw, dim = 3 * p->temporal * P * P;
    float *patches = malloc((size_t)n_patch * (size_t)dim * sizeof(float));
    if (!patches) { free(owned); return fail(err, errlen, "out of memory (image patches)"); }
    /* torchvision normalize with the fused mean/std: (x - 127.5) / 127.5 in f32 */
    const float mean = 0.5f * 255.0f, std = 0.5f * 255.0f;
    float normalized[256];
    for (int i = 0; i < 256; i++) normalized[i] = ((float)i - mean) / std;
    size_t row = 0;
    for (int by = 0; by < gh / m; by++)
        for (int bx = 0; bx < gw / m; bx++)
            for (int my = 0; my < m; my++)
                for (int mx = 0; mx < m; mx++, row++) {
                    const int py = by * m + my, px = bx * m + mx;
                    float *dst = patches + row * (size_t)dim;
                    for (int c = 0; c < 3; c++) {
                        float *plane = dst + (size_t)c * p->temporal * P * P;
                        for (int y = 0; y < P; y++)
                            for (int x = 0; x < P; x++) {
                                const uint8_t v = resized[(((size_t)py * P + y) * rw + (size_t)px * P + x) * 3 + c];
                                plane[y * P + x] = normalized[v];
                            }
                        /* A still image repeats the same spatial plane in every temporal slot. */
                        for (int t = 1; t < p->temporal; t++)
                            memcpy(plane + (size_t)t * P * P, plane, (size_t)P * P * sizeof(float));
                    }
                }
    free(owned);
    out->width = rw;
    out->height = rh;
    out->grid_h = gh;
    out->grid_w = gw;
    out->n_patch = n_patch;
    out->n_tokens = n_patch / (m * m);
    out->patch_dim = dim;
    out->patches = patches;
    return true;
}

void clef_image_patches_free(clef_image_patches *p) {
    if (!p) return;
    free(p->patches);
    memset(p, 0, sizeof(*p));
}

/* torch.linspace(0, side - 1, n) in float32: start + step * i below the midpoint, end - step *
 * (n - 1 - i) above it (RangeFactoriesKernel.cpp's scalar path; the vector path can differ by an
 * ulp, which moves a weight by ~1e-7 and never the interpolation result beyond that). */
static float linspace_f32(float start, float end, int n, int i) {
    const float step = (end - start) / (float)(n - 1);
    return i < n / 2 ? start + step * (float)i : end - step * (float)(n - 1 - i);
}

void clef_image_pos_interp(int grid_h, int grid_w, int side, int merge, int32_t *idx, float *w) {
    size_t row = 0;
    for (int by = 0; by < grid_h / merge; by++)
        for (int bx = 0; bx < grid_w / merge; bx++)
            for (int my = 0; my < merge; my++)
                for (int mx = 0; mx < merge; mx++, row++) {
                    const int y = by * merge + my, x = bx * merge + mx;
                    const float hg = grid_h > 1 ? linspace_f32(0.0f, (float)(side - 1), grid_h, y) : 0.0f;
                    const float wg = grid_w > 1 ? linspace_f32(0.0f, (float)(side - 1), grid_w, x) : 0.0f;
                    const int h0 = (int)hg, w0 = (int)wg;             /* .int(): truncation, values >= 0 */
                    const int h1 = h0 + 1 < side ? h0 + 1 : side - 1;  /* clamp(max=side-1) */
                    const int w1 = w0 + 1 < side ? w0 + 1 : side - 1;
                    const float hf = hg - (float)h0, wf = wg - (float)w0;
                    idx[row * 4 + 0] = h0 * side + w0;
                    idx[row * 4 + 1] = h0 * side + w1;
                    idx[row * 4 + 2] = h1 * side + w0;
                    idx[row * 4 + 3] = h1 * side + w1;
                    w[row * 4 + 0] = (1.0f - hf) * (1.0f - wf);
                    w[row * 4 + 1] = (1.0f - hf) * wf;
                    w[row * 4 + 2] = hf * (1.0f - wf);
                    w[row * 4 + 3] = hf * wf;
                }
}
