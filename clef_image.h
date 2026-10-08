#ifndef CLEF_IMAGE_H
#define CLEF_IMAGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Image input for the vision tower: decoding and the Qwen2VLImageProcessor preprocessing the
 * reference runs, reproduced byte for byte up to the f32 patch values (tests/test_image.py):
 *   base64 -> PNG/JPEG decode -> RGB (PIL convert("RGB"): alpha dropped, palette applied)
 *   -> smart_resize to multiples of patch*merge within [min_pixels, max_pixels]
 *   -> torchvision resize: PyTorch's native uint8 antialiased bicubic CPU kernel (int16 weights,
 *      integer accumulation, width pass then height pass), which is what the oracle's processor
 *      runs on this Mac (the AVX2 and float paths round differently)
 *   -> (x / 255 - 0.5) / 0.5 in f32
 *   -> 16x16 patches in 2x2 merge-window order, each patch holding the frame twice for the two
 *      temporal taps of the Conv3d patch embedding: [n_patch][3 * temporal * patch * patch].
 * Decoded images are untrusted input: encoded size, dimensions and pixel counts are capped. */

#define CLEF_IMAGE_MAX_ENCODED (64u << 20)    /* bytes of one encoded image */
#define CLEF_IMAGE_MAX_DIMENSION 16384
#define CLEF_IMAGE_MAX_PIXELS (64u << 20)     /* decoded pixels of one image */

typedef struct {
    int width, height;
    uint8_t *rgb;          /* [height][width][3] */
} clef_rgb;

typedef struct {
    long min_pixels, max_pixels;   /* smart_resize bounds on the resized pixel count */
    int patch, merge, temporal;    /* vision tower geometry (16, 2, 2) */
} clef_image_params;

typedef struct {
    int width, height;     /* resized size, multiples of patch * merge */
    int grid_h, grid_w;    /* patches per side */
    int n_patch;           /* grid_h * grid_w */
    int n_tokens;          /* n_patch / merge^2: the image's tokens in the backbone */
    int patch_dim;         /* 3 * temporal * patch * patch */
    float *patches;        /* [n_patch][patch_dim], merge-window order */
} clef_image_patches;

/* Standard base64 (RFC 4648, padding required, no whitespace), optionally wrapped as a
 * "data:<type>;base64," URL (prefix matched case-insensitively). *out is malloc'd. */
bool clef_base64_decode(const char *s, size_t len, uint8_t **out, size_t *out_len, char *err, size_t errlen);

/* PNG or JPEG by signature, to 8-bit RGB. */
bool clef_image_decode(const uint8_t *data, size_t len, clef_rgb *out, char *err, size_t errlen);

/* The same with a source-pixel limit below CLEF_IMAGE_MAX_PIXELS (max_pixels <= 0: that cap).
 * The decoders enforce it where they read the header (PNG IHDR, every JPEG SOF), before any
 * pixel buffer exists: decoding allocates several times the image's RGB size, so a limit
 * checked after decoding bounds nothing. */
bool clef_image_decode_limited(const uint8_t *data, size_t len, long max_pixels, clef_rgb *out, char *err, size_t errlen);
void clef_rgb_free(clef_rgb *img);

/* Qwen2VLImageProcessor.smart_resize: both sides rounded to the factor, scaled into the pixel
 * budget, aspect ratio kept. False for an aspect ratio of 200 or more (the reference raises). */
bool clef_smart_resize(int height, int width, int factor, long min_pixels, long max_pixels,
                       int *out_height, int *out_width, char *err, size_t errlen);

/* PyTorch upsample_bicubic2d_aa on a uint8 CHW tensor (antialias, align_corners=False), on
 * interleaved RGB. dst is [dh][dw][3]; src and dst may not alias. */
bool clef_resize_bicubic_aa(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int dh, char *err, size_t errlen);

/* The whole pipeline after decoding. out->patches is malloc'd. */
bool clef_image_preprocess(const clef_rgb *img, const clef_image_params *p, clef_image_patches *out, char *err, size_t errlen);
/* One temporal group at an already checked video resize geometry. Both frames have the same
 * source dimensions; pass the last frame twice to pad an odd-length video. */
bool clef_video_pair_preprocess(const clef_rgb *first, const clef_rgb *second, int rh, int rw,
                               clef_image_patches *out, char *err, size_t errlen);
void clef_image_patches_free(clef_image_patches *p);

/* Learned position table interpolation (get_vision_bilinear_indices_and_weights): for each patch
 * in merge-window order, the four table rows and bilinear weights of torch.linspace(0, side - 1, n)
 * sampling. idx is [n_patch][4], w is [n_patch][4]. */
void clef_image_pos_interp(int grid_h, int grid_w, int side, int merge, int32_t *idx, float *w);

#endif
