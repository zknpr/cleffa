/* Differential libFuzzer target: every JPEG the vendored decoder accepts is decoded again with
 * libjpeg-turbo (what Pillow uses, so what the reference sees), and when libjpeg decodes it
 * without a single warning the RGB pixels must be identical. A sanitizer cannot see a wrong but
 * well-defined decode; this can (review #3: separate DC scans, 4:4:0 chroma and SOF1 were all
 * wrong or refused while two sanitizer fuzz runs stayed clean). Corrupt input that libjpeg
 * repairs with a warning is not compared: both decoders are allowed their own garbage there.
 * Refusals are not compared either; tests/test_image_diff.py classifies those against the
 * documented divergences. ASan and UBSan stay on, so this is also a sanitizer run.
 *   make fuzz-jpeg-diff
 *   ./fuzz-jpeg-diff -dict=tests/fuzz_image.dict -fork=14 -ignore_crashes=1 -max_total_time=1200 \
 *       -artifact_prefix=fuzz/artifacts-diff/ fuzz/corpus-diff fuzz/seeds
 * Needs Homebrew LLVM (libFuzzer) and Homebrew jpeg-turbo (static libjpeg). */
/* set by the decoder's range hook (see jpeg.h) when an input leaves the range encoders produce */
static _Thread_local int pathological;
#define JPEG_RANGE_HOOK(out_of_range) ((out_of_range) ? (void)(pathological = 1) : (void)0)
#include "../clef_image.c"

/* libjpeg defines its own JPEG_MAX_DIMENSION; the vendored decoder above was compiled with ours */
#undef JPEG_MAX_DIMENSION
#include <jpeglib.h>
#include <setjmp.h>
#include <stdio.h>

typedef struct {
    struct jpeg_error_mgr pub;
    jmp_buf jump;
    int warnings;
} diff_error;

static void diff_error_exit(j_common_ptr c) { longjmp(((diff_error *)c->err)->jump, 1); }
static void diff_emit(j_common_ptr c, int level) { if (level < 0) ((diff_error *)c->err)->warnings++; }
static void diff_output(j_common_ptr c) { (void)c; }

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 4 || data[0] != 0xff || data[1] != 0xd8) return 0;
    clef_rgb rgb;
    char err[256];
    pathological = 0;
    if (!clef_image_decode_limited(data, size, 1L << 20, &rgb, err, sizeof(err))) return 0;

    struct jpeg_decompress_struct cinfo;
    diff_error je;
    uint8_t *volatile ref = NULL;
    cinfo.err = jpeg_std_error(&je.pub);
    je.pub.error_exit = diff_error_exit;
    je.pub.emit_message = diff_emit;
    je.pub.output_message = diff_output;
    je.warnings = 0;
    if (setjmp(je.jump)) {   /* libjpeg refused what the vendored decoder accepted: not compared */
        jpeg_destroy_decompress(&cinfo);
        free(ref);
        clef_rgb_free(&rgb);
        return 0;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, (unsigned long)size);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;   /* Pillow decodes YCbCr to RGB; grayscale replicates like convert("RGB") */
    jpeg_start_decompress(&cinfo);
    const size_t w = cinfo.output_width, h = cinfo.output_height;
    ref = malloc(w * h * 3 + 1);
    if (!ref) longjmp(je.jump, 1);
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row = ref + (size_t)cinfo.output_scanline * w * 3;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    const int warnings = je.warnings;
    jpeg_destroy_decompress(&cinfo);
    if (warnings == 0 && !pathological) {
        if ((size_t)rgb.width != w || (size_t)rgb.height != h) {
            fprintf(stderr, "jpeg diff: size %dx%d, libjpeg %zux%zu\n", rgb.width, rgb.height, w, h);
            __builtin_trap();
        }
        size_t n = 0, first = 0;
        int worst = 0;
        for (size_t i = 0; i < w * h * 3; i++) {
            const int d = abs((int)rgb.rgb[i] - (int)ref[i]);
            if (d) { if (!n) first = i; n++; if (d > worst) worst = d; }
        }
        if (n) {
            fprintf(stderr, "jpeg diff: %zu of %zu values differ from libjpeg, max %d, first at pixel %zu\n",
                    n, w * h * 3, worst, first / 3);
            __builtin_trap();
        }
    }
    free(ref);
    clef_rgb_free(&rgb);
    return 0;
}
