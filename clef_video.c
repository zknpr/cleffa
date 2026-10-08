#include "clef_video.h"

#include <math.h>
#include <stdio.h>

static bool error(char *err, size_t n, const char *s) { snprintf(err, n, "video: %s", s); return false; }

bool clef_video_resize(int frames, int h, int w, long min_pixels, long max_pixels,
                       int *rh, int *rw, char *err, size_t errlen) {
    if (frames < 1 || frames > CLEF_VIDEO_MAX_FRAMES || h < 32 || w < 32 ||
        (double)(h > w ? h : w) / (h < w ? h : w) > 200 || min_pixels <= 0 || max_pixels < min_pixels)
        return error(err, errlen, "invalid resize geometry or pixel bounds");
    double a = rint(h / 32.0) * 32, b = rint(w / 32.0) * 32;
    int padded = (frames + 1) / 2 * 2;
    if (padded * a * b > max_pixels) {
        double beta = sqrt((double)frames * h * w / max_pixels);
        a = fmax(32, floor(h / beta / 32) * 32); b = fmax(32, floor(w / beta / 32) * 32);
    } else if (padded * a * b < min_pixels) {
        double beta = sqrt(min_pixels / ((double)frames * h * w));
        a = ceil(h * beta / 32) * 32; b = ceil(w * beta / 32) * 32;
    }
    if (a > CLEF_IMAGE_MAX_DIMENSION || b > CLEF_IMAGE_MAX_DIMENSION || a * b > CLEF_IMAGE_MAX_PIXELS)
        return error(err, errlen, "resized frame exceeds dimension/pixel cap");
    *rh = (int)a; *rw = (int)b;
    return true;
}
