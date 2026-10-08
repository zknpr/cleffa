#ifndef CLEF_VIDEO_H
#define CLEF_VIDEO_H

#include "clef_image.h"

/* Pinned Qwen3VLVideoProcessor defaults, shared by both Clef snapshots. */
#define CLEF_VIDEO_MIN_PIXELS 4096
#define CLEF_VIDEO_MAX_PIXELS 25165824
#define CLEF_VIDEO_MAX_FRAMES 768
#define CLEF_VIDEO_MAX_SOURCE_FRAMES 18000

bool clef_video_resize(int frames, int h, int w, long min_pixels, long max_pixels,
                       int *rh, int *rw, char *err, size_t errlen);

#endif
