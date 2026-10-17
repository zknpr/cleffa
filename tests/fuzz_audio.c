/* libFuzzer target for the untrusted-audio path of Clef-Omni requests: WAV parsing
 * (clef_audio_decode), JSON sample arrays and the request forms with their item, sample and
 * context budgets (clef_omni_encode_record_audio: data URLs, base64 objects, video soundtracks),
 * the mel frontend over clips of different lengths in one record, the chunk/window layout the
 * audio tower reads (clef_audio_layout_make), and audio/visual token placement and 3D positions
 * (clef_omni_append_visual_tokens, clef_omni_positions). Built with ASan and UBSan and no
 * recovery, so undefined behavior is a crash. The first input byte selects the mode:
 *   0  one WAV file
 *   1  up to four WAV files, each after a 16-bit little-endian length, as the clips of one record
 *   2  three option bytes, then a request JSON object ("audio", "videos")
 *   3  a synthetic record: still images, video frame groups and audio clips sized from the bytes,
 *      placed by the encoder's own code, then given positions
 * Beyond memory safety it checks what the GPU side relies on: finite features, a layout covering
 * exactly the record's audio tokens with windows inside it, a successful encode inside max_length
 * and a failed one leaving the budget alone, every audio placeholder claimed by exactly one clip
 * token, and positions succeeding on any record the encoder placed. Sample limits are small so a
 * mutated length cannot spend the run in the frontend; production compares its limits at the same
 * points. The tokenizer is stubbed for the newline the placement code encodes; nothing else here
 * tokenizes.
 *   make fuzz-audio
 *   .venv/bin/python -B tests/fuzz_audio_seeds.py fuzz/audio-seeds
 *   ./fuzz-audio -dict=tests/fuzz_audio.dict -max_len=65536 -fork=8 -ignore_crashes=1 -max_total_time=900 -artifact_prefix=fuzz/artifacts-audio/ fuzz/corpus-audio fuzz/audio-seeds
 * Not part of make test: it needs Homebrew LLVM (Apple's clang ships without libFuzzer). */
#include "../clef_record_omni.h"

#include <math.h>

enum { NEWLINE = 10, VIS_START = 1, VIS_END = 2, IMAGE = 3, VIDEO = 4, AUDIO = 5, AUDIO_START = 6, AUDIO_END = 7, TEXT = 100 };
enum { MAX_SAMPLES = 48000 };

static bool stub_encode(const clef_tokenizer *t, const char *text, size_t len, size_t max_tokens,
                        bool split_special, clef_tokens *out) {
    (void)t; (void)max_tokens; (void)split_special;
    return len == 1 && text[0] == '\n' && clef_tokens_push(out, NEWLINE);
}
#define clef_tok_encode_ex stub_encode
#include "../clef_record_omni.c"
#undef clef_tok_encode_ex

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "fuzz-audio: check failed: %s (line %d)\n", #c, __LINE__); abort(); } } while (0)

static clef_encode_opts options(void) {
    clef_encode_opts o = CLEF_ENCODE_DEFAULTS;
    o.audio.token_id = AUDIO; o.audio.start_token_id = AUDIO_START; o.audio.end_token_id = AUDIO_END;
    o.vision.image_token_id = IMAGE; o.vision.video_token_id = VIDEO;
    o.vision.start_token_id = VIS_START; o.vision.end_token_id = VIS_END;
    return o;
}

/* Features and their layout, as clef_metal_audio.m consumes them. */
static void check_features(const clef_audio_features *f, int n, int32_t want_tokens) {
    int64_t tokens = 0;
    for (int i = 0; i < n; i++) {
        CHECK(f[i].mel && f[i].frames >= 1 && f[i].frames <= f[i].padded_frames && f[i].padded_frames == f[0].padded_frames);
        for (size_t j = 0; j < 128 * f[i].padded_frames; j++) CHECK(isfinite(f[i].mel[j]));
        tokens += f[i].n_tokens;
    }
    if (want_tokens >= 0) CHECK(tokens == want_tokens);
    clef_audio_layout layout = {0};
    char err[256];
    if (!clef_audio_layout_make(f, n, &layout, err, sizeof(err))) {
        CHECK(strstr(err, "out of memory"));
        return;
    }
    CHECK(layout.tokens == tokens && layout.max_chunk_frames >= 1 && layout.max_chunk_frames <= 100);
    int64_t covered = 0;
    for (int c = 0; c < layout.n_chunks; c++) {
        const clef_audio_chunk *k = &layout.chunks[c];
        CHECK(k->clip >= 0 && k->clip < n && k->frames >= 1 && k->frames <= layout.max_chunk_frames);
        CHECK(k->frame >= 0 && (size_t)k->frame + (size_t)k->frames <= f[k->clip].frames);
        CHECK(k->first_token >= 0 && k->first_token + (k->frames + 7) / 8 <= layout.tokens);
        covered += k->frames;
    }
    int64_t frames = 0;
    for (int i = 0; i < n; i++) frames += (int64_t)f[i].frames;
    CHECK(covered == frames);
    for (int t = 0; t < layout.tokens; t++) {
        CHECK(layout.window_start[t] <= t && t < layout.window_end[t] && layout.window_end[t] <= layout.tokens);
        CHECK(layout.token_position[t] >= 0 && layout.token_position[t] < 13);
    }
    clef_audio_layout_free(&layout);
}

static void decoded_clips(clef_audio_samples *clips, int n) {
    if (!n) return;
    clef_audio_features f[4] = {0};
    char err[256];
    if (clef_audio_mel_batch(clips, n, f, err, sizeof(err))) {
        check_features(f, n, -1);
        for (int i = 0; i < n; i++) clef_audio_features_free(f + i);
    } else {
        for (int i = 0; i < n; i++) CHECK(!f[i].mel && !f[i].frames);
    }
}

static void wav_files(const uint8_t *d, size_t n, bool prefixed) {
    clef_audio_samples clips[4] = {0};
    int count = 0;
    char err[256];
    size_t total = 0;
    while (count < 4 && n) {
        size_t len = n;
        if (prefixed) {
            if (n < 2) break;
            len = (size_t)d[0] | (size_t)d[1] << 8;
            d += 2; n -= 2;
            if (len > n) len = n;
        }
        if (clef_audio_decode(d, len, MAX_SAMPLES - total, &clips[count], err, sizeof(err))) {
            CHECK(clips[count].data && clips[count].count >= 1 && clips[count].count <= MAX_SAMPLES - total);
            total += clips[count].count;
            count++;
        } else {
            CHECK(!clips[count].data && !clips[count].count);
        }
        d += len; n -= len;
        if (!prefixed || total >= MAX_SAMPLES) break;
    }
    decoded_clips(clips, count);
    for (int i = 0; i < count; i++) clef_audio_samples_free(clips + i);
}

static void request(const uint8_t *d, size_t n) {
    if (n < 3) return;
    clef_encode_opts o = options();
    static const size_t caps[] = {0, 201, 3200, MAX_SAMPLES};
    o.audio.max_samples = caps[d[0] & 3];
    o.audio.max_items = (d[0] >> 2) % 6;   /* 0 removes the item limit */
    /* Without a sample cap only the context bounds the clips: 40 tokens is 51,200 samples. */
    o.max_length = o.audio.max_samples ? 1 + d[1] * 4 : 1 + d[1] % 40;
    size_t reserve = (size_t)d[2] % ((size_t)o.max_length + 2);   /* may exceed max_length */
    jarena *arena = jarena_new();
    if (!arena) return;
    char err[256];
    const jval *req = json_parse(arena, (const char *)d + 3, n - 3, err, sizeof(err));
    if (req && req->type == J_OBJECT) {
        clef_record record = {0};
        const size_t before = reserve;
        if (clef_omni_encode_record_audio(req, &o, &reserve, &record, err, sizeof(err))) {
            if (record.n_audio) {
                CHECK(reserve > before && reserve <= (size_t)o.max_length);
                CHECK(record.n_audio_tokens >= record.n_audio && reserve - before >= (size_t)record.n_audio_tokens);
                /* the tower takes the features as one array; with no item limit, max_length bounds the count */
                clef_audio_features *f = malloc((size_t)record.n_audio * sizeof(*f));
                CHECK(f);
                for (int i = 0; i < record.n_audio; i++) {
                    CHECK(record.audio[i].token_rows);
                    f[i] = record.audio[i].feat;
                }
                check_features(f, record.n_audio, record.n_audio_tokens);
                free(f);
            } else {
                CHECK(reserve == before && !record.audio);
            }
        } else {
            CHECK(reserve == before);
        }
        clef_record_free(&record);
    }
    jarena_free(arena);
}

/* Production geometry: grids are even and a video's frame pairs share one grid. */
static clef_image_ref image(uint8_t h, uint8_t w, int group) {
    clef_image_ref r = {.video_group = group};
    r.pt.grid_h = 2 * (1 + h % 6);
    r.pt.grid_w = 2 * (1 + w % 6);
    r.pt.n_tokens = (r.pt.grid_h / 2) * (r.pt.grid_w / 2);
    return r;
}

static void synthetic(const uint8_t *d, size_t n) {
    size_t at = 0;
#define NEXT() (at < n ? d[at++] : 0)
    const int stills = NEXT() % 4, videos = NEXT() % 4, standalone = NEXT() % 4;
    const bool tracks = videos && (NEXT() & 1);
    clef_image_ref images[3 + 3 * 8];
    int n_images = 0;
    for (int i = 0; i < stills; i++) { const uint8_t h = NEXT(); images[n_images++] = image(h, NEXT(), 0); }
    for (int v = 0; v < videos; v++) {
        const int frames = 1 + NEXT() % 8;
        const uint8_t h = NEXT(), w = NEXT();
        for (int f = 0; f < frames; f++) images[n_images++] = image(h, w, v + 1);
    }
    /* Standalone clips first, then one soundtrack per video, as clef_omni_encode_record_audio joins them. */
    clef_audio_ref audio[3 + 3];
    const int n_audio = standalone + (tracks ? videos : 0);
    for (int a = 0; a < n_audio; a++) {
        const size_t lo = NEXT(), frames = 1 + (lo | (size_t)NEXT() << 8) % 3000;
        audio[a] = (clef_audio_ref){.video_index = a < standalone ? -1 : a - standalone};
        audio[a].feat.frames = audio[a].feat.padded_frames = frames;
        audio[a].feat.n_tokens = (int)(13 * (frames / 100) + (frames % 100 + 7) / 8);
        audio[a].token_rows = malloc((size_t)audio[a].feat.n_tokens * sizeof(int32_t));
        CHECK(audio[a].token_rows);
        for (int t = 0; t < audio[a].feat.n_tokens; t++) audio[a].token_rows[t] = -1;
    }
    clef_record record = {.images = n_images ? images : NULL, .n_images = n_images, .audio = n_audio ? audio : NULL, .n_audio = n_audio};
    clef_tokens ids = {0};
    const int before = NEXT() % 8, after = NEXT() % 8;
    for (int i = 0; i < before; i++) CHECK(clef_tokens_push(&ids, TEXT));
    clef_encode_opts o = options();
    char err[256];
    CHECK(clef_omni_append_visual_tokens(NULL, &record, &o, &ids, err, sizeof(err)));
    for (int i = 0; i < after; i++) CHECK(clef_tokens_push(&ids, TEXT));
    record.ids = ids;
    /* every audio placeholder belongs to exactly one clip token, and vice versa */
    uint8_t *claimed = calloc(ids.len ? ids.len : 1, 1);
    CHECK(claimed);
    for (int a = 0; a < n_audio; a++) for (int t = 0; t < audio[a].feat.n_tokens; t++) {
        const int32_t row = audio[a].token_rows[t];
        CHECK(row >= 0 && (size_t)row < ids.len && ids.ids[row] == AUDIO && !claimed[row]);
        claimed[row] = 1;
    }
    for (size_t t = 0; t < ids.len; t++) CHECK((ids.ids[t] == AUDIO) == (claimed[t] == 1));
    free(claimed);
    for (int i = 0; i < n_images; i++) {
        CHECK(images[i].tok_start >= 0 && (size_t)images[i].tok_start + (size_t)images[i].pt.n_tokens <= ids.len);
        for (int j = 0; j < images[i].pt.n_tokens; j++)
            CHECK(ids.ids[images[i].tok_start + j] == (images[i].video_group ? VIDEO : IMAGE));
    }
    if (clef_omni_positions(&record, err, sizeof(err))) {
        if (n_images) {
            CHECK(record.pos3);
            for (size_t t = 0; t < 3 * ids.len; t++) CHECK(record.pos3[t] >= 0);
        }
    } else {
        CHECK(strstr(err, "out of memory"));   /* any record the encoder placed must get positions */
    }
    free(record.pos3);
    for (int a = 0; a < n_audio; a++) free(audio[a].token_rows);
    clef_tokens_free(&ids);
#undef NEXT
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!size) return 0;
    switch (data[0] & 3) {
    case 0: wav_files(data + 1, size - 1, false); break;
    case 1: wav_files(data + 1, size - 1, true); break;
    case 2: request(data + 1, size - 1); break;
    case 3: synthetic(data + 1, size - 1); break;
    }
    return 0;
}
