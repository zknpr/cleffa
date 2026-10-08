# Vision

How images enter the engine, what was checked against the reference, and what it costs. Numbers
are from the one tested M5 Max (128 GB), one GPU job at a time, as the other reports.

## Pipeline

The reference (`joint_schema_model.py` on `transformers` 5.10.2) takes PIL images, runs
`Qwen2VLImageProcessor`, the Qwen3.5 vision tower (`Qwen3_5VisionModel`) and scatters the
features over the `<|image_pad|>` tokens that follow the template prefix; the decoder then runs
with the 3D rotary positions of `get_rope_index`. The engine does the same from base64:

1. **Decode** (`clef_record.c` `image_bytes`, `clef_image.c`, `third_party/iris`): an `images`
   entry in the hosted API's forms (its input schema, `src/content/workers-ai-models/clef.json`
   in Cloudflare's docs): a `data:` URL (prefix in any case) or `{"content_type", "base64"}`;
   a bare base64 string is an extension. PNG or JPEG by signature, to RGB as PIL's
   `convert("RGB")` would (alpha dropped, palette applied, gray replicated). A declared
   `content_type` must match the signature; `image/webp` is rejected.
2. **Resize**: `smart_resize` to multiples of 32 inside `[min_pixels, max_pixels]`
   (65,536 to 16,777,216 from `processor_config.json`), then PyTorch's native uint8
   antialiased bicubic CPU kernel, ported arithmetic for arithmetic (int16 weights, integer
   accumulation, width pass then height pass). That kernel is what torchvision's `resize` runs
   on Apple Silicon for uint8 input; it differs from PIL by ±1 in a few hundred pixels per image
   and from the float path by much more, so the port is specific to it.
3. **Normalize and patch**: `(x - 127.5) / 127.5` in f32, 16x16 patches in 2x2 merge-window
   order with the frame duplicated for the two temporal taps, `[n_patch][1536]`.
4. **Tower** (`metal/clef.metal` "vision tower", `clef_metal.m` `encode_vision`): patch
   embedding as one GEMM, bilinearly resampled learned positions, 27 blocks of LayerNorm,
   bidirectional attention with 2D rotary embedding, LayerNorm, GELU MLP; merger LayerNorm,
   four patches per row, GELU, projection to the backbone width. One dispatch sequence per
   image, independent of the batch.
5. **Backbone**: the features replace the placeholder rows in the embedding (`embed` kernel),
   and `attn_prep*` apply interleaved M-RoPE: pair `i` of the 32 rotary pairs turns by the
   token's temporal, row or column position for `i mod 3 = 0, 1, 2`, which is the plain RoPE for
   text tokens. `clef_record_positions` reproduces `get_rope_index`: an image's tokens share its
   start on the temporal axis and take their merged-grid row and column, and the text after it
   continues from start + max(rows, columns).

## Host parity

Byte parity with the reference, `make test`:

| Check | Test | Result |
|---|---|---|
| Decode, resize, patches, position interpolation | `tests/test_image.py` (60 random PNG/JPEG images of every color type, up- and downscaling, `media_kwargs` bounds, plus three DEFLATE variants and seven JPEG layouts Pillow cannot write: two with a DC scan per component, 4:4:0 chroma, SOF1, two with fill bytes before every marker and a one-component frame declaring 2x2; 13 with rewritten JFIF/Adobe markers and component ids, four of them RGB-coded; and nine for quantization-table latching, AC categories above 10, the IDCT range and runs overshooting their band; two PNGs with empty IDAT chunks; and four sequential frames split across scans, bytes after EOI, and a PNG chunk before IHDR) | 100/100 byte-identical to PIL, torchvision and the processor's `pixel_values` |
| Request encoding with images (ids, spans, image runs, 3D positions) | `tests/test_record.py` (43 image requests among 3,089) | 3,089/3,089 |
| Rejections | both | lone `media_kwargs` bound, other processor arguments, videos, non-list images, bad base64, truncated/unsupported images, placeholder text in parity mode |

Divergences from the reference (it answers or behaves differently; the engine errors):
videos; `media_kwargs` other than `min_pixels` and `max_pixels`, or only one of the two (the
processor silently ignores a lone bound), or bounds that are not positive integers within int32; a
data URL whose media type is not `image/png` or `image/jpeg` or contradicts the bytes (the object
form's `content_type` is checked the same way; the reference's PIL decode ignores both labels); an
image whose resized grid alone exceeds the 16,384-token context, refused before any resize (the
reference builds the patches and then fails on length); PNGs that are 16-bit, sub-8-bit grayscale or
interlaced; JPEGs that are CMYK, 12-bit, lossless or arithmetic-coded, whose first component is
sampled below another (Y 1x1 under Cb 2x2, which libjpeg accepts), or progressive with scans that
stop before full precision (libjpeg smooths those; the smoothing is not implemented), or whose
entropy data has fill bytes before a stuffed zero (`FF FF 00`, not standard); images over 64 MiB encoded,
16,384 pixels on a side or 64 megapixels decoded; a literal `<|image_pad|>` in request text in
parity mode (the reference raises on the placeholder count); the server's per-request image
and per-image token limits. EXIF orientation is ignored, as the reference ignores it.

Two input checks were added to the vendored JPEG decoder after an automated review of this
change (2026-10-07), each confirmed with a crafted file under AddressSanitizer against the
unmodified copy and rejected cleanly after: a progressive scan header with Se > 63 drove the
coefficient index past the zigzag table (global over-read feeding a heap write), and a legal
luma-under-chroma sampling layout read past the luma plane in the color conversion. Both
files are regression cases in `tests/test_image.py`.

A second automated review, of the pull request (2026-10-07), raised seven points; each was
checked against the code and, where it mattered, reproduced before the fix: (1) the per-image
token limit was enforced after `clef_image_preprocess` had resized and allocated the patches, so a
40x40 PNG with `media_kwargs` bounds of 67,108,864 pixels reached 1.85 GB resident (control at the
default bounds: 40 MB) before being refused; the encoder now computes the geometry first and refuses
the image at 38 MB and 0.26 s, also when its grid alone exceeds the context; (2) a data URL's
media type was not checked against the signature while the object form's `content_type` was, so
`data:image/jpeg;base64,<PNG>` was accepted; both forms now share one check; (3) `--max-images foo`
and `--max-image-tokens 1k` started the server with "at most 0 per request, 1 tokens each" because
`atoi`/`atol` read a typo as 0 (unlimited); both binaries now exit 2 on anything but a whole
number; (4) `media_kwargs` integers were converted with `atol`, undefined for out-of-range input
and safe here only because macOS saturates; they are parsed with `strtol` and `errno`; (5) the
startup warm-up ran text only, so the first image request was also the first allocation of the
vision scratch and the first touch of the tower's weight pages; the warm-up now includes an image
sized to the per-image limit (1,024 tokens, 743 ms at startup on Flash). Measured with the two
builds alternated, fresh server each, warm page cache, `v009` first then steady: old 0.816/0.814 s
then about 0.775 s, new 0.820/0.824 s then about 0.770 s. The 40-50 ms first-request difference
is the same with either build, so it is not the vision path (most likely activation capacity
growth to the request's 1,363 tokens, which a text warm-up at `--batch-tokens` would cover); the
image pass is kept for the cold-cache case, which was not measured; (6) `bench/vision_cache_latency.py`
asserted prefix-token reuse only, which a resume after the image could satisfy without reusing
features; it now requires the `image cache reused` count on every hit; (7) the
`tests/test-vision-buffers` rule lacked `$(DEPFLAGS)`. Regressions: `tests/test_record.py`,
`tests/test_server_images.py` (including the timed huge-bounds case) and `tests/test_cli_errors.py`.

The review of that fix commit raised three more, each reproduced before the change. (8) The
geometry check still ran after decoding, and decoding a PNG holds its inflated rows and the
decoded image: eight requests carrying a compressible 8192x8192 PNG (1.4 MB bodies) took the server
4.1 GB above its idle footprint in 0.16 s, each then refused by the token limit. Source images are
now limited to 16,777,216 pixels on the server (`--max-image-pixels`), enforced inside the
decoders where they read the PNG IHDR or each JPEG SOF, through a thread-local limit rather than a
separate pre-parse that could disagree with the decoder. The same probe showed a second
amplification the review did not name: an encoded 1,024-token image holds 24 MiB of f32 patches,
so sixteen requests of four one-pixel PNGs upscaled through `media_kwargs` (about 1 KB each) held
2.1 GB while queued. Connection threads encode concurrently, so per-image bounds multiply by
`--max-conn`; the server now admits at most `--max-image-requests` (8) image requests to decoding
and the queue at once, and frees their patches before writing the response. (9) A crafted GGUF
with a vision patch or temporal size of 2^24 overflowed `int` in the patch-width product before
the shape check rejected it (UBSan abort at `clef.c:95`); the product now follows the check, and
`tests/test_vision_config.c` runs crafted headers under UBSan in `make test`. (10) The image
feature buffer was sized with every activation-capacity growth, so a 16,347-token text request
allocated 256 MiB of features it never used; it now belongs to the vision scratch, sized by the
pass's image rows, and `tests/test_vision_buffers.m` checks that text-only capacity allocates none.
Logits are byte-identical to the previous build on the text and vision corpora (Flash) and the
27B vision corpus. After the change the eight huge-source requests are refused at the header
10 MB above the idle footprint (was 4.1 GB), the sixteen-request burst peaks 1.18 GB above idle
(was 2.07 GB; peak RSS 2.49 GB against 3.24 GB, and the idle footprint now includes the vision
scratch the startup warm-up allocates), and the 16,347-token text request's peak footprint falls
from 6.72 to 6.44 GB. Probe script and logs: `golden/review3-memory-2026-10-07/`.

A third round (on `767c02d`) found undefined behavior in the vendored JPEG decoder and a broken
check in the cache benchmark. `tests/test_jpeg_ub.c` builds five crafted JPEGs and decodes them
under UBSan and ASan; before the change each aborted: (11) a scan selecting a DC table no DHT
defined read `values[-1]` of the zeroed table; (12) a DHT mapping a DC code to 255 shifted by 255
bits; (13) a legal 1x1 baseline block with DC difference 2047 at quantizer 255 overflowed `int` in
the IDCT's second pass; plus two cases the review did not name, (14) the same difference repeated
over 8,192 blocks overflowing the DC prediction times the quantizer, and (15) a progressive DC
scan with Al = 13 overflowing the prediction's shift. ASan reported no access outside a buffer
(`values[-1]` stays inside the table struct); built without sanitizers, case 11 decoded instead
of being refused and case 13 rendered a far-above-white block as black. The decoder now refuses
scans that select undefined tables and DC symbols above 15 (both libjpeg's rules), wraps the DC
predictor and stores the coefficient in 16 bits before dequantization (as libjpeg's JCOEF), and
runs the IDCT in 64-bit integers (libjpeg's JLONG here); in-range blocks compute identically, so
the 63-image parity and the vision corpus logits are unchanged. (16) The benchmark's reuse check
from round one could never pass, because its sanitized environment dropped `CLEF_STAGE_TIME`,
which prints the line it looks for; both arms now set it.

A fourth round (on `571417e`) found two more. (17) A three-component baseline scan naming
component 1 three times passed the defined-table check, which looked at the scan's components,
while baseline decoding walks every frame component, so components 2 and 3 decoded with a table
never defined (`values[-1]` again, `tests/test_jpeg_ub.c` case 6). A scan may now name each
component once, as libjpeg requires; with baseline scans naming every component, the table check
covers everything they decode. (18) The context check compared each image alone with the whole
context, so an image of exactly 16,384 tokens, or several images that fit one at a time, were
preprocessed before the later length check refused the request: 542 MB and 492 MB resident for a
4096x4096 image and two 4096x2048 ones. The encoder now reserves the fixed prompt (57 tokens with
one image), the markers and the earlier images before preprocessing each one: 139 MB and 291 MB,
the first of the two images still fitting alone. The schema was not reserved at first; the eighth
round below adds it. `tests/test_record.py` checks both peaks.

The decoders were then fuzzed, since every review round had found something new by reading.
`tests/fuzz_image.c` is a libFuzzer target over decode, `smart_resize`, the antialiased resize,
patches and position interpolation, with ASan and UBSan and no recovery, a 4-megapixel source
limit, a PNG/JPEG token dictionary and 194 seed files (`make fuzz-image`; Apple's clang ships
without libFuzzer, so it uses Homebrew LLVM 22). The first run, on the decoder before round four
and with 14 workers for 30 minutes, executed 28.5 million inputs and found nothing, but its
coverage stopped at 770 edges: a mutated PNG almost always fails a chunk CRC or the zlib trailer,
so the fuzzer rarely got past them. The harness now also decodes a copy with the zlib header
check, the Adler-32 trailer and every chunk CRC repaired. The second run, on the decoder as of
round four, executed 48.7 million inputs in 30 minutes with 14 workers and reached 849 edges, with
no crash, out-of-memory or timeout. Both logs are in `golden/fuzz-image-2026-10-07/`. Two runs of
this length bound what was searched, not what is there; the dependency-free PNG inflater, which
cleffa builds never compile, was not fuzzed.

A fifth round (Codex on `d4d7661`) found three correctness gaps, each reproduced first. (19) A
progressive scan carrying one component's DC coefficients was walked as interleaved MCUs, so in a
4:2:0 file with a DC scan per component (`cjpeg -scans`) the luma blocks were read in MCU order or
past the scan: a 37x21 file was refused and a 32x32 one decoded with 1,534 values wrong by up to 49,
where libjpeg decodes both. Single-component DC scans now walk the component's blocks in raster
order, as AC scans already did, and both files are parity cases in `tests/test_image.py`. Fuzzing
had not found this: the fuzzer saw no valid multi-scan layout to mutate, and a wrong-but-defined
decode is not a sanitizer finding. (20) The rotary kernels pick each pair's position axis as
`lane % 3`, which is interleaved M-RoPE with section [11, 11, 10] and nothing else, but the loader
never read `clef.rope.mrope_section`; text hides a different layout because its three positions are
equal. The loader now requires that section for a model with a vision tower, and the converter
refuses any rope configuration but interleaved [11, 11, 10]. (21) Vision token ids were not
compared with the vocabulary, so a GGUF with an out-of-range id loaded and then failed every image
request after decoding it; `clef_vision_opts_load` now requires all four below `clef.vocab_size`.
`tests/test_vision_config.c` covers both with crafted headers.

Fuzzing cannot see a wrong but well-defined decode, so the decoders were then tested
differentially. `tests/test_image_diff.py` (`make test-image-diff`) builds 1,421 files with cjpeg
(eight sampling layouts, progressive scan scripts with spectral selection, successive
approximation and DC scans per component, restart intervals, quantizer settings), macOS sips,
ffmpeg and Pillow, decodes each with cleffa and with Pillow, and fails on any pixel difference or
on a refusal that is not a documented divergence. Its first run found two more classes. Vertical-only
chroma subsampling (4:4:0, `-sample 1x2,1x1,1x1`) was upsampled by row replication instead of
libjpeg's h1v2 triangle filter: 38 files up to 69 levels off. SOF1 frames (extended sequential,
8-bit), which libjpeg writes whenever a quantizer exceeds 255 (`-quality` below about 25), were
refused: 26 files. Both are fixed, with one file of each in `tests/test_image.py` (now 67 images).
After the fixes all 1,275 files Pillow decodes and cleffa should decode are identical; the rest are
the documented refusals (arithmetic, lossless, luma under chroma, 16-bit PNG) or files both refuse.

`tests/fuzz_jpeg_diff.c` (`make fuzz-jpeg-diff`, Homebrew LLVM and jpeg-turbo) carries the
comparison into fuzzing: every JPEG the vendored decoder accepts is decoded again with libjpeg-turbo
and must match pixel for pixel whenever libjpeg reports no warning, with ASan and UBSan on. Inputs
whose coefficients or IDCT outputs leave the range encoders produce are excluded through a no-op
hook in the IDCT: libjpeg masks such outputs with its range-limit table, wrapping where this
decoder clamps. Replaying the earlier fuzz corpus found two more. A progressive refinement run
that overshoots the band's end still writes its new coefficient in libjpeg, through the padding of
its natural-order table, without a warning; the vendored decoder now writes it at the same
position. And libjpeg reconstructs a progressive image whose scans stop before full precision with
block smoothing (`do_block_smoothing`, on in Pillow); that is not implemented, so such images are
refused under libjpeg's own `smoothing_ok()` condition. The first fuzz run then found the decoder
reading past an EOI preceded by a fill byte: both marker loops stepped over `FF FF` as a pair, so
in `FF FF D9` the EOI's own `FF` was consumed as fill, where libjpeg's `next_marker` skips fill bytes
one at a time. Pillow-written JPEGs with a fill byte before every marker were refused outright by
the old loops; two such files are now cases in `tests/test_image.py` (69 images). The second run
found the decoder ending a scan at `FF FF` inside entropy data. libjpeg's slow path reads a run
of FFs followed by `00` as one FF data byte (`jpeg_fill_bit_buffer`), but Pillow decoded noise
JPEGs with stuffed bytes rewritten as `FF FF 00` differently from the originals, without a
warning. Measured on 48x64 to 512x512 files: padding only the last stuffed byte decoded as
documented everywhere; padding the first one changed the pixels in three of four files, and
padding all of them in every file, and the result depended on how the input was buffered. The same Homebrew libjpeg-turbo 3.2 differed from itself
by up to 142 levels reading the same file from memory and through `djpeg`'s 4 KiB stdio buffers,
and Pillow's bundled 3.1 differed from both on some files, so version or Pillow's own buffering
may contribute too. In libjpeg-turbo's `jdhuff.c`, `decode_mcu` takes a fast path only when
`bytes_in_buffer >= BUFSIZE * blocks_in_MCU`, and `decode_mcu_fast` returns FALSE on a marker after
writing coefficients, leaving the slow path to redecode that MCU; that is the likely mechanism,
not traced further (measurements: `golden/fuzz-image-2026-10-07/ff-ff-00/`). The reference has no stable answer for this pattern, which the standard does
not allow, so such scans are now refused; the fuzz finding
and a padded Pillow file are expected refusals in `tests/test_image.py`. The third run found a
one-component frame declaring 2x1 sampling decoded with its blocks misplaced: libjpeg ignores a
one-component frame's factors (each block is its own MCU, in raster order), and the decoder now
treats such frames as 1x1; a Pillow grayscale file patched to declare 2x2 is a parity case (70
images). The encoder corpus also gained 260 jpegtran files (grayscale conversion and lossless
rotations of cjpeg output), all identical: 1,535 comparable files of 1,681. A fourth run on the
final decoder, 20 minutes with 14 workers, executed 63.5 million inputs with no pixel difference,
sanitizer report, out-of-memory or timeout. Its comparison covers files libjpeg decodes without a
warning and whose coefficients stay in the range encoders produce; refusals are classified by
`tests/test_image_diff.py`, not by the fuzzer. Logs and every finding are in
`golden/fuzz-image-2026-10-07/`.

A sixth round (Codex on `b1e7dd2`) raised three points. (22) Confirmed: every three-component JPEG
was converted from YCbCr, but libjpeg (`default_decompress_parms`, which Pillow leaves in charge)
treats one as RGB-coded and copies its planes when it has no JFIF marker and either an Adobe APP14
transform of 0 or, without an Adobe marker, component ids `R`, `G`, `B`. `cjpeg -rgb` writes such
files; none were in the encoder corpus, the fuzzer never produced one from YCbCr seeds, and the old
decoder got all 78 that the corpus now builds (baseline, progressive and 2x2-sampled) wrong by up to
255 levels. The decoder now records the JFIF and Adobe markers seen before the first scan, where
libjpeg fixes the colour space, and copies RGB-coded planes after the same upsampling.
`tests/test_image.py` adds 13 files with rewritten markers and ids, four RGB-coded and the rest
YCbCr controls (JFIF over Adobe, Adobe over ids, an unknown transform, unknown ids, and an RGB
transform that only arrives after the first scan); the seed corpus and fuzz dictionary gained
RGB-coded files and the APP14 tokens. (23) Confirmed: a progressive file cut before its first scan
and closed with EOI decoded as a blank image, where libjpeg fails with `JERR_SOF_NO_SOS` ("missing
SOS marker") and Pillow refuses it; such a frame is now refused (the baseline cut already was). (24) Refuted: two copies
of one `media_kwargs` bound do not pass the both-bounds check, because the JSON DOM merges a
repeated key as `json.loads` does (also through escapes, `max\u005fpixels`), so the object has one
member and is refused like any lone bound. `tests/test_server_images.py` now sends both repeated
forms as raw bodies to keep the merge and the check tied together. A fifth differential run on
the fixed decoder, from the earlier corpus plus the new seeds, executed 46.8 million inputs in 15
minutes with 14 workers with no pixel difference, sanitizer report, out-of-memory or timeout
(`golden/fuzz-image-2026-10-07/diff-run5.log`; the old build's corpus failures and the cut files
are in `review6/` there). The encoder corpus now has 1,759 files, 1,613 of them comparable and
identical.

A seventh round (Codex on `3f3eb03`) raised two points, measured against Pillow and libjpeg-turbo
3.2's `djpeg` on crafted files before any change. (25) Confirmed: a quantization table that is not
defined when its component's first scan starts (no DQT, a DQT for another id, or one only after
the scan) decoded from the zero-filled slot as flat gray, where libjpeg fails with
`JERR_NO_QUANT_TABLE` and Pillow refuses the file. Checking that also showed a parity bug Codex did
not name: libjpeg latches each component's table at its first scan (`latch_quant_tables`), so a
DQT redefining it between progressive scans does not apply, but the decoder dequantized with the
last table defined, 1,146 values off by up to 114. Tables are now required and latched per
component at the first scan. (26) Refuted as stated: libjpeg does not reject AC symbols above
category 10; it decodes them, and the engine already matched it on category 11 and on run/size
symbols with size 0. Category 15 did differ, by up to 128 levels, because its IDCT output was far
outside the pixel range, the class the differential fuzzer had excluded. Refusing large categories
would also have refused files that match. Instead, a block whose IDCT output leaves [-512, 511]
is now refused. Measured with libjpeg-turbo 3.2, that is the range where its builds agree: the C
path (`JSIMD_FORCENONE=1`) wraps an output of 512 to black, while the NEON path, which Pillow uses
on this Mac, clamps to about +-1,024 and then wraps differently. The first threshold tried,
[-384, 383] (the fuzzer's old exclusion), refused 15 `cjpeg -quality 3` checkerboards that
decoded identically. Over 3,692 files, from the encoder corpus and 40 photographs found on this
Mac re-encoded at quality 1 to 100 by Pillow and cjpeg, the largest output was 397, and 2,640
harsher synthetic encodes were all accepted. Sweeps of one- and two-coefficient blocks, with
outputs up to about 7,300, found no accepted file that differs from Pillow; the previous build
differed from about 1,024. The
range check also stops `tests/test_jpeg_ub.c` cases 3 and 4, now expected refusals; case 3 still
runs the whole 64-bit IDCT under UBSan, and case 4's long DC accumulation can no longer be reached
through a baseline file. The fuzzer's range exclusion is gone, so it now compares every file the
decoder accepts. Its first run under that rule (15 minutes, 14 workers, 57.8 million inputs) found
a 7x3 progressive file 805 seconds in, 29 of 63 values off by up to 18, that libjpeg's C and NEON
paths and Pillow all decode alike: in Cr's first AC scan a run passed position 63 at the end of the
data. libjpeg's `decode_mcu_AC_first` writes such a coefficient at `natural_order[k]`, a real
position when k is past Se but within 63 and position 63 through the table's padding beyond it,
then ends the band, without a warning; the decoder dropped it at the end of the data and refused
the file anywhere else. Baseline `decode_mcu` writes the same way past 63, which the decoder also
refused. Both now write where libjpeg does (refinement scans already did). `tests/test_image.py`
keeps the fuzz file and two crafted single blocks, a progressive run ending past Se = 5 and a
baseline run past 63; the old build differs on the first and refuses the others. A second
15-minute run on the final decoder executed 67.7 million inputs, every accepted file compared,
with no finding. Evidence: `golden/fuzz-image-2026-10-07/review7/`.

An eighth round (Codex on `600ddfe`) raised two points. (27) Confirmed as described: the image
budget left the schema out, on the stated ground that reserving it could refuse what the reference
accepts. That was wrong: the final length check already refuses any request whose fixed prompt,
images and schema exceed the context, so the reserve can include the schema and only refuse
earlier. A request whose schema could never fit still decoded and preprocessed its images: a
14,336-token image with a 2,500-token question peaked at 479 MB before the late refusal. The
schema is now built before the images and counted in each image's budget; the same request is
refused before preprocessing at 127 MB, a third `tests/test_record.py` peak. This is not an
amplification, since an accepted request with a short schema allocates the same patches, but it
also puts the schema's cheap validation before image decoding. (28) Refuted for this platform:
`png.h` grew its IDAT buffer with `realloc(idat_data, idat_len + chunk_len)`, and for an empty first
IDAT that is `realloc(NULL, 0)`. C lets that return NULL, and the following `memcpy(NULL + 0, ...,
0)` is undefined in C11. macOS returns a minimum-sized object instead. Even with an allocator
substituted to return NULL, measured under ASan and UBSan, the file decoded identically to Pillow
without a report: clang's pointer-overflow check excludes a null pointer plus zero, which C2y
(N3322) also defines. An empty IDAT is now skipped before reallocating anyway, since it costs one
branch, and two PNGs with empty IDATs, first and between data chunks, are parity cases. A
15-minute `make fuzz-image` run on the result (ASan and UBSan, 30.6 million inputs, 971 edges)
found nothing. Evidence: `golden/fuzz-image-2026-10-07/review8/`.

A ninth round (Codex on `ed8abd7`) raised one point. (29) Confirmed for a crafted model file:
`vis_qkv_rope` zeroes the attention kernels' 32 tail rows with one thread per head and patch, so a
four-patch image (the smallest) covers the tail only with at least 8 heads. A GGUF with 4 heads of
72 passed every load check, and its stale tail rows would meet masked zero probabilities, where a
NaN poisons the output. This was inferred from the dispatch, not run end to end, since it needs a
whole crafted tower. Both released towers have 16 heads, the reads stay inside the 128-row slack,
and a model file already controls its outputs, so the impact is limited to a malformed model. The
loader now requires at least 8 vision heads, as it enforces the kernels' other shape assumptions;
`tests/test_vision_config.c` adds a 4-head header, which loaded before and is now refused.

A tenth round (Codex on `a82292d`) raised three points, all confirmed. (30) A sequential (SOF0)
JPEG may code its components in separate scans, one each or Y then Cb and Cr; Pillow decodes such
files and the decoder, which required every component in the first scan, refused them. libjpeg
buffers coefficients for the whole frame when its first scan lacks a component and dequantizes at
output, writing only each block's DC and decoded AC coefficients into zero-initialized arrays, so a
component never scanned decodes as zero and one scanned twice keeps the first scan's values where
the second codes zeros, both without a warning. The decoder now does the same through the
progressive path's block walk and finish, with a sequential block decoder that stores quantized
coefficients. `tests/test_image.py` embeds a 24x16 cjpeg file with one scan per component and
derives both edge cases from it; `tests/test_image_diff.py` adds 312 cjpeg files from three
sequential scan scripts, three samplings and restart intervals, all refused before and identical
after (1,925 identical of 2,071). (31) The decoding pass skipped frame headers because a first pass
had looked ahead for one, so a scan placed before the frame header decoded with the later frame's
geometry, and a file with two frame headers decoded too; libjpeg fails both (`JERR_SOS_NO_SOF`,
`JERR_SOF_DUPLICATE`), and so does the decoder now, counting every SOFn marker type. (32)
`tools/convert.py` refuses any image processor but the reference's and records it as
`clef.vision.image_processor`, but the loader never read it, so a GGUF from elsewhere describing
other preprocessing loaded and was served with this one. The loader now parses it and checks every
field the converter checks, and that its patch geometry and pixel bounds equal the numeric keys the
engine uses; four crafted headers (bilinear resampling, CLIP normalization, a different pixel bound,
no record) loaded before and are refused now. A 15-minute differential run on the result executed
86.3 million inputs with no finding; coverage rose from 706 to 752 edges with the new path.
Evidence: `golden/fuzz-image-2026-10-07/review10/`.

An eleventh round (Codex on `a99a8a9`) raised one point, the state's counterpart of (27). (33)
Confirmed: with truncation refused, which is the server's default, a state that cannot fit is
refused, but only after the images were decoded and preprocessed. A 14,336-token image with a
2,500-token state peaked at 479 MB before that refusal. When truncation is refused, the state is
now tokenized before the images, capped at `max_length + 1` tokens (a limited encode is a prefix
of the full one), and counted in each image's budget. The later state check reuses those tokens,
and the same request is refused before preprocessing at 127 MB, a fourth `tests/test_record.py`
peak. With truncation allowed the state yields to the images and is tokenized after them, as before.
`tests/test_truncation.py` adds the same boundary sweep with a 256-token image: refusal is exactly
where the reference would truncate, and accepted ids equal reference-mode encoding.

A twelfth round (Codex on `1a0b186`) raised two points, both confirmed against `djpeg` and Pillow.
(34) A full baseline scan listing its components out of frame order was decoded in frame order,
coefficients landing on the wrong planes; libjpeg refuses all five other orders of a three-component
scan ("Invalid component ID"). Its `get_sos` searches each scan entry's component from that entry's
position on, so a partial scan may still reorder forward (Cr then Cb), and libjpeg decodes that in
scan order, as the buffered sequential path already did. The decoder now searches the same way.
(35) A refinement scan whose Al is not Ah - 1 (Ah 2 and Al 0, or Ah 1 and Al 1) was decoded where
libjpeg fails with `JERR_BAD_PROGRESSION`; it is now refused, and that rule replaces an `Ah > 13`
check that had refused the legal Ah 14, Al 13. `tests/test_image.py` adds both scan orders and both
approximation pairs as refusals (all decoded before), and a Y-then-CbCr cjpeg file with its chroma
scan reordered as a parity case (a guard: it decoded correctly before too). A 15-minute
differential run on the result executed 89.8 million inputs with no finding
(`golden/fuzz-image-2026-10-07/review12/`).

A thirteenth round (Codex on `e4f4e1d`) raised one point. (36) Confirmed for a crafted model file:
the loader did not require the four vision token ids to differ. With a start or end id equal to the
image id, the encoder's start and end tokens counted as placeholders, so every image request failed
its placeholder check after being decoded and preprocessed. The loader now requires image, start,
end and video ids to be distinct, as they are in both released models; three crafted headers
(start equal to image, end equal to image, start equal to end) loaded before and are refused now.

A fourteenth round (Codex on `b6cbe49`) raised one point. (37) Confirmed, measured on Pillow
progressive and baseline files cut two or six bytes before EOI, with and without the EOI. The bit
reader pads past a scan's data with 1-bits, and the progressive decoders took a Huffman failure there
as an implicit EOB. A progressive scan cut short therefore decoded without error: up to 50 levels
from Pillow when the file kept its EOI, where libjpeg fills the gap with a "premature end of data
segment" warning, and accepted when it had no EOI, where Pillow refuses it as truncated. Baseline
scans were refused in both forms already. A Huffman failure is now a failure everywhere, and the
reader counts the padding it appends, so a scan that consumed any of it, even inside a code that
happened to decode, is refused (`jpeg_overran`); lookahead may still buffer padding without
consuming it. Every cut file is refused now, Pillow's warning case as a deliberate divergence listed
in the README. `tests/test_image.py` adds a progressive file cut with and without EOI (both decoded
before); all 98 parity images and 1,925 corpus files still decode identically. A 15-minute
differential run on the result executed 133.7 million inputs with no finding
(`golden/fuzz-image-2026-10-07/review14/`).

A fifteenth round (Codex on `5d9ea53`) raised two points, measured first. (38) Confirmed: the
decoder returned pixels for a JPEG without EOI, the baseline path straight after its scan and the
progressive path at the end of the data. Pillow refuses every progressive one as truncated, but for
baseline its answer depends on what follows the scan: it refused a file ending there or one byte
later, and decoded one with eight bytes or two other bytes in place of EOI, so it follows its own
read-ahead, not the file. EOI is now required for every JPEG; the baseline image is held until the
decoding pass reaches it. The two baseline files Pillow decodes become refusals listed in the README;
bytes after EOI are still ignored. `tests/test_image.py` adds three files without EOI (all decoded
before) and one with bytes after EOI as a parity case. (39) Refuted: an ancillary chunk before IHDR
(tEXt, gAMA or a private chunk), which the PNG specification forbids, is decoded by Pillow, and the
decoder already matched it pixel for pixel; refusing it would diverge. It stays accepted, and a PNG
with tEXt before IHDR is a parity case. A 15-minute differential run on the result executed
185.9 million inputs with no finding; most mutated files now stop at the EOI requirement, as
libjpeg's warning already kept them out of the comparison (`golden/fuzz-image-2026-10-07/review15/`).

## Numerical parity

Corpus: `ref/corpus_vision.py`, 16 requests / 41 questions with 1 to 3 images each, 16 to
1,024 image tokens, PNG and JPEG, gray and alpha sources, JSON and ~1K-token text states.
Golden data: `ref/oracle.py --corpus vision --dtype float32`. `tests/test_parity.py` with its
default limits (exact token ids, argmax agreement, max |Δlogit| ≤ 0.05, max |Δp| ≤ 0.002,
per-layer relative L2 ≤ 0.01).

**Initial FP32 implementation, 2026-10-07** (before the optimizations below)

| Model, vision tower operands | ids | argmax | max \|Δlogit\| | max \|Δp\| | `v000` image rows rel. L2 | `v000` final norm rel. L2 |
|---|---|---|---|---|---|---|
| **clef-flash, f32** | 16/16 | **41/41** | **0.0126** | **0.0016** | 1.1e-5 | 1.1e-3 |
| clef-flash, 16-bit (`CLEF_VIS_F32=0`, FP16 per `ACT_VIS`) | 16/16 | 41/41 | 0.0724 | 0.0025 | 2.9e-3 | 5.6e-3 |
| **Clef 27B, f32**, `golden/clef-vision-f32` (streamed, `--safe-attn`) | 16/16 | **41/41** | **0.0066** | **0.0010** | 4.0e-6 | 1.4e-3 |

The tower runs 27 layers before any text is read, so operand rounding there compounds: with
FP16 operands two of 41 questions exceed the limits (`v009/outdoors`, the 1,024-token image, and
`v011/subject`). With f32 operands the image feature rows are 1.1e-5 from the FP32 oracle and
every question is within the limits, for the GEMM cost below. The text corpus is unchanged by
the vision changes: clef-flash 22/22 ids, 46/46, max |Δp| 0.0002 (`golden/clef-flash-f32`);
27B 22/22, 46/46, max |Δp| 0.0011 (`golden/clef-f32`, as before).

The streamed FP32 oracle (`ref/oracle_f32_stream.py --corpus vision`), which the 27B needs, is
bitwise identical to the full FP32 oracle on clef-flash for the vision corpus (41/41 questions,
all dumped layers and vision features).

Invariants on the vision corpus (`tests/test_batch.sh`, `tests/test_poison.sh`): logits
bitwise identical between `--batch 1` and `--batch 8`, and under NaN-poisoned buffers, in
both operand modes and with the reference attention kernel. `--template-cache` and
`--prefix-cache` give the plain logits on every image request (keyed image identity and feature
reuse are described below). `tests/test_server_images.py`: HTTP bytes equal the CLI's for every
request, with and without a cache key.

## Cost

`./clef --time`, warm, single request, f32 tower operands, 2026-10-07:

| Request | Image | Image tokens | Total tokens | clef-flash | Clef 27B |
|---|---|---|---|---|---|
| `v001` (webcam frame) | 336x252 JPEG | 80 | 373 | 145 ms | 435 ms |
| `v009` | 1024x1024 JPEG | 1,024 | 1,363 | 883 ms | 2,053 ms |

The same requests on the final build of this change (after the attention, GEMM and fusion work
below), measured the same way on 2026-10-07 as the median of the last three of four back-to-back
passes: `v001` 134 ms on clef-flash and 416 ms on the 27B; `v009` 773 ms and 1,857 ms. The
sections below hold the paired measurements that justify each step.

The 27B's text-only 346-token request measures 439 ms in the README's table, about what the
373-token webcam request with its 80 image tokens costs, so on the 27B the backbone dominates and
the tower adds little; the tower is the same size on both models (the merger projects to 5,120
instead of 4,096).

Serialized profiles (`CLEF_PROFILE=1`, not latency): on `v001` the tower's attention takes
5 ms and its GEMMs about 19 ms more with f32 operands than with 16-bit ones; on `v009` the
attention takes 277 ms (4,096 patches, quadratic) and the f32 GEMMs 165 ms more. The first
attention kernel (one lane per key, `vis_attention`, kept as the `CLEF_ATTN_REF=1` reference)
took 42 ms and 4.8 s respectively; the simdgroup-matrix kernel (`vis_attention_mma`, after
`attention_fa`) replaced it.

An image costs its tokens in the backbone like text: one token per 32x32 pixels of the
resized image, up to the reference's 16,384 (16.7 megapixels). The server rejects, by
default, more than 4 images per request or an image above 1,024 tokens
(`--max-images`, `--max-image-tokens`), naming the `media_kwargs.max_pixels` that would fit,
rather than downscaling silently; the reference's own downscaling is available through
`media_kwargs` with both bounds. A 336x252 webcam frame, clef-webcam's size for 4:3 video, resizes
to 320x256: 80 tokens (a 16:9 frame at 336x189 is lifted to the 65,536-pixel floor, 352x192,
66 tokens).

## FP32 attention optimization, 2026-10-07

Images with at least 2,048 patches (512 image tokens) now use `vis_attention_mpp`.
It keeps Q, K, V and softmax probabilities in FP32, uses 32-query/128-key Metal matrix
primitive tiles, and retains the online softmax and bounded scratch memory. Smaller images
keep `vis_attention_mma`; `CLEF_VIS_MPP=0` restores that kernel for every image. The reference
attention override still takes precedence. No image is resized differently and no tokens are
removed. The wider tile changes floating-point reduction order, so output is not promised to
be bit-identical to the old kernel.

The selection depends only on the individual image's patch count. Q/K/V each have 128 slack
rows, rewritten before attention, so the final tile cannot consume stale or poisoned values.
This adds 81 KiB to vision scratch compared with the original three 32-row tails; it does not
allocate a patches-squared score matrix.

`make test-vision-attention` runs the production kernels on 16 patch counts, including ragged
32/128-row tails and both sides of the 2,048-patch dispatch boundary. It compares sampled
outputs against float64 on flat and sharp softmax distributions, checks every output for
finiteness, verifies output guards and FP16/BF16 rounding, and checks per-record overflow
flag isolation. The sharpest sampled case puts both FP32 implementations near 3.5e-5 maximum
absolute error; the test caps it at 5e-5 and permits at most a 5% increase in aggregate squared
error over the original kernel. Full-model parity limits are unchanged.

After stopping the three old localhost keep-warm servers, paired standalone kernel
measurements at 4,096 patches were 9.91 ms for the original kernel and 7.25 ms for MPP,
about 27% less time per layer. This is the attention kernel only, on the tested M5 Max.

`bench/vision_latency.py` alternates the old and new kernel in separate CLI processes, drops
two warmup requests per process, checks token counts and response repeatability, and records
five timed requests per arm per round. It refuses other active Clef processes by default;
`--allow-other-engines` explicitly marks results provisional. CLI timings cover inference and
response construction, excluding image decoding, request encoding and model loading.

The initial isolated two-round comparison measured large-image `v009` medians of
991.2 → 861.6 ms on Flash and 2,121.0 → 2,055.1 ms on 27B. The Flash control drifted between
rounds, so a separate three-round large-image confirmation is recorded alongside it. That
confirmation measured **1022.4 → 926.0 ms (9.4% less time)**,
with fifteen samples per arm. The two-round 27B result is **3.1% less time**. Small
`v001` images use the same kernel in both arms and have identical response bytes; their timing
variation illustrates the noise floor. These are warm measurements from one M5 Max, not a
claim about other devices or cold starts. Raw samples are in `latency-isolated.json` and
`latency-flash-confirm.json` in the evidence directory. The earlier `latency.json` is retained
but marked provisional because the three old servers were still running.

Validation: both vision corpora still pass 41/41 questions against FP32 with the default
logit/probability limits, including the original small-image layer dump. Maximum probability
error remains 0.0016 on Flash and 0.0010 on 27B. Vision batch invariance and NaN poisoning pass
in both operand modes, and reference-attention poisoning passes. The text corpora pass 46/46
on both models. `make test`, `make test-errors` and the HTTP image suite pass.

Only `v009` changes logits in the vision corpus. Mean absolute probability error across all
options changes from 6.906e-5 to 6.867e-5 on Flash and from 1.449e-4 to 1.499e-4 on 27B.
For `v009` itself, maximum probability error is 6.62e-5 → 6.89e-5 on Flash and
1.57e-4 → 3.84e-4 on 27B. Maximum logit error there is 0.000619 → 0.003949 and
0.001050 → 0.001492 respectively. All remain below the unchanged limits, but the 27B's
probability error increases slightly; unchanged decisions do not imply identical scores.
The per-request metrics and raw logits are in `quality.json` and `quality-*.jsonl`.

The additional `v009` layer references are generated separately under the evidence directory
from the verified pinned snapshots, using `oracle_f32_stream.py --corpus vision --only v009`
and `--safe-attn` for 27B. Both fresh references reproduce the existing golden logits bitwise.
Both large-image layer comparisons pass: embedding relative L2 is 2.03e-5 on Flash and
1.50e-5 on 27B; final-norm relative L2 falls from 0.00223 to 0.00203 on Flash and from
0.00977 to 0.00847 on 27B, below the existing 0.01 limit. The existing corpus and golden data are unchanged.

The raw kernel, parity, error, HTTP and timing logs are under
`golden/vision-opt-2026-10-07/`. Compensated half-operand attention was also tried, but regressed
some small shapes and was slower than the selected FP32 path on the large image. Materializing
bounded score slabs was slower as well. Neither experiment is enabled or retained in the
production source.

## GEMM and bias optimization, 2026-10-07

The default now compensates **non-residual** vision projections (patch, QKV, MLP up,
and merger). The initial implementation materialized each producer's FP32 output before
`vis_split_gemm` wrote `hi = half(x)` and
`lo = half((x - float(hi)) * 2048)` into two planes; `vis_gemm_comp` computes the two
FP32-accumulating products against the unchanged BF16 weights and combines them as
`hi_product + lo_product / 2048`. Scaling preserves small residuals across FP16's limited
exponent range. This is not bitwise FP32 arithmetic.

Residual projections remain direct FP32, using a 16x128 tile with the same per-element
reductions as the previous 32x128 tile. Applying compensation to these projections too
failed the Flash probability gate (0.00235 > 0.002), so that variant was rejected.
A three-plane variant passed parity but gave little full-request improvement and was
also rejected. Compensated tiles use 32 rows below 1,024 rows and 64 above, selected per
image. Tests require exact output across tile shapes and packed row offsets.

The separate residual-bias passes are folded into the following LayerNorm. Each thread
updates its own residual elements before computing the norm; the FP32 rounding and
writeback are unchanged. This removes 54 dispatches per image. Both complete vision
corpora have byte-identical logits before/after this fusion and FP32 tile change, with
the same compensated projections enabled.

`CLEF_VIS_COMP=0` restores direct FP32 products throughout the tower. Clearing `ACT_VIS`
in `CLEF_ACT_F16` or running a BF16 retry also disables compensation. The split reuses
the record-local saturation/overflow path: NaNs and magnitudes past the FP16 limit flag
only that record and write finite halves. The retry uses the original FP32 tower operands.
`CLEF_VIS_F32=0` retains the older plain 16-bit mode. No image is resized differently.
The `CLEF_VIS_COMP=0` control reproduces the saved starting binary's logits byte-for-byte
over the full vision corpus on both models.

The initial scratch was one reusable pair of half planes sized for the largest image's
largest input, 67.25 MiB at 4,096 patches. The subsequent producer fusion below reduces it
to 24 MiB. Capacity growth remains atomic on allocation failure.
The keep-warm buffer list is sized from its activation and KV arrays, including the new
scratch and all 16 KV slots; an ASan regression covers the maximum populated list.
Reinstating the old 64-slot bound makes that regression fail with a stack-buffer-overflow;
the derived bound passes.

On the same M5 Max, alternating saved starting binary / candidate, three rounds with
three warmups and seven measured requests per arm (21 samples per result):

| Model / image | Starting build, ms | New build, ms | Median reduction |
|---|---:|---:|---:|
| Flash, `v001` webcam | 142.4 | 132.4 | 7.0% |
| Flash, `v009` large image | 884.5 | 815.8 | 7.8% |
| 27B, `v001` webcam | 441.6 | 435.6 | 1.4% |
| 27B, `v009` large image | 1872.3 | 1811.9 | 3.2% |

The starting build already includes the attention optimization above. These are warm CLI
inference times, excluding image decoding, request encoding, loading, response serialization
and HTTP. The small 27B change is within run-to-run variation. Its large-image
rounds improved by 0.3–4.4%, so treat 3.2% as this run's median, not a guaranteed gain.
Evidence: `golden/vision-gemm-2026-10-07/latency-final.json` includes samples, request and
binary hashes, responses, and interference preflight. `timed-clef` preserves the measured
binary; the subsequent rebuild only adjusts keep-warm capacity and comments.
Reproduce against a saved binary with:

```sh
.venv/bin/python -B bench/vision_latency.py golden/vision-gemm-latency.json \
  --baseline-binary /path/to/saved-clef --rounds 3 --samples 7 --warmup 3
```

Both models retain 41/41 vision decisions and 46/46 text decisions against FP32. Host
tokenization is exact. Numerical error across every vision question:

| Metric | Flash starting → new | 27B starting → new |
|---|---:|---:|
| Mean absolute probability error | 6.86653e-5 → 6.35583e-5 | 1.49865e-4 → 1.39946e-4 |
| Maximum absolute probability error | 0.00161825 → 0.00089441 | 0.00100868 → 0.00099687 |
| Mean absolute logit error | 0.00106286 → 0.00123315 | 0.00096335 → 0.00084136 |
| Maximum absolute logit error | 0.01259571 → 0.01722068 | 0.00663567 → 0.00702220 |

Probability error improves on average, but this is **not a uniform numerical improvement**:
Flash logit error increases. Every unchanged acceptance limit still passes. The fresh `v009`
layer references also pass: image-embedding relative L2 is 2.26e-5 / 1.19e-5 (Flash / 27B),
and final-norm relative L2 is 0.00204 / 0.00953. The latter 27B value is close to the existing
0.01 limit. Per-request errors and raw outputs are in `quality.json` and `quality-*.jsonl`
in the same evidence directory; no oracle or acceptance threshold was changed.

Verification: `make test`, `make test-errors`, `make test-vision-attention`, the vision GEMM
kernel tests (121 shapes, FP64 samples, exact tile/packing parity, poisoned tails, finite
saturation and record-local flags), bias/LayerNorm fusion in all three output types,
allocation-failure recovery and ASan keep-warm coverage; vision batch and poison tests
on both models in default and plain 16-bit modes; vision-only forced overflow and dump
retry parity on both models; full text/vision parity and large-image layer dumps; HTTP
image/limit/cache checks on Flash and reference-attention poison checks. Commands:

```sh
make test-vision-gemm
.venv/bin/python -B tests/test_vision_gemm.py gguf/clef-flash.gguf golden/clef-flash-vision-f32/requests.jsonl
```

## Exact fusion follow-up, 2026-10-07

Norm and merger-GELU producers now write the compensated half planes directly. They retain
the standalone conversion's FP32 arithmetic, saturation and record-local overflow flags.
Only patch embedding needs separate split scratch, reducing that allocation from 67.25 to
24 MiB at 4,096 patches. This removes 56 conversion dispatches per image. QKV/RoPE preparation
also clears the attention kernel's masked tail, removing another 81 fill dispatches.
Neither change changes an output bit on either complete vision corpus.

Controlled repeated-dispatch probes measured producer/split fusion at 1.08–1.35x and
QKV/tail fusion at 1.01–1.87x across the tested shapes. Complete fresh-request timings were
noisy: an apparent regression in blocked runs did not persist in the interleaved baseline,
cache-only and fused comparison. No reliable whole-request speedup is claimed for these
fusions. Their verified benefits are fewer dispatches and 43.25 MiB less scratch at 4,096
patches. See `split-timing.log`, `tail-timing.log` and `paired-stage.json` in the evidence directory.

The producer regression compares every half-plane bit, residual writeback and overflow flag
against the separate producer and split, including large values, low forced limits and ragged
widths. The attention regression starts with NaN-filled Q/K/V storage, verifies the exact
cleared tail and untouched guard, then checks FP64 samples and both output precisions.

The corresponding ledger, source snapshots, raw outputs and timing runs are under
`golden/vision-exhaust-2026-10-07/`. Further GEMM/GELU fusion, thirteen backbone tile/order
variants and an attention accumulator-initialization shortcut were screened. Kernel parity
passed, but full-request latency or broader timing checks rejected them. Their experimental
builds remain only in the ignored evidence directory, not the shipped dispatch.

## Caches

Images sit right after the 36 template tokens, so the fixed-template entry still serves image
requests normally. Keyed caching (`--prefix-cache` on the CLI; `--prefix-cache-mb` plus
`X-Clef-Prefix-Cache` over HTTP) also retains merged image features and exact owned copies of
the canonical patch bytes. Reuse requires identical pixel bytes, image order, token placement,
patch dimensions and grid geometry within the same engine instance. Token placeholders alone
never establish a match.

A hit skips the vision tower even if changed text prevents backbone prefix reuse, or the last
checkpoint lies inside an image's token run. Feature rows retain the full image-list layout;
only suffix tokens read them. Changing images invalidates the backbone prefix too. A failed
or overflowing pass is never eligible for reuse, and the BF16 retry follows the ordinary path.
CPU patch copies and GPU features both count toward admission and eviction. Image requests can
cache at a 32-token boundary; text retains the 128-token minimum. Cache keys are client-visible
identifiers, not authentication: assign them at an authenticating proxy as described in README.

`tests/test_vision_cache.py` checks exact raw logits on every corpus fill/hit, same-shape pixel
changes, image order and resize changes, altered text/questions, text/image transitions, poisoned
scratch and overflow invalidation. `tests/test_server_images.py` checks HTTP response parity,
key isolation, replacement and rejection of entries above budget. Cache allocation failure,
recovery, accounting and keep-warm are covered in `tests/test_vision_buffers.m`.

Warm CLI timing while alternating two question schemas on the same image, two reversed-order
rounds, four warmups and eight measured requests per arm and round:

| Model / image | Uncached median | Cache-hit median | Speedup |
|---|---:|---:|---:|
| Flash, `v001` | 126.1 ms | 85.7 ms | 1.47x |
| Flash, `v009` | 797.2 ms | 96.1 ms | 8.30x |
| 27B, `v001` | 419.1 ms | 310.4 ms | 1.35x |
| 27B, `v009` | 1890.6 ms | 321.9 ms | 5.87x |

Every raw-logit response is byte-identical across the uncached and cached sequences. This
measures inference, excluding decode, request encoding, model loading, response serialization
and HTTP. Initial fills still compute the complete request and allocate/copy retained
state; these hit numbers do not describe fill latency or new frames. Evidence:
`golden/vision-exhaust-2026-10-07/latency-cache.json` and `cache-timed-clef`.
Reproduce with `.venv/bin/python -B bench/vision_cache_latency.py golden/vision-cache-latency.json`.

The final build was also measured through loopback HTTP, including image decoding, record
encoding and response serialization. The same warm server alternated plain and keyed requests,
reversing arm order each pair and alternating two question schemas. After four warmup pairs,
eighteen measured pairs gave these medians; every paired response was byte-identical:

| Model / image | Plain HTTP | Cache-hit HTTP | Speedup |
|---|---:|---:|---:|
| Flash, `v001` | 150.7 ms | 102.0 ms | 1.48x |
| Flash, `v009` | 837.8 ms | 115.8 ms | 7.23x |
| 27B, `v001` | 438.9 ms | 312.1 ms | 1.41x |
| 27B, `v009` | 1921.1 ms | 358.7 ms | 5.36x |

Model startup and client-side JSON serialization are excluded. These are warm hit figures,
not new-frame or initial-fill latency. Desktop load changed absolute timings between runs;
the arms above were interleaved within each run. The script, raw samples, server binary and
its hash are `http_latency.py`, `latency-http.json`, `final-timed-clef-server` and
`final-binaries.json` in the evidence directory.

## CPU image processing follow-up, 2026-10-07

Both new images and cache hits still need decoding and canonical patches. The host path now
uses table lookup for base64, system zlib for PNG inflate/CRC, transfers already-RGB decoder
buffers instead of copying them, skips identity-resize allocation, shares bicubic weights
across RGB, and looks up the 256 exact FP32 normalized values. Still-image temporal planes
are copied contiguously. Bicubic accumulation order, resize dimensions and every patch bit
are preserved; no image resolution or model precision is reduced.

PNG output remains bounded by validated dimensions. Inflate must consume one complete stream
and produce exactly the required bytes; checksum errors, dictionaries, truncated/extra output
and trailing compressed data are rejected. Base64's bulk path still rejects padding outside
the final quartet and all non-alphabet bytes. Tests cover every invalid byte/position, PNG
stored/fixed/dynamic blocks, malformed streams and corrupt IHDR/IDAT/IEND checksums.

CPU-only timings on the M5 Max, median of three rounds, each with four warmups and thirty
measured samples. Arm order reverses between rounds; totals include base64, decode and
preprocessing, excluding JSON parsing, inference and response serialization:

| Image | Previous total | Current total | Speedup |
|---|---:|---:|---:|
| 1024x1024 RGB gradient PNG | 16.67 ms | 6.49 ms | 2.57x |
| 1024x1024 RGB random PNG | 62.74 ms | 8.99 ms | 6.98x |
| 1920x1080 RGBA random PNG, resized to the 1M-pixel budget | 174.87 ms | 33.84 ms | 5.17x |
| Corpus `v001`, JPEG | 1.84 ms | 1.53 ms | 1.20x |
| Corpus `v009`, JPEG | 12.63 ms | 9.98 ms | 1.27x |
| Corpus `v013`, small PNG with upscaling | 0.67 ms | 0.43 ms | 1.56x |

These are host-stage gains, not multipliers for complete inference. The original large vision
fixture is JPEG, so it benefits from preprocessing but not the PNG decoder changes. Raw
per-stage timings, fixtures and before/after binaries are in `latency-host.json`, `host-*.log`
and `image-*-requests.jsonl` under `golden/vision-exhaust-2026-10-07/`. Reproduce the stage
measurement with `make image-bench` and `./image-bench REQUESTS.jsonl`.

## Follow-up validation

Both models retain all 41/41 vision decisions and 46/46 text decisions against the FP32
oracles, using unchanged tolerances. Every raw vision logit is byte-identical to the build
before this follow-up. The retained changes improve speed or memory use; they make no
evaluation-quality claim beyond preserving those outputs.

The GPU/cache changes passed batch invariance, NaN poison, vision-only overflow/BF16 retry,
reference-attention and narrow-activation checks, allocation failure/recovery, engine ownership,
cache budget/LRU/key isolation and HTTP parity. The subsequent host changes passed `make test`,
`make test-errors`, another complete vision/cache/HTTP run on both models, 203/203 additional
image cases against Pillow/torchvision, and 103/103 image cases under ASan. Base64's exhaustive
invalid-byte/position checks also passed ASan. The two command/result ledgers are
`validation.json` and `validation-host-final.json` in the evidence directory.

## Not done

- Live or unknown-length video streams; video codecs outside H.264, HEVC, ProRes and MJPEG.
- WebP (the hosted API accepts it); 16-bit, low-bit and interlaced PNGs; CMYK JPEGs.
- Attention remains quadratic in patches. The large-image FP32 MPP path reduces its cost;
  further tensor-unit work remains open.
