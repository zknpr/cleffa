# Third-party notices

## ds4 (MIT)

cleffa exists because of [ds4](https://github.com/antirez/ds4). It took ds4's approach whole: a
narrow, model-specific C + Metal engine; one GGUF wrapped zero-copy as a Metal buffer; parity tested
against the reference implementation; a download script that makes the setup reproducible. Some
code is adapted directly. The Qwen byte-level pre-tokenizer in `clef_tok.c` (`pretokenize`) follows
ds4's `ds4.c` implementation, changed to the Qwen2 split regex that Clef's tokenizer uses. ds4's
license:

```
MIT License

Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors
Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## iris PNG and JPEG decoders (MIT)

`third_party/iris/png.h` and `third_party/iris/jpeg.h` are single-header image decoders by
Salvatore Sanfilippo, taken from ds4's `third_party/iris` copy, which carries ds4's decode limits
for untrusted input and its JPEG changes for libjpeg agreement (centered chroma interpolation,
rounded YCbCr conversion, retained progressive scan bits). cleffa changes `jpeg.h` as documented
in its header comment: scan headers (Ss/Se/Ah/Al) are validated before decoding, a multi-component
image's first component must carry the largest sampling factors, scans may only select Huffman
tables a DHT defined and name each component once, DC table symbols above 15 are refused, the DC predictor wraps and is stored
as 16 bits before dequantization, progressive scans validate each coefficient's prior bitplane
and consume a cumulative block-work budget, the IDCT computes in 64-bit, and a progressive DC scan of one
component walks that component's blocks in raster order. Matching libjpeg on files from other
encoders, 4:4:0 chroma uses its h1v2 triangle filter, SOF1 frames decode as baseline, a run
overshooting its band (refinement, first scan or baseline) writes its coefficient where libjpeg does, progressive images libjpeg would
reconstruct with block smoothing are refused, 0xFF fill bytes before a marker are skipped one at a
time, entropy data with fill bytes before a stuffed zero is refused, a one-component frame
decodes as 1x1 whatever sampling it declares, three components are copied as RGB when libjpeg would
(no JFIF marker, and an Adobe transform of 0 or, without an Adobe marker, component ids R, G, B),
a progressive frame that reaches EOI before any scan is refused, each component's quantization
table must be defined by its first scan and is latched there as libjpeg does, and a block whose
IDCT output leaves [-512, 511], where libjpeg's builds disagree, is refused. A sequential frame may
split its components across scans, which are buffered and finished like a progressive frame's, and
a scan before the frame header, or a second frame header, is refused. Scan components are looked
up from the scan position on, as libjpeg does, and a refinement scan must have Al = Ah - 1. A scan
whose entropy data ends early is refused, and so is a file without EOI. `clef_image.c` includes the
headers and sets their limits; `tests/test_image.py` checks their output against Pillow, and it and
`tests/test_jpeg_ub.c` and `tests/test_jpeg_regressions.py` hold the crafted files behind those changes. The PNG header skips empty IDAT chunks and has optional inflate and CRC32 hooks;
cleffa supplies these through the system zlib library, preserving size and checksum validation.
Their license (`third_party/iris/LICENSE`):

```
MIT License

Copyright (c) 2026 Salvatore Sanfilippo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## PyTorch (BSD-3-Clause)

`clef_image.c`'s resize reproduces the arithmetic of PyTorch's CPU `upsample_bicubic2d_aa`
kernel for uint8 input (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`: antialiased Keys cubic
weights normalized in double, rescaled to int16 at the precision chosen from the largest weight,
integer accumulation), because that is what the reference's image processor runs on Apple
Silicon. It is a re-implementation in this project's code, not a copy; PyTorch's license is
BSD-3-Clause (https://github.com/pytorch/pytorch/blob/main/LICENSE).

## Unicode Character Database (Unicode License v3)

`clef_unicode.inc` is generated by `tools/gen_unicode.py` from the Unicode Character Database,
version 15.0.0, as bundled with Python 3.12's `unicodedata` module. It covers general categories,
White_Space, canonical combining classes and canonical decompositions/compositions.

```
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
```

## Clef and Clef-Flash (Apache-2.0)

cleffa does not include or redistribute model weights or Cloudflare's code.
- **Downloads:** `download_models.sh` fetches [Cloudflare/clef](https://huggingface.co/Cloudflare/clef)
  and [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) from Hugging Face at
  pinned revisions. Both are licensed under the Apache License 2.0; the license file ships with each
  snapshot.
- **Your conversions:** the GGUF files that `tools/convert.py` produces are derivatives of those
  weights. If you redistribute them, the Apache-2.0 terms apply.
- **Reference code:** the reference oracles in `ref/` import `joint_schema_model.py` from the
  downloaded snapshot. That code is not copied into this repository.

## FFmpeg

The optional `tools/video_request.py` client tool invokes locally installed `ffmpeg` and `ffprobe`
executables for MP4/MOV conversion. The engine does not link FFmpeg. Its source and binaries are
not vendored here.
Its license depends on the build configuration; see the installed distribution's notices and
[FFmpeg's license page](https://ffmpeg.org/legal.html).

cleffa is an independent project. It is not affiliated with or endorsed by Cloudflare.
