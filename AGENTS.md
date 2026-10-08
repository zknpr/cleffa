# Repository Guidelines

Cleffa is a C11 and Metal inference engine for Cloudflare Clef models on Apple Silicon. Read `README.md` for usage and `CLAUDE.md` for architecture and parity invariants.

## Project structure

Root `clef_*.c` and headers implement GGUF loading, JSON, tokenization, image decoding and preprocessing (`clef_image.c`, decoders vendored in `third_party/iris/`), request encoding, and inference. `clef_metal.m` dispatches kernels from `metal/clef.metal`, including the vision tower. `tests/` contains C, Python, and shell checks; `ref/` holds PyTorch oracles and the shared corpus; `bench/` contains standalone benchmarks. `tools/` handles conversion and Unicode generation. Model snapshots live in `model/` and `model-flash/`, converted weights in `gguf/`, and reference outputs in `golden/`; all are created locally (`./download_models.sh`, `ref/oracle*.py`) and git-ignored.

## Build, test, and development commands

Use macOS on Apple Silicon with Metal 4 support, Clang, and Accelerate. Run Python through `.venv/bin/python`.

- `./download_models.sh clef-flash`: download the pinned snapshot, verify its hashes, convert to `gguf/clef-flash.gguf`, verify the GGUF (`clef` for the 27B).
- `make`: build `clef`, `clef-server`, and `clef-tool`.
- `make test`: run head, JSON, image-pipeline, tokenizer and model-free regression checks; requires `gguf/clef-flash.gguf`, `model-flash/`, and Python dependencies.
- `./clef -m gguf/clef-flash.gguf requests.jsonl`: process JSONL requests.
- `./clef-server -m gguf/clef-flash.gguf --port 8080`: start the localhost HTTP server.
- `.venv/bin/python tests/test_parity.py gguf/clef-flash.gguf golden/clef-flash-f32 --dump`: compare against the FP32 oracle (against the BF16 golden it fails exactly where BF16 itself is wrong).

## Coding style and naming

Match existing four-space indentation, snake_case identifiers, and `clef_` API prefixes. Use uppercase constants and same-line C braces. Explain precision, ownership, and security decisions in comments. Propagate errors explicitly through return values and caller-provided error buffers. No formatter or linter is configured; preserve surrounding style and compiler warnings.

Edit `metal/clef.metal`, then let Make regenerate `clef_metal_src.inc`. Regenerate `clef_unicode.inc` deliberately with `make unicode`, preserving the expected Unicode version.

## Testing guidelines

Tests use standalone Python scripts, C assertions/comparisons, and shell checks. Follow `tests/test_*.py`, `test_*.c`, and `test_*.sh` naming. No coverage percentage is configured. Add regression cases for changed behavior. Require byte parity for host processing; use the FP32 oracle for numerical accuracy. Kernel changes must pass batch-invariance and NaN-poison checks, on the vision corpus too when they touch the tower or the embedding. Run relevant server tests for HTTP changes. Changing `ref/corpus.py` or `ref/corpus_vision.py` invalidates the corresponding golden data.

## Commits and pull requests

The repository is git (`cleffa`). Stage explicit paths and use concise imperative commit subjects. PRs should include Problem, Solution, Architecture, Per-file Changes, Security, and Test Plan, with relevant issues, exact test results, and performance evidence.

## Security and configuration

Treat requests and GGUF files as untrusted. Preserve bounds checks, request ownership, and strict-mode defaults. Keep public serving behind an authenticating, rate-limiting proxy. Import reference model code only from pinned snapshots. API keys (`jev.api` for `bench/jev_compare.py`) live in git-ignored files and are never printed or committed.
