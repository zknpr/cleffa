#!/bin/sh
# cleffa model setup, after ds4's download_model.sh. Cloudflare publishes safetensors, not GGUF, so
# the setup is four steps per model:
#   1. download the snapshot at a pinned revision (Hugging Face `hf download`);
#   2. check the full inventory, sizes and hashes against checked-in pinned manifests
#      (tools/verify_snapshot.py), independently of local metadata. The oracles import
#      joint_schema_model.py from the snapshot, so it is code that runs;
#   3. convert it to one GGUF (tools/convert.py);
#   4. check the GGUF against the safetensors tensor by tensor (tests/verify_gguf.py).
# The conversion is written to a temporary file and moved into place only after it verifies.
set -eu

FLASH_REPO="Cloudflare/clef-flash"
FLASH_REV="17f0b0ad64efb65d273590632833508766b2aae6"
CLEF_REPO="Cloudflare/clef"
CLEF_REV="2f3de3dd85f379784083b0814d997ab627200f0c"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

usage() {
    cat <<EOF
cleffa model setup: download, verify, convert to GGUF, verify.

Usage:
  ./download_models.sh clef-flash [--skip-download] [--token TOKEN]   # 9B, ~18 GB  -> gguf/clef-flash.gguf
  ./download_models.sh clef       [--skip-download] [--token TOKEN]   # 27B, ~54 GB -> gguf/clef.gguf
  ./download_models.sh all        [--skip-download] [--token TOKEN]

  --skip-download  use the snapshot already in model-flash/ or model/ (it is still verified)
  --token TOKEN    Hugging Face token (or set HF_TOKEN); the Clef repositories are public

The Python environment (.venv, from requirements.txt) is created on first use, with uv when it is
installed and python3.12 -m venv otherwise.
EOF
}

die() { echo "download_models.sh: $*" >&2; exit 1; }

target=""; skip_download=0; token="${HF_TOKEN:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        clef-flash|clef|all) [ -z "$target" ] || die "more than one model given"; target=$1 ;;
        --skip-download) skip_download=1 ;;
        --token) [ $# -ge 2 ] || die "--token needs a value"; token=$2; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
    shift
done
[ -n "$target" ] || { usage >&2; exit 1; }

setup_python() {
    [ -x "$PY" ] && return 0
    echo "== creating .venv from requirements.txt"
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.12 .venv
        uv pip install --python "$PY" -r requirements.txt
    else
        command -v python3.12 >/dev/null 2>&1 || die "need uv or python3.12 to create .venv"
        python3.12 -m venv .venv
        .venv/bin/pip install -r requirements.txt
    fi
}

# fetch REPO REVISION DIR
fetch() {
    if [ "$skip_download" -eq 0 ]; then
        echo "== downloading $1 @ $2 into $3"
        if [ -n "$token" ]; then
            .venv/bin/hf download "$1" --revision "$2" --local-dir "$3" --token "$token"
        else
            .venv/bin/hf download "$1" --revision "$2" --local-dir "$3"
        fi
    else
        [ -d "$3" ] || die "--skip-download: $3 does not exist"
        echo "== using the existing snapshot in $3"
    fi
    echo "== verifying $3 against $2"
    "$PY" tools/verify_snapshot.py "$3" "$2"
}

# convert DIR NAME
convert() {
    mkdir -p gguf
    out="gguf/$2.gguf"; tmp="gguf/$2.gguf.tmp"
    echo "== converting $1 -> $out"
    rm -f "$tmp"
    "$PY" tools/convert.py "$1" "$tmp"
    echo "== verifying $out against $1"
    "$PY" tests/verify_gguf.py "$1" "$tmp"
    mv "$tmp" "$out"
    echo "== $out ready"
}

setup_python
case "$target" in
    clef-flash) fetch "$FLASH_REPO" "$FLASH_REV" model-flash; convert model-flash clef-flash ;;
    clef)       fetch "$CLEF_REPO" "$CLEF_REV" model;       convert model clef ;;
    all)        fetch "$FLASH_REPO" "$FLASH_REV" model-flash; convert model-flash clef-flash
                fetch "$CLEF_REPO" "$CLEF_REV" model;       convert model clef ;;
esac
echo "done. Build with: make"
