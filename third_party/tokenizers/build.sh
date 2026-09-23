#!/usr/bin/env bash
# Build libtokenizers.a (HuggingFace tokenizers via github.com/daulet/tokenizers)
# from source and install it to third_party/tokenizers/lib/<goos>_<goarch>/.
#
# The tag below MUST match the github.com/daulet/tokenizers version in go.mod:
# the Go binding contains a link-time version check against the Rust crate.
#
# Usage: third_party/tokenizers/build.sh [--force]
# Env:   CARGO           cargo binary (default: cargo on PATH, else ~/.cargo/bin/cargo)
#        INSTALL_RUST=1  install a minimal rustup toolchain if cargo is missing
#        SRC_DIR         checkout location (default: third_party/tokenizers/build/src)
set -euo pipefail

TAG="v1.27.0"
# Commit the tag pointed to when it was pinned; guards against a moved tag.
COMMIT="f678a7768d5479d9d5a5161c4fc45c8a5ba46146"
REPO="https://github.com/daulet/tokenizers"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SRC_DIR:-$HERE/build/src}"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

case "$(uname -s)" in
  Darwin) GOOS=darwin ;;
  Linux)  GOOS=linux ;;
  *) echo "build.sh: unsupported OS $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  arm64|aarch64) GOARCH=arm64 ;;
  x86_64|amd64)  GOARCH=amd64 ;;
  *) echo "build.sh: unsupported arch $(uname -m)" >&2; exit 1 ;;
esac
if [[ "$GOOS" == darwin && "$GOARCH" != arm64 ]]; then
  echo "build.sh: only darwin/arm64 is supported on macOS" >&2; exit 1
fi

OUT_DIR="$HERE/lib/${GOOS}_${GOARCH}"
STAMP="$OUT_DIR/VERSION"
if [[ $FORCE -eq 0 && -f "$OUT_DIR/libtokenizers.a" && -f "$STAMP" && "$(cat "$STAMP")" == "$TAG $COMMIT" ]]; then
  echo "libtokenizers.a $TAG already built at $OUT_DIR"
  exit 0
fi

# Locate cargo.
CARGO="${CARGO:-}"
if [[ -z "$CARGO" ]]; then
  if command -v cargo >/dev/null 2>&1; then CARGO="$(command -v cargo)"
  elif [[ -x "$HOME/.cargo/bin/cargo" ]]; then CARGO="$HOME/.cargo/bin/cargo"
  fi
fi
if [[ -z "$CARGO" ]]; then
  if [[ "${INSTALL_RUST:-0}" == 1 ]]; then
    command -v curl >/dev/null || { echo "build.sh: curl is required to install rust" >&2; exit 1; }
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    CARGO="$HOME/.cargo/bin/cargo"
  else
    echo "build.sh: cargo not found; install rust (https://rustup.rs) or set INSTALL_RUST=1" >&2
    exit 1
  fi
fi
# onig_sys compiles C; make sure a compiler exists before a long cargo run.
command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || command -v clang >/dev/null 2>&1 || {
  echo "build.sh: a C compiler (cc/gcc/clang) is required" >&2; exit 1; }
command -v git >/dev/null || { echo "build.sh: git is required" >&2; exit 1; }

# Fetch the pinned source.
if [[ ! -d "$SRC_DIR/.git" ]]; then
  rm -rf "$SRC_DIR"
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone --quiet --depth 1 --branch "$TAG" "$REPO" "$SRC_DIR"
else
  git -C "$SRC_DIR" fetch --quiet --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
  git -C "$SRC_DIR" checkout --quiet --force "$TAG"
fi
got="$(git -C "$SRC_DIR" rev-parse HEAD)"
if [[ "$got" != "$COMMIT" ]]; then
  echo "build.sh: tag $TAG resolved to $got, expected $COMMIT" >&2; exit 1
fi

# Build. --locked uses the upstream Cargo.lock so the Rust tokenizers crate
# version (0.22.x) is exactly the one this tag was released with.
if [[ "$GOOS" == darwin ]]; then
  # Match the Go toolchain's minimum macOS so the linker does not warn.
  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"
fi
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$SRC_DIR/target}"
# Keep build-machine paths (panic locations) out of the archive, even when this
# script runs outside `make deps` (which exports its own RUSTFLAGS remaps).
# RUSTFLAGS is part of the cargo fingerprint, so a change rebuilds.
if [[ "${RUSTFLAGS:-}" != *remap-path-prefix* ]]; then
  # rustc applies the LAST matching remap, so the broad $HOME one goes first.
  CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
  export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$HOME=~ \
--remap-path-prefix=$SRC_DIR=tokenizers \
--remap-path-prefix=$CARGO_HOME_DIR/registry/src=crates \
--remap-path-prefix=$CARGO_HOME_DIR/git/checkouts=crates-git"
fi
(cd "$SRC_DIR" && "$CARGO" build --release --locked -p tokenizers-ffi)

mkdir -p "$OUT_DIR"
cp "$CARGO_TARGET_DIR/release/libtokenizers_ffi.a" "$OUT_DIR/libtokenizers.a"
echo "$TAG $COMMIT" > "$STAMP"
echo "installed $OUT_DIR/libtokenizers.a ($TAG)"
