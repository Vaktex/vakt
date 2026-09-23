#!/usr/bin/env bash
# Runs inside the nvidia/cuda:13 devel container (see build.yml).
#   linux-build.sh cuda       native deps + obfuscated CUDA 13 release build + audit
#   linux-build.sh test-fake  unit tests without the native engine
set -euo pipefail

mode=${1:?usage: linux-build.sh cuda|test-fake}
: "${GARBLE_VERSION:=v0.18.0}"
GO_VERSION=$(awk '/^go /{print $2; exit}' go.mod)

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
	ca-certificates curl git build-essential cmake ninja-build ccache pkg-config \
	libopenblas-dev liblapack-dev liblapacke-dev >/dev/null

arch=$(dpkg --print-architecture)
tarball="go${GO_VERSION}.linux-${arch}.tar.gz"
# Pinned checksums (from go.dev); bump together with go.mod's go directive.
case "${GO_VERSION}-${arch}" in
1.27.1-amd64) want=63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445 ;;
1.27.1-arm64) want=3450b45a3f9ee8568792736a5c5e70a1f2e9b36c35a8f74958c03e51d7d92bec ;;
*) echo "no pinned checksum for go${GO_VERSION} ${arch}; add it to linux-build.sh" >&2; exit 1 ;;
esac
curl --proto '=https' --tlsv1.2 -fsSL "https://go.dev/dl/${tarball}" -o "/tmp/${tarball}"
echo "${want}  /tmp/${tarball}" | sha256sum -c -
tar -C /usr/local -xzf "/tmp/${tarball}"
export PATH=/usr/local/go/bin:/root/go/bin:/root/.cargo/bin:$PATH

git config --global --add safe.directory /src

case $mode in
cuda)
	# Pinned, checksum-verified rustup-init and toolchain (see docs/BUILD.md).
	case $arch in
	amd64) triple=x86_64-unknown-linux-gnu; rsum=dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71 ;;
	arm64) triple=aarch64-unknown-linux-gnu; rsum=15f6e4ce9f583b929c996c91562bad6d4454f3281de858b02cdfdef615fac433 ;;
	esac
	curl --proto '=https' --tlsv1.2 -fsSL "https://static.rust-lang.org/rustup/archive/1.29.1/${triple}/rustup-init" -o /tmp/rustup-init
	echo "${rsum}  /tmp/rustup-init" | sha256sum -c -
	chmod +x /tmp/rustup-init
	/tmp/rustup-init -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN:-1.98.1}" >/dev/null
	go install "mvdan.cc/garble@${GARBLE_VERSION}"
	make deps BACKEND=cuda
	make prod BACKEND=cuda VERSION="${VERSION:-dev}" COMMIT="${COMMIT:-none}"
	make audit BACKEND=cuda
	# Files created in the container are root-owned; hand them back.
	chown -R "$(stat -c %u:%g /src)" dist third_party .ccache 2>/dev/null || true
	;;
test-fake)
	make test BACKEND=fake
	;;
*)
	echo "unknown mode: $mode" >&2
	exit 2
	;;
esac
