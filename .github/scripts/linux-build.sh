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
curl --proto '=https' --tlsv1.2 -fsSL "https://go.dev/dl/${tarball}" -o "/tmp/${tarball}"
# Verify against go.dev's published checksum.
want=$(curl --proto '=https' --tlsv1.2 -fsSL "https://go.dev/dl/?mode=json&include=all" |
	python3 -c "import json,sys; f=sys.argv[1]; print(next(x['sha256'] for r in json.load(sys.stdin) for x in r['files'] if x['filename']==f))" "$tarball")
echo "${want}  /tmp/${tarball}" | sha256sum -c -
tar -C /usr/local -xzf "/tmp/${tarball}"
export PATH=/usr/local/go/bin:/root/go/bin:/root/.cargo/bin:$PATH

git config --global --add safe.directory /src

case $mode in
cuda)
	curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/rustup.sh
	sh /tmp/rustup.sh -y --profile minimal >/dev/null
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
