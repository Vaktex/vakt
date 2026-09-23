#!/bin/sh
# Build Linux release binaries locally with Docker buildx.
#   scripts/docker/build-linux.sh [cuda|cpu|all] [amd64|arm64|all]
set -eu

backends=${1:-all}
arches=${2:-all}
[ "$backends" = all ] && backends="cuda cpu"
[ "$arches" = all ] && arches="amd64 arm64"

root=$(cd "$(dirname "$0")/../.." && pwd)
version=$(git -C "$root" describe --tags --always --dirty 2>/dev/null || echo dev)
commit=$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo none)

for backend in $backends; do
	for arch in $arches; do
		echo "==> linux/$arch $backend"
		docker buildx build \
			-f "$root/scripts/docker/Dockerfile.linux" \
			--platform "linux/$arch" \
			--build-arg "BACKEND=$backend" \
			--build-arg "VERSION=$version" \
			--build-arg "COMMIT=$commit" \
			--target out \
			--output "type=local,dest=$root/dist" \
			"$root"
	done
done
ls -lh "$root/dist"
