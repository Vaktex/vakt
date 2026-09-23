#!/usr/bin/env bash
# Build static MLX + mlx-c for the vakt native engine.
#
#   third_party/mlx/build.sh metal   # darwin/arm64, Metal GPU + Accelerate CPU
#   third_party/mlx/build.sh cpu     # CPU only (Accelerate on mac, OpenBLAS on linux)
#   third_party/mlx/build.sh cuda    # linux, CUDA GPU + CPU
#
# Output: third_party/mlx/install/<goos>_<goarch>_<backend>/{lib,include}
# The metal build also produces lib/mlx.metallib, which is embedded into the Go
# binary (internal/engine/mlx/metallib) so that vakt needs no external files.
#
# Env: JOBS (default: all cores), MLX_CLEAN=1 to wipe the build dir,
#      CMAKE_BUILD_TYPE (default Release), MACOSX_DEPLOYMENT_TARGET (default 14.0).
set -euo pipefail

backend="${1:-}"
case "$backend" in
metal | cpu | cuda) ;;
*)
	echo "usage: $0 metal|cpu|cuda" >&2
	exit 2
	;;
esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
while IFS='=' read -r k v; do
	[[ "$k" =~ ^[A-Z_]+$ ]] && printf -v "$k" '%s' "$v"
done <"$here/VERSIONS"

case "$(uname -s)" in
Darwin) goos=darwin ;;
Linux) goos=linux ;;
*)
	echo "unsupported OS $(uname -s)" >&2
	exit 2
	;;
esac
case "$(uname -m)" in
arm64 | aarch64) goarch=arm64 ;;
x86_64 | amd64) goarch=amd64 ;;
*)
	echo "unsupported arch $(uname -m)" >&2
	exit 2
	;;
esac

if [[ "$backend" == metal && "$goos" != darwin ]]; then
	echo "metal backend requires macOS" >&2
	exit 2
fi
if [[ "$backend" == cuda && "$goos" != linux ]]; then
	echo "cuda backend requires linux" >&2
	exit 2
fi

target="${goos}_${goarch}_${backend}"
src="$root/third_party/src"
build="$root/third_party/mlx/build/$target"
prefix="$here/install/$target"
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)}"
build_type="${CMAKE_BUILD_TYPE:-Release}"

fetch() { # dir repo tag commit
	local dir="$1" repo="$2" tag="$3" commit="$4"
	if [[ ! -d "$dir/.git" ]]; then
		git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$tag" "$repo" "$dir"
	fi
	local have
	have="$(git -C "$dir" rev-parse HEAD)"
	if [[ "$have" != "$commit" ]]; then
		echo "error: $dir is at $have, expected $commit ($tag). Remove it and rerun." >&2
		exit 1
	fi
}

mkdir -p "$src"
fetch "$src/mlx" "$MLX_REPO" "$MLX_TAG" "$MLX_COMMIT"
fetch "$src/mlx-c" "$MLXC_REPO" "$MLXC_TAG" "$MLXC_COMMIT"

# Local patches (idempotent).
for p in "$here"/patches/mlx-*.patch; do
	[[ -e "$p" ]] || continue
	if git -C "$src/mlx" apply --check "$p" 2>/dev/null; then
		git -C "$src/mlx" apply "$p"
		echo "applied $(basename "$p")"
	elif git -C "$src/mlx" apply --reverse --check "$p" 2>/dev/null; then
		: # already applied
	else
		echo "error: patch $(basename "$p") does not apply to mlx $MLX_TAG" >&2
		exit 1
	fi
done

[[ "${MLX_CLEAN:-0}" == 1 ]] && rm -rf "$build"
mkdir -p "$build" "$prefix"

launcher=()
if command -v ccache >/dev/null 2>&1; then
	launcher=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
fi

opts=(
	-DCMAKE_BUILD_TYPE="$build_type"
	-DCMAKE_INSTALL_PREFIX="$prefix"
	-DCMAKE_POSITION_INDEPENDENT_CODE=ON
	-DBUILD_SHARED_LIBS=OFF
	-DMLX_BUILD_TESTS=OFF
	-DMLX_BUILD_EXAMPLES=OFF
	-DMLX_BUILD_BENCHMARKS=OFF
	-DMLX_BUILD_PYTHON_BINDINGS=OFF
	-DMLX_BUILD_PYTHON_STUBS=OFF
	-DMLX_BUILD_GGUF=OFF
	-DMLX_BUILD_SAFETENSORS=ON
	-DMLX_C_BUILD_EXAMPLES=OFF
	-DMLX_C_USE_SYSTEM_MLX=ON
	${launcher[@]+"${launcher[@]}"}
)

case "$backend" in
metal)
	# JIT keeps mlx.metallib small (~a dozen precompiled kernel families; the
	# rest is compiled from embedded source on first use) so it can be
	# embedded in the Go binary. A non-JIT metallib is ~180 MB.
	opts+=(-DMLX_BUILD_METAL=ON -DMLX_METAL_JIT=ON -DMLX_BUILD_CPU=ON
		-DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}")
	if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
		echo "error: the Metal toolchain is missing. Run: xcodebuild -downloadComponent MetalToolchain" >&2
		exit 1
	fi
	;;
cpu)
	opts+=(-DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_CPU=ON)
	[[ "$goos" == darwin ]] && opts+=(-DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}")
	# On linux MLX finds BLAS/LAPACK via CMake (install libopenblas-dev
	# liblapacke-dev). Set BLA_STATIC=ON to prefer static archives.
	[[ -n "${BLA_STATIC:-}" ]] && opts+=(-DBLA_STATIC="$BLA_STATIC")
	;;
cuda)
	opts+=(-DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=ON -DMLX_BUILD_CPU=ON
		-DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-75;80;86;89;90;100;120}")
	[[ -n "${BLA_STATIC:-}" ]] && opts+=(-DBLA_STATIC="$BLA_STATIC")
	;;
esac

# 1) MLX
cmake -S "$src/mlx" -B "$build/mlx" "${opts[@]}"
cmake --build "$build/mlx" -j "$jobs"
cmake --install "$build/mlx"

# 2) mlx-c against the installed MLX
cmake -S "$src/mlx-c" -B "$build/mlx-c" "${opts[@]}" -DMLX_DIR="$prefix/share/cmake/MLX"
cmake --build "$build/mlx-c" -j "$jobs"
cmake --install "$build/mlx-c"

# Record what was built (read by the Go package's link-line generator / CI).
{
	echo "MLX_TAG=$MLX_TAG"
	echo "MLXC_TAG=$MLXC_TAG"
	echo "BACKEND=$backend"
	echo "TARGET=$target"
} >"$prefix/BUILDINFO"

if [[ "$backend" == metal ]]; then
	lib="$prefix/lib/mlx.metallib"
	[[ -f "$lib" ]] || {
		echo "error: $lib missing" >&2
		exit 1
	}
	# The Go package embeds this copy (go:embed cannot reach outside the module
	# package dir). It is gitignored.
	emb="$root/internal/engine/mlx/metallib"
	mkdir -p "$emb"
	cp "$lib" "$emb/mlx.metallib"
	echo "metallib: $(du -h "$lib" | cut -f1) -> $emb/mlx.metallib"
fi

echo "installed $target -> $prefix"
