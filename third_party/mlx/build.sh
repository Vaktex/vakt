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

fetch() { # dir repo tag commit (empty tag: fetch the commit itself)
	local dir="$1" repo="$2" tag="$3" commit="$4"
	if [[ -d "$dir/.git" && -n "$(git -C "$dir" rev-parse -q --verify "$commit^{commit}" 2>/dev/null)" && "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]]; then
		git -C "$dir" -c advice.detachedHead=false checkout --quiet --force "$commit"
	fi
	if [[ -d "$dir/.git" && -z "$tag" && "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]]; then
		git -C "$dir" fetch --quiet --depth 1 origin "$commit"
		git -C "$dir" -c advice.detachedHead=false checkout --quiet --force FETCH_HEAD
	fi
	if [[ ! -d "$dir/.git" ]]; then
		if [[ -n "$tag" ]]; then
			git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$tag" "$repo" "$dir"
		else
			git init --quiet "$dir"
			git -C "$dir" remote add origin "$repo"
			git -C "$dir" fetch --quiet --depth 1 origin "$commit"
			git -C "$dir" -c advice.detachedHead=false checkout --quiet FETCH_HEAD
		fi
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
# A checkout at the right commit can still carry stray local edits: reset it
# (our patches are re-applied below).
for d in "$src/mlx" "$src/mlx-c"; do
	git -C "$d" reset --quiet --hard HEAD
	git -C "$d" clean --quiet -fdx
done

# Dependencies MLX would otherwise download at configure time. They are
# fetched here, verified against pinned hashes, and handed to CMake with
# FETCHCONTENT_FULLY_DISCONNECTED so configure makes no network requests.
# nlohmann/json parses the (untrusted) safetensors header inside MLX.
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
fetch_pinned() { # name url sha256 -> $deps/<name> (extracted)
	local name="$1" url="$2" want="$3" out="$deps/$1"
	[[ -f "$out/.vakt-sha256" && "$(cat "$out/.vakt-sha256")" == "$want" ]] && return 0
	rm -rf "$out" && mkdir -p "$out"
	local arc="$deps/$name.download"
	curl --proto '=https' --tlsv1.2 -fsSL "$url" -o "$arc"
	local got
	got="$(sha256 "$arc")"
	if [[ "$got" != "$want" ]]; then
		echo "error: $name checksum mismatch (got $got, want $want)" >&2
		rm -f "$arc"
		exit 1
	fi
	case "$url" in
	*.zip) (cd "$out" && unzip -q "$arc") ;;
	*) tar -C "$out" -xf "$arc" ;;
	esac
	# Archives wrap everything in one top-level directory; flatten it.
	local inner
	inner="$(find "$out" -mindepth 1 -maxdepth 1 -type d)"
	if [[ "$(printf '%s\n' "$inner" | wc -l)" -eq 1 && -n "$inner" ]]; then
		(shopt -s dotglob && mv "$inner"/* "$out"/ && rmdir "$inner")
	fi
	rm -f "$arc"
	echo "$want" > "$out/.vakt-sha256"
}
deps="$src/deps"
mkdir -p "$deps"
fetch_pinned json https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz \
	d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
fetch "$deps/fmt" https://github.com/fmtlib/fmt.git 12.1.0 407c905e45ad75fc29bf0f9bb7c5c2fd3475976f
dep_opts=(
	-DFETCHCONTENT_FULLY_DISCONNECTED=ON
	-DFETCHCONTENT_SOURCE_DIR_JSON="$deps/json"
	-DFETCHCONTENT_SOURCE_DIR_FMT="$deps/fmt"
)
if [[ "$backend" == cuda ]]; then
	fetch_pinned cccl https://github.com/NVIDIA/cccl/releases/download/v3.1.3/cccl-v3.1.3.zip \
		30f388ef784eb691d7de9d2cf918d53ab33464672500a17945b99d16af136cd6
	fetch "$deps/nvtx3" https://github.com/NVIDIA/NVTX.git v3.1.1 6230bdf710bc94f44d433acceba735aaa9090ba5
	fetch "$deps/cudnn" https://github.com/NVIDIA/cudnn-frontend.git v1.16.0 be6c079be8aaffa0fc079fcf039887e637c289c7
	fetch "$deps/cutlass" https://github.com/NVIDIA/cutlass.git v4.3.5 4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b
	dep_opts+=(
		-DFETCHCONTENT_SOURCE_DIR_CCCL="$deps/cccl"
		-DFETCHCONTENT_SOURCE_DIR_NVTX3="$deps/nvtx3"
		-DFETCHCONTENT_SOURCE_DIR_CUDNN="$deps/cudnn"
		-DFETCHCONTENT_SOURCE_DIR_CUTLASS="$deps/cutlass"
	)
fi
if [[ "$backend" == metal ]]; then
	fetch_pinned metal_cpp https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip \
		4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4
	dep_opts+=(-DFETCHCONTENT_SOURCE_DIR_METAL_CPP="$deps/metal_cpp")
fi

# Local patches (idempotent).
for p in "$here"/patches/mlx-*.patch; do
	[[ -e "$p" ]] || continue
	if git -C "$src/mlx" apply --check "$p" 2>/dev/null; then
		git -C "$src/mlx" apply "$p"
		echo "applied $(basename "$p")"
	elif git -C "$src/mlx" apply --reverse --check "$p" 2>/dev/null; then
		: # already applied
	else
		echo "error: patch $(basename "$p") does not apply to mlx ${MLX_TAG:-$MLX_COMMIT}" >&2
		exit 1
	fi
done

# mlx-c patches: vakt's C entry points for MLX APIs mlx-c does not wrap yet.
for p in "$here"/patches/mlxc-*.patch; do
	[[ -e "$p" ]] || continue
	if git -C "$src/mlx-c" apply --check "$p" 2>/dev/null; then
		git -C "$src/mlx-c" apply "$p"
		echo "applied $(basename "$p")"
	elif git -C "$src/mlx-c" apply --reverse --check "$p" 2>/dev/null; then
		: # already applied
	else
		echo "error: patch $(basename "$p") does not apply to mlx-c ${MLXC_TAG:-$MLXC_COMMIT}" >&2
		exit 1
	fi
done

[[ "${MLX_CLEAN:-0}" == 1 ]] && rm -rf "$build"
mkdir -p "$build" "$prefix"

launcher=()
if command -v ccache >/dev/null 2>&1; then
	launcher=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
fi

# Strip absolute source paths from debug info, __FILE__ and error strings so the
# release binary contains no /Users/... or /home/... paths.
pfx="-ffile-prefix-map=$root=. -ffile-prefix-map=$src=third_party/src"
opts=(
	-DCMAKE_BUILD_TYPE="$build_type"
	-DCMAKE_C_FLAGS="${CFLAGS:-} $pfx"
	-DCMAKE_CXX_FLAGS="${CXXFLAGS:-} $pfx"
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
	"${dep_opts[@]}"
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
		# MLX reads its own MLX_CUDA_ARCHITECTURES and otherwise probes the
		# local GPU (fails on GPU-less CI). Turing through Blackwell (CUDA 13
		# dropped anything older).
		-DMLX_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75;80;86;89;90;100;120}"
		-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75;80;86;89;90;100;120}")
	[[ -n "${BLA_STATIC:-}" ]] && opts+=(-DBLA_STATIC="$BLA_STATIC")
	;;
esac

# 1) MLX
cmake -S "$src/mlx" -B "$build/mlx" "${opts[@]}"
cmake --build "$build/mlx" -j "$jobs"
cmake --install "$build/mlx"

# 2) mlx-c against the installed MLX
mlxc_opts=()
if [[ "$backend" == cuda ]]; then
	# MLX's exported config links CUDA::cublasLt and CUDNN::cudnn_all but
	# calls neither find_package(CUDAToolkit) nor its own installed
	# FindCUDNN.cmake, so mlx-c's configure fails without them.
	inc="$build/mlxc-cuda-deps.cmake"
	printf '%s\n' \
		'find_package(CUDAToolkit REQUIRED)' \
		"list(APPEND CMAKE_MODULE_PATH \"$prefix/share/cmake/MLX\")" \
		'find_package(CUDNN REQUIRED)' > "$inc"
	mlxc_opts+=(-DCMAKE_PROJECT_INCLUDE="$inc")
fi
cmake -S "$src/mlx-c" -B "$build/mlx-c" "${opts[@]}" ${mlxc_opts[@]+"${mlxc_opts[@]}"} -DMLX_DIR="$prefix/share/cmake/MLX"
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
