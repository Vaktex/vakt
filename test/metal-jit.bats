#!/usr/bin/env bats
# Exercise the pinned upstream generator without downloading/building MLX.

setup() {
	root="$BATS_TEST_DIRNAME/.."
	work="$BATS_TEST_TMPDIR/jit"
	mkdir -p "$work/src/mlx/backend/metal/kernels" "$work/bin" "$work/out"
	[ -d "$root/third_party/src/mlx/.git" ] || skip "requires pinned MLX checkout (make deps)"
	git -C "$root/third_party/src/mlx" show HEAD:mlx/backend/metal/make_compiled_preamble.sh > "$work/generate.sh"
	patch="$root/third_party/mlx/patches/mlx-jit-fail-fast.patch"
	if [ -f "$patch" ]; then
		mkdir -p "$work/patch/mlx/backend/metal"
		cp "$work/generate.sh" "$work/patch/mlx/backend/metal/make_compiled_preamble.sh"
		git -C "$work/patch" apply "$patch"
		cp "$work/patch/mlx/backend/metal/make_compiled_preamble.sh" "$work/generate.sh"
	fi
	printf '// shader\n' > "$work/src/mlx/backend/metal/kernels/probe.h"
	export PATH="$work/bin:$PATH"
}

@test "Metal preprocessing errors fail generation instead of embedding diagnostics" {
	printf '#!/bin/sh\nprintf "fatal error: MetalPerformancePrimitives.h not found\\n" >&2\nexit 1\n' > "$work/bin/xcrun"
	chmod +x "$work/bin/xcrun"
	run bash "$work/generate.sh" "$work/out" unused "$work/src" probe
	[ "$status" -ne 0 ]
	[[ "$output" == *"MetalPerformancePrimitives.h not found"* ]]
	[ ! -f "$work/out/probe.cpp" ]
}

@test "Successful preprocessing with no local includes still generates source" {
	printf '#!/bin/sh\nexit 0\n' > "$work/bin/xcrun"
	chmod +x "$work/bin/xcrun"
	run bash "$work/generate.sh" "$work/out" unused "$work/src" probe
	[ "$status" -eq 0 ]
	[ -f "$work/out/probe.cpp" ]
}
