#!/usr/bin/env bats

load helpers

setup() {
	unset STUB_NVIDIA STUB_LDCONFIG STUB_KERNEL STUB_MACHINE STUB_TRANSLATED STUB_GLIBC STUB_LDD
	mkdir -p "$BATS_TEST_TMPDIR/home"
}

# ---------------------------------------------------------------- macOS

@test "macOS arm64 installs vakt-darwin-arm64" {
	export STUB_KERNEL=Darwin STUB_MACHINE=arm64
	setup_stubs
	publish vakt-darwin-arm64
	run_installer --version v1.0.0
	[ "$status" -eq 0 ]
	[[ "$output" == *"asset: vakt-darwin-arm64"* ]]
	[[ "$output" == *"sha256 verified"* ]]
	[ -x "$PREFIX/bin/vakt" ]
	grep -q '^xattr -d com.apple.quarantine' "$CALLS"
	grep -q "doctor" <("$PREFIX/bin/vakt" doctor)
}

@test "native Intel Mac is rejected" {
	export STUB_KERNEL=Darwin STUB_MACHINE=x86_64 STUB_TRANSLATED=0
	setup_stubs
	run_installer --version v1.0.0
	[ "$status" -ne 0 ]
	[[ "$output" == *"Apple Silicon"* ]]
	[ ! -e "$PREFIX/bin/vakt" ]
}

@test "Rosetta shell on Apple Silicon installs arm64" {
	export STUB_KERNEL=Darwin STUB_MACHINE=x86_64 STUB_TRANSLATED=1
	setup_stubs
	publish vakt-darwin-arm64
	run_installer --version v1.0.0
	[ "$status" -eq 0 ]
	[[ "$output" == *"Rosetta"* ]]
	[ -x "$PREFIX/bin/vakt" ]
}

# ---------------------------------------------------------------- Linux GPU

@test "Ubuntu amd64 with driver 580 and CUDA 13 libs installs cuda13" {
	export STUB_NVIDIA="580.65.06, NVIDIA H100 80GB HBM3" STUB_LDCONFIG="$CUDA_LIBS_ALL"
	setup_stubs
	os_release ubuntu 24.04
	publish vakt-linux-amd64-cuda13
	run_installer --version v1.0.0
	[ "$status" -eq 0 ]
	[[ "$output" == *"asset: vakt-linux-amd64-cuda13"* ]]
	[[ "$output" == *"CUDA 13 runtime libraries present"* ]]
	[ -x "$PREFIX/bin/vakt" ]
}

@test "driver 580 without libs and --yes installs runtime via apt with sudo" {
	export STUB_NVIDIA="580.65.06, NVIDIA L4" STUB_APT_PROVIDES="$CUDA_LIBS_ALL"
	setup_stubs
	stub_ldconfig_dynamic
	os_release ubuntu 24.04
	publish vakt-linux-amd64-cuda13
	cp "$ASSETS/vakt-bin" "$ASSETS/cuda-keyring_1.1-1_all.deb"
	run_installer --version v1.0.0 --yes
	[ "$status" -eq 0 ]
	grep -q '^sudo dpkg -i' "$CALLS"
	grep -q '^sudo apt-get install -y cuda-libraries-13-0 libcudnn9-cuda-13' "$CALLS"
	grep -q 'repos/ubuntu2404/x86_64/cuda-keyring' "$CALLS"
	[[ "$output" == *"asset: vakt-linux-amd64-cuda13"* ]]
}

@test "driver 580 without libs and --no-deps falls back to cpu" {
	export STUB_NVIDIA="580.65.06, NVIDIA L4"
	setup_stubs
	os_release ubuntu 24.04
	publish vakt-linux-amd64-cpu
	run_installer --version v1.0.0 --no-deps
	[ "$status" -eq 0 ]
	[[ "$output" == *"asset: vakt-linux-amd64-cpu"* ]]
	! grep -q '^sudo' "$CALLS"
}

@test "without a tty and without --yes, no packages are installed" {
	export STUB_NVIDIA="580.65.06, NVIDIA L4"
	setup_stubs
	os_release ubuntu 24.04
	publish vakt-linux-amd64-cpu
	run_installer --version v1.0.0
	[ "$status" -eq 0 ]
	! grep -q '^sudo' "$CALLS"
	[[ "$output" == *"asset: vakt-linux-amd64-cpu"* ]]
}

@test "driver 550 falls back to cpu with an explanation" {
	export STUB_NVIDIA="550.54.14, NVIDIA A100" STUB_LDCONFIG="$CUDA_LIBS_ALL"
	setup_stubs
	os_release ubuntu 22.04
	publish vakt-linux-amd64-cpu
	run_installer --version v1.0.0
	[ "$status" -eq 0 ]
	[[ "$output" == *">= 580"* ]]
	[[ "$output" == *"asset: vakt-linux-amd64-cpu"* ]]
}

@test "Fedora arm64 without a GPU installs cpu; RHEL-family repo not touched" {
	export STUB_MACHINE=aarch64
	setup_stubs
	os_release fedora 42
	publish vakt-linux-arm64-cpu
	run_installer --version v1.0.0
	[ "$status" -eq 0 ]
	[[ "$output" == *"asset: vakt-linux-arm64-cpu"* ]]
	! grep -q '^dnf' "$CALLS"
}

@test "Rocky 9 arm64 builds the sbsa dnf repo command" {
	export STUB_MACHINE=aarch64 STUB_NVIDIA="580.65.06, NVIDIA GH200"
	setup_stubs
	os_release rocky 9.4
	publish vakt-linux-arm64-cpu
	run_installer --version v1.0.0 --dry-run
	[ "$status" -eq 0 ]
	[[ "$output" == *"repos/rhel9/sbsa/cuda-rhel9.repo"* ]]
}

@test "--cpu forces the cpu build even with a GPU" {
	export STUB_NVIDIA="580.65.06, NVIDIA H100" STUB_LDCONFIG="$CUDA_LIBS_ALL"
	setup_stubs
	publish vakt-linux-amd64-cpu
	run_installer --version v1.0.0 --cpu
	[ "$status" -eq 0 ]
	[[ "$output" == *"asset: vakt-linux-amd64-cpu"* ]]
}

# ---------------------------------------------------------------- libc

@test "musl is rejected" {
	export STUB_GLIBC="" STUB_LDD="musl libc (x86_64) Version 1.2.4"
	setup_stubs
	run_installer --version v1.0.0
	[ "$status" -ne 0 ]
	[[ "$output" == *"musl"* ]]
}

@test "glibc 2.31 is rejected" {
	export STUB_GLIBC=2.31
	setup_stubs
	run_installer --version v1.0.0
	[ "$status" -ne 0 ]
	[[ "$output" == *"glibc 2.31 is too old"* ]]
}

# ---------------------------------------------------------------- integrity

@test "checksum mismatch fails and installs nothing" {
	setup_stubs
	publish vakt-linux-amd64-cpu
	echo tampered >>"$ASSETS/vakt-linux-amd64-cpu"
	run_installer --version v1.0.0
	[ "$status" -ne 0 ]
	[[ "$output" == *"checksum mismatch"* ]]
	[ ! -e "$PREFIX/bin/vakt" ]
}

@test "missing SHA256SUMS entry fails" {
	setup_stubs
	cp "$ASSETS/vakt-bin" "$ASSETS/vakt-linux-amd64-cpu"
	echo "0000000000000000000000000000000000000000000000000000000000000000  vakt-darwin-arm64" >"$ASSETS/SHA256SUMS"
	run_installer --version v1.0.0
	[ "$status" -ne 0 ]
	[[ "$output" == *"no entry for vakt-linux-amd64-cpu"* ]]
	[ ! -e "$PREFIX/bin/vakt" ]
}

@test "every curl call is https-only" {
	setup_stubs
	publish vakt-linux-amd64-cpu
	run_installer
	[ "$status" -eq 0 ]
	[[ "$output" == *"v9.9.9"* ]]
	n=$(grep -c '^curl' "$CALLS")
	[ "$n" -ge 2 ]
	[ "$(grep '^curl' "$CALLS" | grep -vc -- '--proto =https')" -eq 0 ]
}

@test "invalid --version is rejected" {
	setup_stubs
	run_installer --version '../../evil'
	[ "$status" -ne 0 ]
	[[ "$output" == *"invalid version"* ]]
}

# ---------------------------------------------------------------- modes

@test "--local installs from a directory into a temp prefix" {
	setup_stubs
	local_dir="$BATS_TEST_TMPDIR/local"
	mkdir -p "$local_dir"
	cp "$ASSETS/vakt-bin" "$local_dir/vakt-linux-amd64-cpu"
	(cd "$local_dir" && shasum -a 256 vakt-linux-amd64-cpu >SHA256SUMS)
	run_installer --local "$local_dir"
	[ "$status" -eq 0 ]
	[ -x "$PREFIX/bin/vakt" ]
	! grep -q '^curl' "$CALLS"
}

@test "--uninstall removes the binary" {
	setup_stubs
	mkdir -p "$PREFIX/bin"
	cp "$ASSETS/vakt-bin" "$PREFIX/bin/vakt"
	run_installer --uninstall
	[ "$status" -eq 0 ]
	[ ! -e "$PREFIX/bin/vakt" ]
}

@test "--dry-run changes nothing" {
	export STUB_NVIDIA="580.65.06, NVIDIA L4"
	setup_stubs
	os_release ubuntu 24.04
	run_installer --version v1.0.0 --dry-run --yes
	[ "$status" -eq 0 ]
	[ ! -e "$PREFIX" ]
	! grep -qE '^(sudo|dpkg|apt-get|curl)' "$CALLS"
	[[ "$output" == *"+ fetch https://github.com/vaktex/vakt/releases/download/v1.0.0/vakt-linux-amd64-cuda13"* ]]
}

@test "PATH advice is printed when prefix/bin is not on PATH" {
	setup_stubs
	publish vakt-linux-amd64-cpu
	run_installer --version v1.0.0
	[ "$status" -eq 0 ]
	[[ "$output" == *"is not on your PATH"* ]]
}
