#!/usr/bin/env bats
# scripts/gcp/build-cuda.sh argument and naming logic. The parts that talk to
# Google Cloud are exercised by running a real build; these are the pure
# decisions that must not be wrong when a release depends on them.

SCRIPT="$BATS_TEST_DIRNAME/../scripts/gcp/build-cuda.sh"

run_plan() { run sh "$SCRIPT" plan "$@"; }

@test "amd64 plans a 16-vCPU x86 C4 VM" {
	run_plan amd64 v1.2.3
	[ "$status" -eq 0 ]
	[[ "$output" == *"arch=amd64"* ]] || false
	[[ "$output" == *"asset=vakt-linux-amd64-cuda13"* ]] || false
	[[ "$output" == *"engine=gce"* ]] || false
	[[ "$output" == *"machine=c4-highcpu-16"* ]] || false
}

@test "arm64 plans a native Axion VM, never QEMU on Cloud Build" {
	run_plan arm64 v1.2.3
	[ "$status" -eq 0 ]
	[[ "$output" == *"asset=vakt-linux-arm64-cuda13"* ]] || false
	[[ "$output" == *"engine=gce"* ]] || false
	[[ "$output" == *"machine=c4a-highcpu-16"* ]] || false
	# Cloud Build has no Arm workers, and this project's quota refuses its
	# 32-vCPU x86 type, so neither arch goes through Cloud Build.
	[[ "$output" != *"cloudbuild"* ]] || false
}

@test "an unknown architecture is refused" {
	run_plan riscv64 v1.2.3
	[ "$status" -ne 0 ]
	[[ "$output" == *"arch must be amd64 or arm64"* ]] || false
}

# The CPU builds are the same shape as CUDA: same script, same spot VM, a
# different backend and a smaller base image.

@test "a cpu build names the cpu asset and the ubuntu base" {
	run sh "$SCRIPT" plan amd64 v1.2.3 cpu
	[ "$status" -eq 0 ]
	[[ "$output" == *"asset=vakt-linux-amd64-cpu"* ]] || false
	[[ "$output" == *"backend=cpu"* ]] || false
	run sh "$SCRIPT" plan arm64 v1.2.3 cpu
	[ "$status" -eq 0 ]
	[[ "$output" == *"asset=vakt-linux-arm64-cpu"* ]] || false
}

@test "the backend defaults to cuda, so existing calls are unchanged" {
	run_plan amd64 v1.2.3
	[ "$status" -eq 0 ]
	[[ "$output" == *"backend=cuda"* ]] || false
	[[ "$output" == *"asset=vakt-linux-amd64-cuda13"* ]] || false
}

@test "an unknown backend is refused" {
	run sh "$SCRIPT" plan amd64 v1.2.3 metal
	[ "$status" -ne 0 ]
	[[ "$output" == *"backend must be cuda or cpu"* ]] || false
}

@test "cpu and cuda builders get distinct VM names so they can run at once" {
	cuda=$(sh "$SCRIPT" plan amd64 v1.2.3 | sed -n 's/.*vm=\([^ ]*\).*/\1/p')
	cpu=$(sh "$SCRIPT" plan amd64 v1.2.3 cpu | sed -n 's/.*vm=\([^ ]*\).*/\1/p')
	[ -n "$cuda" ] && [ -n "$cpu" ]
	[ "$cuda" != "$cpu" ] || false
	printf '%s' "$cpu" | grep -Eq '^[a-z][a-z0-9-]{0,61}$' || false
}

@test "a version that is not a release tag is refused" {
	for bad in 1.2.3 v1.2 latest 'v1.2.3; rm -rf /' '../../evil' ''; do
		run_plan amd64 "$bad"
		[ "$status" -ne 0 ]
		[[ "$output" == *"not a release tag"* ]] || false
	done
}

@test "a pre-release tag is accepted" {
	run_plan amd64 v1.2.3-rc.1
	[ "$status" -eq 0 ]
	[[ "$output" == *"version=v1.2.3-rc.1"* ]] || false
}

@test "the VM name is derived from the tag and is a legal GCE name" {
	run_plan arm64 v1.2.3-rc.1
	[ "$status" -eq 0 ]
	name=$(printf '%s\n' "$output" | sed -n 's/.*vm=\([^ ]*\).*/\1/p')
	[ -n "$name" ]
	# GCE: lowercase letters, digits and hyphens; must start with a letter.
	printf '%s' "$name" | grep -Eq '^[a-z][a-z0-9-]{0,61}$' || false
}

@test "spot is the default provisioning model, with an on-demand fallback" {
	# Cost: spot is ~half price, and a build is the ideal spot workload. The
	# fallback exists so an unavailable discount cannot block a release.
	grep -q 'PROVISIONING=${VAKT_GCE_PROVISIONING:-SPOT}' "$SCRIPT" || false
	grep -q 'for model in \$PROVISIONING STANDARD' "$SCRIPT" || false
	# A reclaimed spot builder must delete itself, not linger as a stopped VM.
	grep -q 'instance-termination-action=DELETE' "$SCRIPT" || false
}

@test "the boot disk is sized to what a build measured, not a round guess" {
	# A full CUDA build uses 19 GB; 200 GB billed ~5x the disk it used.
	grep -q 'DISK_SIZE=${VAKT_GCE_DISK_SIZE:-40GB}' "$SCRIPT" || false
	! grep -q 'boot-disk-size 200GB' "$SCRIPT" || false
}

@test "a build has a hard cost ceiling that does not depend on this script" {
	# GCE deletes the VM at the deadline even if the script dies or the laptop
	# sleeps, so a hung build cannot bill overnight.
	grep -q 'MAX_MINUTES=${VAKT_GCE_MAX_MINUTES:-90}' "$SCRIPT" || false
	grep -q -- '--max-run-duration="${MAX_MINUTES}m"' "$SCRIPT" || false
	grep -q -- '--instance-termination-action=DELETE' "$SCRIPT" || false
}

@test "the builder VM is deleted on every exit path" {
	grep -q "trap 'cleanup_vm' EXIT INT TERM" "$SCRIPT" || false
	# Swept across zones: a VM left in a zone we moved on from must not bill.
	grep -q 'for z in \$ZONES' "$SCRIPT" || false
}

@test "the builder gets no credentials" {
	# The bucket is written from the developer's own session; the VM never needs, and never gets, a service account.
	grep -q -- '--no-service-account --no-scopes' "$SCRIPT" || false
}

@test "usage is refused without both arguments" {
	run sh "$SCRIPT" plan amd64
	[ "$status" -ne 0 ]
	run sh "$SCRIPT"
	[ "$status" -ne 0 ]
}
