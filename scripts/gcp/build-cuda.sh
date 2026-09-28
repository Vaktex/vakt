#!/bin/sh
# Build the CUDA 13 release binaries on Google Cloud instead of GitHub Actions.
#
#   scripts/gcp/build-cuda.sh build amd64 v0.1.0        # C4 VM, x86 CUDA 13
#   scripts/gcp/build-cuda.sh build arm64 v0.1.0        # C4A Axion VM, native arm64
#   scripts/gcp/build-cuda.sh build amd64 v0.1.0 cpu    # the CPU (OpenBLAS) build
#   scripts/gcp/build-cuda.sh plan  amd64 v0.1.0        # print the plan, touch nothing
#
# Needs gcloud, signed in, and VAKT_GCP_PROJECT set.
#
# Why Compute Engine and not Cloud Build: hosted runners give the CUDA build 4 vCPUs, so MLX takes
# ~110 minutes per arch and a release costs ~220 Actions minutes -- paid again
# on every re-tag, because tag builds deliberately skip the native-deps cache.
# Cloud Build looked like the answer and is not, for two measured reasons:
# its workers are x86 ONLY (the default-pool enum is E2/N1, private pools are
# e2/n2d/c3), so arm64 would run under QEMU and be SLOWER than the runner it
# replaced; and this project's quota refuses E2_HIGHCPU_32 outright
# ("due to quota restrictions", needs a Google support request), leaving
# E2_HIGHCPU_8 -- only 2x a runner.
#
# Compute Engine has no such limit here (CPUS quota 200, 5 in use) and does have
# native Arm, so both arches build on a 16-vCPU VM -- c4-highcpu-16 for amd64,
# c4a-highcpu-16 (Axion) for arm64 -- created per build and deleted after.
#
# Measured: arm64 CUDA in ~14 min of compute for ~$0.09 on spot, against
# ~108 minutes of Actions time.
#
# Both paths run .github/scripts/linux-build.sh unchanged, inside the same
# pinned CUDA image, so the binary is built exactly as CI built it.
#
# The result lands in dist/ and is published with .github/scripts/publish-r2.sh.
set -eu

PROJECT=${VAKT_GCP_PROJECT:-}
# Zones, tried in order: C4/C4A capacity is per-zone and moves through the day,
# so a stockout in one zone must not fail a release.
ZONES=${VAKT_GCP_ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f}
# Same digest-pinned image as .github/workflows/build.yml.
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda@sha256:047cef4ac7248a01c715042dea75757268851abf1a0e6dcbd7648ff1180f1b89}
# The CPU build needs no CUDA toolkit, just the glibc floor install.sh enforces
# (2.35 = ubuntu 22.04). Digest-pinned like the CUDA image.
CPU_IMAGE=${CPU_IMAGE:-ubuntu@sha256:b8b6ee6aa931ecd9d0d952abc34dc0e5f7c6a30c6bb71b079fe399fde0329c02}
# 16 vCPUs on both arches. It fits CPUS_ALL_REGIONS (32 for this project, with
# ~5 held by long-running VMs), it is the largest C4A step below 32, and it is
# already ~7x the 4 vCPUs a hosted runner gives the build. Raise that quota and
# set VAKT_GCE_MACHINE_* to the -32/-64 types (C4A goes to 72) to go faster;
# JOBS follows the machine size automatically.
GCE_MACHINE_AMD64=${VAKT_GCE_MACHINE_AMD64:-c4-highcpu-16}
GCE_MACHINE_ARM64=${VAKT_GCE_MACHINE_ARM64:-c4a-highcpu-16}
GCE_IMAGE_PROJECT=${VAKT_GCE_IMAGE_PROJECT:-ubuntu-os-cloud}
# Spot first: about half the on-demand price, and a build is the ideal Spot
# workload -- it is minutes long, has no state, and can simply be run again.
# Falls back to on-demand when no zone has Spot capacity, so a release is never
# blocked by the discount (see the create loop). SPOT also implies
# --instance-termination-action DELETE, which is another way the VM cannot leak.
#
# (PREEMPTIBLE_CPUS is 0 in this project, but that quota governs the legacy
# preemptible model; Spot is charged against ordinary CPUS, which is 200.)
PROVISIONING=${VAKT_GCE_PROVISIONING:-SPOT}
# C4/C4A take Hyperdisk only, not pd-*. hyperdisk-balanced is the cheapest of
# the four available, at ~$0.09/GB-month.
DISK_TYPE=${VAKT_GCE_DISK_TYPE:-hyperdisk-balanced}
# 40 GB, measured: a full CUDA build uses 19 GB, 12 GB of which is the pinned
# image. The old 200 GB was a guess that billed ~5x the disk it used.
DISK_SIZE=${VAKT_GCE_DISK_SIZE:-40GB}
# A hard ceiling on what one build can cost. The VM is also deleted by the trap
# and by GCE itself at this deadline, so a hung build or a lost laptop cannot
# leave a 16-vCPU machine billing overnight. 90 min at spot rates is ~$0.50.
MAX_MINUTES=${VAKT_GCE_MAX_MINUTES:-90}

die() { printf 'build-cuda: %s\n' "$*" >&2; exit 1; }

sha256_of() {
	if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
	else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# stream_asset: pull the built binary over ssh into dist/.
stream_asset() {
	gcloud compute ssh "$vm" --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap \
		--command "cat /tmp/src/dist/$asset" >"dist/$asset"
}

# retry <n> <command...>: for gcloud calls over the IAP tunnel, which drops.
retry() {
	n=$1
	shift
	i=1
	while :; do
		if "$@" >/dev/null 2>&1; then return 0; fi
		[ "$i" -lt "$n" ] || return 1
		printf 'retrying (%d/%d): %s\n' "$i" "$n" "$1" >&2
		i=$((i + 1))
		sleep 5
	done
}

is_tag() { printf '%s\n' "$1" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$'; }

# resolve <arch> <version> [backend]: validate and set arch, backend, asset,
# engine, machine, image, jobs and vm.
resolve() {
	arch=${1:-}
	version=${2:-}
	backend=${3:-cuda}
	case $arch in
	amd64) engine=gce; machine=$GCE_MACHINE_AMD64 ;;
	arm64) engine=gce; machine=$GCE_MACHINE_ARM64 ;;
	*) die "arch must be amd64 or arm64 (got '${arch:-}')" ;;
	esac
	is_tag "$version" || die "not a release tag: '${version:-}' (expected vX.Y.Z[-pre])"
	case $arch in
	amd64) image_family=${VAKT_GCE_IMAGE_AMD64:-ubuntu-2204-lts} ;;
	arm64) image_family=${VAKT_GCE_IMAGE_ARM64:-ubuntu-2204-lts-arm64} ;;
	esac
	case $backend in
	cuda) image=$CUDA_IMAGE; suffix=cuda13 ;;
	cpu) image=$CPU_IMAGE; suffix=cpu ;;
	*) die "backend must be cuda or cpu (got '$backend')" ;;
	esac
	jobs=${machine##*-}
	asset="vakt-linux-$arch-$suffix"
	# A GCE instance name: lowercase, digits, hyphens, starts with a letter.
	vm="vakt-$(printf '%s' "$version" | tr 'A-Z._' 'a-z--' | tr -cd 'a-z0-9-')-$arch-$suffix"
}

plan() {
	resolve "$@"
	printf 'arch=%s version=%s backend=%s asset=%s engine=%s machine=%s vm=%s project=%s\n' \
		"$arch" "$version" "$backend" "$asset" "$engine" "$machine" "$vm" "$PROJECT"
}

# ---------------------------------------------------------------- arm64

# create_vm: one builder, in $ZONE with $model. Errors land in create.err so the
# caller can tell a stockout from a real failure.
create_vm() {
	gcloud compute instances create "$vm" \
		--project "$PROJECT" --zone "$ZONE" \
		--machine-type "$machine" \
		--image-family "$image_family" --image-project "$GCE_IMAGE_PROJECT" \
		--boot-disk-size "$DISK_SIZE" --boot-disk-type "$DISK_TYPE" \
		--provisioning-model "$model" \
		--max-run-duration="${MAX_MINUTES}m" --instance-termination-action=DELETE \
		--no-service-account --no-scopes \
		--metadata-from-file startup-script="$(dirname "$0")/startup.sh" \
		--metadata max-minutes="$MAX_MINUTES" \
		--labels purpose=vakt-release,arch="$arch",backend="$backend" \
		>/dev/null 2>"$workdir/create.err"
}

cleanup_vm() {
	for z in $ZONES; do
		gcloud compute instances delete "$vm" --project "$PROJECT" --zone "$z" \
			--quiet >/dev/null 2>&1 || true
	done
}

# Compute Engine: Cloud Build has no Arm worker, so build on a real Axion VM.
# Spot, deleted on every exit path including failure and Ctrl-C.
build_gce() {
	# Delete the builder on every exit path, including failure and Ctrl-C. Sweep
	# all candidate zones: a create that half-succeeded before a stockout, or a
	# zone we moved on from, must not leave a 16-vCPU VM billing.
	trap 'cleanup_vm' EXIT INT TERM
	# Every Spot zone first, then on-demand. Spot is ~half price and a build is
	# the ideal Spot workload, but Spot capacity is per-zone and comes and goes,
	# and a release must not be blocked by a discount being unavailable.
	created=0
	for model in $PROVISIONING STANDARD; do
		[ "$created" = 1 ] && break
		for ZONE in $ZONES; do
			printf 'creating %s (%s, %s) in %s\n' "$vm" "$machine" "$model" "$ZONE"
			if create_vm; then
				created=1
				break
			fi
			if grep -qE 'ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT|QUOTA_EXCEEDED' "$workdir/create.err"; then
				printf '  no %s %s capacity in %s\n' "$model" "$machine" "$ZONE"
				continue
			fi
			# A builder of this name from an earlier run, or one whose delete is
			# still in flight. The name encodes version+arch+backend, so it is
			# ours and disposable: remove it and try this zone once more.
			if grep -q 'already exists' "$workdir/create.err"; then
				printf '  removing a stale %s in %s\n' "$vm" "$ZONE"
				gcloud compute instances delete "$vm" --project "$PROJECT" --zone "$ZONE" \
					--quiet >/dev/null 2>&1 || true
				if create_vm; then
					created=1
					break
				fi
			fi
			cat "$workdir/create.err" >&2
			die "could not create $vm"
		done
	done
	[ "$created" = 1 ] || die "no zone had $machine capacity, spot or on-demand (tried: $ZONES)"

	printf 'waiting for docker on %s\n' "$vm"
	# The first SSH attempts fail while the VM boots and the startup script
	# installs docker, and the IAP tunnel itself times out occasionally. Both are
	# expected here: keep polling rather than failing the build (`set -e` would
	# otherwise abort on a transient gcloud crash).
	ready=0
	i=0
	while [ "$i" -lt 90 ]; do
		if gcloud compute ssh "$vm" --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap \
			--command 'test -f /var/lib/vakt-ready' >/dev/null 2>&1; then
			ready=1
			break
		fi
		sleep 10
		i=$((i + 1))
	done
	[ "$ready" = 1 ] || die "$vm did not become ready"

	# Ship the checkout rather than cloning: the VM has no credentials by design (--no-service-account --no-scopes).
	# Ship a CLEAN tree. Local build state must not travel: third_party/src is a
	# checkout MLX's build.sh clones into and refuses to overwrite, and shipping
	# local objects would mean the release binary was not built from source here.
	# (This is also what makes the upload small enough to matter over IAP.)
	tar -czf "$workdir/src.tgz" \
		--exclude .git --exclude .worktrees --exclude dist --exclude bin \
		--exclude .ccache --exclude third_party/src \
		--exclude 'third_party/*/build' --exclude 'third_party/*/install' \
		--exclude 'third_party/tokenizers/lib' \
		--exclude testdata/models --exclude demo \
		.
	retry 3 gcloud compute scp "$workdir/src.tgz" "$vm:/tmp/src.tgz" \
		--project "$PROJECT" --zone "$ZONE" --tunnel-through-iap

	gcloud compute ssh "$vm" --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap --command "
		set -eu
		mkdir -p /tmp/src && tar -C /tmp/src -xzf /tmp/src.tgz
		cd /tmp/src
		sudo docker run --rm -v /tmp/src:/src -w /src \
			-e VERSION='$version' -e COMMIT='$commit' -e JOBS='$jobs' -e CCACHE_DIR=/src/.ccache \
			'$image' bash .github/scripts/linux-build.sh $backend
		sha256sum dist/$asset | cut -d' ' -f1 > /tmp/asset.sha256
	"
	# Stream the binary over ssh rather than `gcloud compute scp`: measured,
	# scp over the IAP tunnel stalls partway through this ~240 MB file, while a
	# piped `cat` moves all of it in seconds. Keeps the VM credential-free
	# (no bucket, no signed URL, no service account).
	printf 'downloading %s\n' "$asset"
	want=$(gcloud compute ssh "$vm" --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap \
		--command 'cat /tmp/asset.sha256' 2>/dev/null | tr -d '\r\n ')
	retry 3 stream_asset
	got=$(sha256_of "dist/$asset")
	[ -n "$want" ] || die "the builder did not report a checksum for $asset"
	[ "$want" = "$got" ] || die "$asset arrived corrupt (expected $want, got $got)"
}

build() {
	resolve "$@"
	command -v gcloud >/dev/null 2>&1 || die "gcloud is not installed"
	[ -f .github/scripts/linux-build.sh ] || die "run this from the repository root"
	commit=$(git rev-parse --short=7 HEAD 2>/dev/null || echo none)
	mkdir -p dist
	workdir=$(mktemp -d "${TMPDIR:-/tmp}/vakt-gcp.XXXXXX")

	build_gce

	[ -s "dist/$asset" ] || die "no dist/$asset came back"
	chmod 0755 "dist/$asset"
	printf 'built dist/%s (%s)\n' "$asset" "$(wc -c <"dist/$asset" | tr -d ' ')"
}

case ${1:-} in
plan)
	[ $# -eq 3 ] || [ $# -eq 4 ] || die "usage: build-cuda.sh plan <amd64|arm64> <vX.Y.Z> [cuda|cpu]"
	plan "$2" "$3" "${4:-cuda}"
	;;
build)
	[ $# -eq 3 ] || [ $# -eq 4 ] || die "usage: build-cuda.sh build <amd64|arm64> <vX.Y.Z> [cuda|cpu]"
	[ -n "$PROJECT" ] || die "set VAKT_GCP_PROJECT to the Google Cloud project to build in"
	build "$2" "$3" "${4:-cuda}"
	;;
*) die "usage: build-cuda.sh plan|build <amd64|arm64> <vX.Y.Z> [cuda|cpu]" ;;
esac
