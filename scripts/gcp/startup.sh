#!/bin/sh
# Startup script for the ephemeral release builder VM (C4 amd64 or C4A arm64).
#
# Installs Docker only: the CUDA toolchain, Go, Rust and cmake all come from
# the pinned CUDA image through .github/scripts/linux-build.sh, so this VM never
# decides what the build is made of.
#
# build-cuda.sh polls for /var/lib/vakt-ready before using the machine.
set -eu

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends docker.io >/dev/null
systemctl enable --now docker

# Last line: the readiness flag, so a half-provisioned VM is never used.
touch /var/lib/vakt-ready
