#!/bin/sh
# Vaktex OSS installer: installs the `vakt` binary for this machine.
#
#   curl -fsSL https://github.com/vaktex/vakt/releases/latest/download/install.sh | sh
#   curl -fsSL .../install.sh | sh -s -- --version v0.1.0 --yes
#   sh install.sh --help
#
# Picks the right build (macOS Apple Silicon/Metal, Linux CUDA 13 or CPU),
# verifies its SHA-256 against the release SHA256SUMS before installing, and
# never executes anything it downloaded except the verified binary.
#
# The whole script is functions; `main "$@"` on the last line means a
# truncated download cannot run a partial script.

set -eu

VAKT_REPO=${VAKT_REPO:-vaktex/vakt}
VAKT_VERSION=${VAKT_VERSION:-latest}
VAKT_OS_RELEASE=${VAKT_OS_RELEASE:-/etc/os-release}
MIN_DRIVER=580
MIN_GLIBC=2.35
CUDA_LIBS="libcublas.so.13 libcublasLt.so.13 libnvrtc.so.13 libcudnn.so.9"

# Options.
opt_prefix=""
opt_system=0
opt_local=""
opt_yes=0
opt_no_deps=0
opt_cpu=0
opt_force_gpu=0
opt_dry_run=0
opt_uninstall=0
opt_modify_path=0
opt_doctor=1
opt_summon=ask

# State.
os=""
arch=""
asset=""
workdir=""
tmp_target=""
bin_dir=""
sudo_cmd=""

# ---------------------------------------------------------------- output

setup_colors() {
	if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
		c_bold=$(printf '\033[1m'); c_dim=$(printf '\033[2m'); c_red=$(printf '\033[31m')
		c_green=$(printf '\033[32m'); c_yellow=$(printf '\033[33m'); c_cyan=$(printf '\033[36m')
		c_reset=$(printf '\033[0m')
	else
		c_bold=""; c_dim=""; c_red=""; c_green=""; c_yellow=""; c_cyan=""; c_reset=""
	fi
}

say()  { printf '%s\n' "$*"; }
info() { printf '%s==>%s %s\n' "$c_cyan" "$c_reset" "$*"; }
ok()   { printf '%s ok%s  %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%swarn%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
die()  { printf '%serror%s %s\n' "$c_red" "$c_reset" "$*" >&2; exit 1; }

header() {
	printf '%sVaktex OSS installer%s %s(vakt)%s\n\n' "$c_bold" "$c_reset" "$c_dim" "$c_reset"
}

usage() {
	cat <<'EOF'
Vaktex OSS installer: install the `vakt` code-security scanner.

Usage: install.sh [options]
       curl -fsSL <url>/install.sh | sh -s -- [options]

Options:
  --version <tag>     Release to install (default: latest; env VAKT_VERSION)
  --prefix <dir>      Install to <dir>/bin (default: $HOME/.local)
  --system            Install to /usr/local/bin (uses sudo if needed)
  --local <dir>       Use the binary and SHA256SUMS from <dir> instead of downloading
  --cpu               Linux: install the CPU build even if an NVIDIA GPU is present
  --force-gpu         Linux: install the CUDA 13 build even if checks fail
  --yes, -y           Answer yes to prompts (CUDA runtime install, summon)
  --no-deps           Never install system packages
  --modify-path       Append the PATH line to your shell rc file if needed
  --no-doctor         Skip `vakt doctor` after installing
  --summon            Download the model after installing, without asking
  --no-summon         Do not offer to download the model
  --uninstall         Remove vakt (and optionally its model cache)
  --dry-run           Print what would change without changing anything
  -h, --help          Show this help

Environment:
  VAKT_REPO           GitHub owner/repo for releases (default: vaktex/vakt)
  VAKT_VERSION        Same as --version
  HF_TOKEN            Hugging Face token for the private vaktex/DOM-0.8B model
  NO_COLOR            Disable colour
EOF
}

# ---------------------------------------------------------------- helpers

have() { command -v "$1" >/dev/null 2>&1; }

# run: execute a state-changing command, or print it under --dry-run.
run() {
	if [ "$opt_dry_run" = 1 ]; then
		printf '+ %s\n' "$*"
		return 0
	fi
	"$@"
}

# ask <question>: yes/no prompt read from the terminal. Honours --yes.
# Returns 1 when there is no terminal to ask.
ask() {
	if [ "$opt_yes" = 1 ]; then return 0; fi
	if [ "$opt_dry_run" = 1 ]; then printf '%s [y/N] (dry-run: no)\n' "$1"; return 1; fi
	if ! { true </dev/tty; } 2>/dev/null; then return 1; fi
	printf '%s [y/N] ' "$1" >/dev/tty
	read -r reply </dev/tty || return 1
	case $reply in [yY]|[yY][eE][sS]) return 0 ;; *) return 1 ;; esac
}

# version_ge a b: true when dotted version a >= b (numeric components).
version_ge() {
	awk -v a="$1" -v b="$2" 'BEGIN {
		na = split(a, x, "."); nb = split(b, y, ".")
		n = (na > nb) ? na : nb
		for (i = 1; i <= n; i++) {
			xi = (i <= na) ? x[i] + 0 : 0; yi = (i <= nb) ? y[i] + 0 : 0
			if (xi > yi) exit 0
			if (xi < yi) exit 1
		}
		exit 0
	}'
}

# fetch <url> <dest>: HTTPS-only download.
fetch() {
	if have curl; then
		curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$2" "$1"
	elif have wget; then
		wget --https-only -q -O "$2" "$1"
	else
		die "need curl or wget to download"
	fi
}

sha256_of() {
	if have sha256sum; then
		sha256sum "$1" | awk '{print $1}'
	elif have shasum; then
		shasum -a 256 "$1" | awk '{print $1}'
	else
		die "need sha256sum or shasum to verify the download"
	fi
}

cleanup() {
	if [ -n "$workdir" ] && [ -d "$workdir" ]; then rm -rf "$workdir"; fi
	# A half-installed temp file in the target bin dir (set in install_binary).
	# shellcheck disable=SC2086
	if [ -n "${tmp_target:-}" ] && [ -e "$tmp_target" ]; then $sudo_cmd rm -f "$tmp_target" 2>/dev/null || true; fi
}

# ---------------------------------------------------------------- arguments

parse_args() {
	while [ $# -gt 0 ]; do
		case $1 in
		--version) [ $# -ge 2 ] || die "--version needs a value"; VAKT_VERSION=$2; shift ;;
		--version=*) VAKT_VERSION=${1#*=} ;;
		--prefix) [ $# -ge 2 ] || die "--prefix needs a value"; opt_prefix=$2; shift ;;
		--prefix=*) opt_prefix=${1#*=} ;;
		--local) [ $# -ge 2 ] || die "--local needs a directory"; opt_local=$2; shift ;;
		--local=*) opt_local=${1#*=} ;;
		--system) opt_system=1 ;;
		--cpu) opt_cpu=1 ;;
		--force-gpu) opt_force_gpu=1 ;;
		--yes|-y) opt_yes=1 ;;
		--no-deps) opt_no_deps=1 ;;
		--modify-path) opt_modify_path=1 ;;
		--no-doctor) opt_doctor=0 ;;
		--summon) opt_summon=yes ;;
		--no-summon) opt_summon=no ;;
		--uninstall) opt_uninstall=1 ;;
		--dry-run) opt_dry_run=1 ;;
		-h|--help) usage; exit 0 ;;
		*) die "unknown option: $1 (see --help)" ;;
		esac
		shift
	done

	validate_version "$VAKT_VERSION"
	case $VAKT_REPO in
	*/*/*|*[!A-Za-z0-9._/-]*|*..*|/*|*/|.*|-*|*/.*|*/-*) die "invalid VAKT_REPO '$VAKT_REPO' (expected owner/repo)" ;;
	*/*) ;;
	*) die "invalid VAKT_REPO '$VAKT_REPO' (expected owner/repo)" ;;
	esac

	if [ "$opt_system" = 1 ]; then
		opt_prefix=/usr/local
	elif [ -z "$opt_prefix" ]; then
		opt_prefix=${HOME:?HOME is not set}/.local
	fi
	# The prefix ends up in PATH advice and, with --modify-path, in a shell rc
	# file, so it must be a plain absolute path: no quotes, $, backticks, ;.
	case $opt_prefix in
	/*) ;;
	*) die "--prefix must be an absolute path" ;;
	esac
	case $opt_prefix in
	*[!A-Za-z0-9._/+@-]*) die "--prefix may only contain letters, digits and ._/+@- (got '$opt_prefix')" ;;
	esac
	bin_dir=${opt_prefix%/}/bin
}

# validate_version: "latest" or a tag made of safe characters only. The tag is
# placed in a URL path, so '/' and '..' must never get through.
validate_version() {
	case $1 in
	latest) return 0 ;;
	v[0-9]*) ;;
	*) die "invalid version '$1' (expected latest or vX.Y.Z)" ;;
	esac
	case $1 in
	*[!A-Za-z0-9.+-]*|*..*) die "invalid version '$1' (expected latest or vX.Y.Z)" ;;
	esac
}

# ---------------------------------------------------------------- detection

detect_platform() {
	kernel=$(uname -s)
	machine=$(uname -m)
	case $kernel in
	Darwin)
		os=darwin
		case $machine in
		arm64) arch=arm64 ;;
		x86_64)
			if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
				info "Rosetta shell detected on Apple Silicon; installing the native arm64 build"
				arch=arm64
			else
				die "Vaktex OSS needs an Apple Silicon Mac (M1 or newer): the model runs on the GPU through Metal/MLX. Intel Macs are not supported."
			fi
			;;
		*) die "unsupported Mac architecture: $machine" ;;
		esac
		asset="vakt-darwin-arm64"
		ok "macOS on Apple Silicon (Metal)"
		;;
	Linux)
		os=linux
		case $machine in
		x86_64|amd64) arch=amd64 ;;
		aarch64|arm64) arch=arm64 ;;
		*) die "unsupported Linux architecture: $machine (need x86_64 or aarch64)" ;;
		esac
		check_glibc
		if [ "$opt_cpu" = 1 ]; then
			asset=vakt-linux-$arch-cpu
			ok "Linux $arch, CPU build requested"
		elif detect_gpu; then
			asset=vakt-linux-$arch-cuda13
		else
			asset=vakt-linux-$arch-cpu
		fi
		;;
	*) die "unsupported operating system: $kernel (macOS and Linux only)" ;;
	esac
}

check_glibc() {
	ver=""
	# musl first: its ldd says so, and getconf may not know GNU_LIBC_VERSION.
	if have ldd; then
		case $(ldd --version 2>&1 | head -n 1) in
		*musl*) die "musl libc (e.g. Alpine) is not supported; use a glibc-based distribution" ;;
		esac
	fi
	if have getconf; then
		ver=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}') || ver=""
	fi
	if [ -z "$ver" ] && have ldd; then
		first=$(ldd --version 2>&1 | head -n 1) || first=""
		case $first in
		*musl*) die "musl libc (e.g. Alpine) is not supported; use a glibc-based distribution" ;;
		esac
		ver=$(printf '%s\n' "$first" | awk '{print $NF}')
	fi
	case $ver in
	[0-9]*.[0-9]*) ;;
	*) die "could not determine the glibc version; vakt needs glibc >= $MIN_GLIBC (musl is not supported)" ;;
	esac
	version_ge "$ver" "$MIN_GLIBC" || die "glibc $ver is too old; vakt needs glibc >= $MIN_GLIBC (Ubuntu 22.04+, Debian 12+, RHEL 9+)"
	ok "glibc $ver"
}

# detect_gpu: 0 when the CUDA 13 build should be installed.
detect_gpu() {
	if ! have nvidia-smi; then
		info "no NVIDIA GPU detected; installing the CPU build"
		return 1
	fi
	query=$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null) || query=""
	if [ -z "$query" ]; then
		warn "nvidia-smi is present but reported no GPU; installing the CPU build"
		return 1
	fi
	driver=$(printf '%s\n' "$query" | head -n 1 | cut -d, -f1 | tr -d ' ')
	gpus=$(printf '%s\n' "$query" | cut -d, -f2- | sed 's/^ *//' | paste -sd ';' -)
	ok "NVIDIA GPU: $gpus (driver $driver)"

	if ! version_ge "$driver" "$MIN_DRIVER"; then
		if [ "$opt_force_gpu" = 1 ]; then
			warn "driver $driver < $MIN_DRIVER; installing the CUDA 13 build anyway (--force-gpu)"
			return 0
		fi
		warn "CUDA 13 needs NVIDIA driver >= $MIN_DRIVER (found $driver). Installing the CPU build; update the driver and re-run for GPU support."
		return 1
	fi

	missing=$(missing_cuda_libs)
	if [ -z "$missing" ]; then
		ok "CUDA 13 runtime libraries present"
		return 0
	fi
	warn "missing CUDA 13 runtime libraries: $missing"
	if [ "$opt_force_gpu" = 1 ]; then return 0; fi
	if install_cuda_runtime && [ -z "$(missing_cuda_libs)" ]; then
		ok "CUDA 13 runtime libraries installed"
		return 0
	fi
	if [ "$opt_dry_run" = 1 ]; then return 0; fi
	warn "continuing with the CPU build; install the libraries above and re-run for GPU support"
	return 1
}

missing_cuda_libs() {
	cache=$(ldconfig -p 2>/dev/null || true)
	out=""
	for lib in $CUDA_LIBS; do
		case $cache in *"$lib "*|*"$lib") continue ;; esac
		found=0
		for dir in /usr/local/cuda*/lib64 /usr/local/cuda*/targets/*/lib /usr/lib/*-linux-gnu /usr/lib64; do
			if [ -e "$dir/$lib" ]; then found=1; break; fi
		done
		[ "$found" = 1 ] || out="$out $lib"
	done
	printf '%s' "${out# }"
}

# install_cuda_runtime: offer to install CUDA 13 runtime packages from
# NVIDIA's repository. Returns 0 only if packages were installed.
install_cuda_runtime() {
	if [ "$opt_no_deps" = 1 ]; then
		info "--no-deps: not installing system packages"
		return 1
	fi
	[ -r "$VAKT_OS_RELEASE" ] || { warn "cannot read $VAKT_OS_RELEASE; install the CUDA 13 runtime manually"; return 1; }
	id=$(os_release_field ID)
	version_id=$(os_release_field VERSION_ID)
	id_like=$(os_release_field ID_LIKE)
	case $arch in amd64) repo_arch=x86_64 ;; arm64) repo_arch=sbsa ;; esac
	base=https://developer.download.nvidia.com/compute/cuda/repos
	pkgs="cuda-libraries-13-0 libcudnn9-cuda-13"

	case $id in
	ubuntu|debian)
		case $id$version_id in
		ubuntu22.04) distro=ubuntu2204 ;;
		ubuntu24.04) distro=ubuntu2404 ;;
		debian12) distro=debian12 ;;
		*) warn "no NVIDIA CUDA repository for $id $version_id; install the CUDA 13 runtime manually"; return 1 ;;
		esac
		family=apt
		;;
	rhel|rocky|almalinux|centos|fedora)
		case $id in
		fedora) distro=fedora$(printf '%s' "$version_id" | cut -d. -f1) ;;
		*) distro=rhel$(printf '%s' "$version_id" | cut -d. -f1) ;;
		esac
		family=dnf
		;;
	*)
		case $id_like in
		*rhel*|*fedora*) distro=rhel$(printf '%s' "$version_id" | cut -d. -f1); family=dnf ;;
		*) warn "unsupported distribution '$id'; install the CUDA 13 runtime manually"; return 1 ;;
		esac
		;;
	esac
	# distro goes into a URL path: only letters followed by a numeric version.
	case $distro in
	ubuntu[0-9]*|debian[0-9]*|rhel[0-9]*|fedora[0-9]*) ;;
	*) warn "unrecognised distribution version; install the CUDA 13 runtime manually"; return 1 ;;
	esac
	case $distro in
	*[!a-z0-9]*) warn "unrecognised distribution version; install the CUDA 13 runtime manually"; return 1 ;;
	esac

	say ""
	say "  The CUDA 13 runtime can be installed from NVIDIA's repository with:"
	if [ "$family" = apt ]; then
		say "    curl -fsSLO $base/$distro/$repo_arch/cuda-keyring_1.1-1_all.deb"
		say "    sudo dpkg -i cuda-keyring_1.1-1_all.deb"
		say "    sudo apt-get update"
		say "    sudo apt-get install -y $pkgs"
	else
		say "    sudo dnf config-manager --add-repo $base/$distro/$repo_arch/cuda-$distro.repo"
		say "    sudo dnf install -y $pkgs"
	fi
	say ""

	ask "Install CUDA 13 runtime libraries now with sudo?" || return 1
	setup_sudo || return 1

	if [ "$family" = apt ]; then
		[ -n "$workdir" ] || die "internal error: work directory not set"
		keyring=$workdir/cuda-keyring_1.1-1_all.deb
		want=$(keyring_sha256 "$distro" "$repo_arch")
		if [ -z "$want" ]; then
			warn "no pinned checksum for the $distro/$repo_arch cuda-keyring; run the commands above manually"
			return 1
		fi
		if [ "$opt_dry_run" = 1 ]; then
			printf '+ fetch %s (sha256 %s)\n' "$base/$distro/$repo_arch/cuda-keyring_1.1-1_all.deb" "$want"
		else
			fetch "$base/$distro/$repo_arch/cuda-keyring_1.1-1_all.deb" "$keyring" || return 1
			got=$(sha256_of "$keyring")
			if [ "$got" != "$want" ]; then
				warn "cuda-keyring checksum mismatch (expected $want, got $got); not installing it"
				return 1
			fi
		fi
		# shellcheck disable=SC2086 # sudo_cmd is intentionally empty or a single word
		run $sudo_cmd dpkg -i "$keyring" || return 1
		# shellcheck disable=SC2086
		run $sudo_cmd apt-get update || return 1
		# shellcheck disable=SC2086
		run $sudo_cmd apt-get install -y $pkgs || return 1
	else
		# shellcheck disable=SC2086
		run $sudo_cmd dnf config-manager --add-repo "$base/$distro/$repo_arch/cuda-$distro.repo" || return 1
		# shellcheck disable=SC2086
		run $sudo_cmd dnf install -y $pkgs || return 1
	fi
	return 0
}

# keyring_sha256 <distro> <arch>: pinned sha256 of NVIDIA's cuda-keyring
# 1.1-1 package, which installs as root and adds an apt signing key.
keyring_sha256() {
	# Test hook only: lets the bats suite exercise the install path with a
	# stub package. Whoever sets it already controls this shell.
	if [ -n "${VAKT_TEST_KEYRING_SHA256:-}" ] && [ "${VAKT_TEST_MODE:-}" = 1 ]; then
		warn "VAKT_TEST_MODE: using a test cuda-keyring checksum"
		echo "$VAKT_TEST_KEYRING_SHA256"
		return 0
	fi
	case $1/$2 in
	ubuntu2204/x86_64) echo d93190d50b98ad4699ff40f4f7af50f16a76dac3bb8da1eaaf366d47898ff8df ;;
	ubuntu2204/sbsa) echo 36d1aed84dfcf93ee9a0212d149f1c4187db92e03ab96a759ec0689aa438fd9e ;;
	ubuntu2404/x86_64) echo d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba ;;
	ubuntu2404/sbsa) echo 6ea7d2737648936820e85677177957a0f6521b840d98eb0bbae0a4f003fa7249 ;;
	debian12/x86_64) echo e7f219eab6fe4819cdb5c15b98233dc3420302d9c00883219cd3d896857cf48d ;;
	debian12/sbsa) echo bf2f53b1a19259501119f732bdef2699f8c80c69a18e6a78dad78d67d96766dc ;;
	*) echo "" ;;
	esac
}

os_release_field() {
	# Parse KEY=value / KEY="value" without sourcing the file.
	sed -n "s/^$1=//p" "$VAKT_OS_RELEASE" | head -n 1 | tr -d '"'"'"
}

setup_sudo() {
	if [ "$(id -u)" = 0 ]; then sudo_cmd=""; return 0; fi
	if have sudo; then sudo_cmd=sudo; return 0; fi
	warn "sudo is not available; run as root or install the packages manually"
	return 1
}

# ---------------------------------------------------------------- install

resolve_version() {
	if [ "$VAKT_VERSION" != latest ] || [ -n "$opt_local" ]; then return 0; fi
	if [ "$opt_dry_run" = 1 ]; then return 0; fi
	url=$(curl --proto '=https' --tlsv1.2 -fsSLI -o /dev/null -w '%{url_effective}' \
		"https://github.com/$VAKT_REPO/releases/latest" 2>/dev/null) || url=""
	tag=${url##*/}
	case $tag in
	v[0-9]*) validate_version "$tag"; VAKT_VERSION=$tag ;;
	*) die "could not resolve the latest release of $VAKT_REPO; pass --version vX.Y.Z" ;;
	esac
}

download_and_verify() {
	if [ -n "$opt_local" ]; then
		[ -f "$opt_local/$asset" ] || die "$opt_local/$asset not found"
		[ -f "$opt_local/SHA256SUMS" ] || die "$opt_local/SHA256SUMS not found"
		cp "$opt_local/$asset" "$workdir/$asset"
		cp "$opt_local/SHA256SUMS" "$workdir/SHA256SUMS"
		info "using local build from $opt_local"
	else
		base="https://github.com/$VAKT_REPO/releases/download/$VAKT_VERSION"
		if [ "$opt_dry_run" = 1 ]; then
			printf '+ fetch %s/%s\n+ fetch %s/SHA256SUMS\n' "$base" "$asset" "$base"
			return 0
		fi
		info "downloading $asset ($VAKT_VERSION)"
		fetch "$base/SHA256SUMS" "$workdir/SHA256SUMS" || die "could not download SHA256SUMS for $VAKT_VERSION"
		fetch "$base/$asset" "$workdir/$asset" || die "could not download $asset for $VAKT_VERSION"
	fi

	entries=$(awk -v f="$asset" '$2 == f || $2 == "*" f {print $1}' "$workdir/SHA256SUMS")
	[ "$(printf '%s\n' "$entries" | grep -c .)" -le 1 ] || die "SHA256SUMS has more than one entry for $asset; refusing to install"
	expected=$entries
	case $expected in
	'') die "SHA256SUMS has no entry for $asset; refusing to install" ;;
	*[!0-9a-f]*) die "malformed checksum for $asset in SHA256SUMS; refusing to install" ;;
	esac
	[ ${#expected} -eq 64 ] || die "malformed checksum for $asset in SHA256SUMS; refusing to install"
	actual=$(sha256_of "$workdir/$asset")
	if [ "$actual" != "$expected" ]; then
		die "checksum mismatch for $asset (expected $expected, got $actual); refusing to install"
	fi
	ok "sha256 verified ($expected)"
}

install_binary() {
	target=$bin_dir/vakt
	if [ "$opt_system" = 1 ] && [ ! -w "$opt_prefix" ] && [ "$opt_dry_run" = 0 ]; then
		ask "Install to $bin_dir with sudo?" || die "not installing to $bin_dir without permission"
		setup_sudo || exit 1
	fi
	if [ "$opt_dry_run" = 1 ]; then
		printf '+ install -m 0755 <verified %s> %s\n' "$asset" "$target"
	else
		# shellcheck disable=SC2086
		$sudo_cmd mkdir -p "$bin_dir"
		# Unpredictable temp name in the target dir, then an atomic rename.
		# shellcheck disable=SC2086
		tmp_target=$($sudo_cmd mktemp "$bin_dir/.vakt.XXXXXX")
		# shellcheck disable=SC2086
		$sudo_cmd cp "$workdir/$asset" "$tmp_target"
		# shellcheck disable=SC2086
		$sudo_cmd chmod 0755 "$tmp_target"
		# shellcheck disable=SC2086
		$sudo_cmd mv -f "$tmp_target" "$target"
		tmp_target=""
	fi
	if [ "$os" = darwin ]; then
		# shellcheck disable=SC2086
		run $sudo_cmd xattr -d com.apple.quarantine "$target" 2>/dev/null || true
		if [ "$opt_dry_run" = 0 ] && have codesign && ! codesign -v "$target" 2>/dev/null; then
			warn "code signature did not verify for $target"
		fi
	fi
	if [ "$opt_dry_run" = 1 ]; then ok "would install $target"; else ok "installed $target"; fi
}

path_advice() {
	case ":${PATH:-}:" in *":$bin_dir:"*) return 0 ;; esac
	# Defence in depth: bin_dir is validated in parse_args, but it is about
	# to be written into a file the user's shell executes.
	case $bin_dir in *[!A-Za-z0-9._/+@-]*) die "refusing to write an unsafe PATH entry" ;; esac
	shell_name=$(basename "${SHELL:-sh}")
	case $shell_name in
	zsh) rc=$HOME/.zshrc; line="export PATH=\"$bin_dir:\$PATH\"" ;;
	bash)
		if [ "$os" = darwin ]; then rc=$HOME/.bash_profile; else rc=$HOME/.bashrc; fi
		line="export PATH=\"$bin_dir:\$PATH\""
		;;
	fish) rc=$HOME/.config/fish/config.fish; line="fish_add_path $bin_dir" ;;
	*) rc=$HOME/.profile; line="export PATH=\"$bin_dir:\$PATH\"" ;;
	esac
	if [ "$opt_modify_path" = 1 ]; then
		if [ -f "$rc" ] && grep -qF "$line" "$rc"; then return 0; fi
		if [ "$opt_dry_run" = 1 ]; then
			printf '+ append to %s: %s\n' "$rc" "$line"
		else
			mkdir -p "$(dirname "$rc")"
			printf '\n# Added by the Vaktex OSS installer\n%s\n' "$line" >>"$rc"
			ok "added $bin_dir to PATH in $rc (open a new shell)"
		fi
	else
		warn "$bin_dir is not on your PATH. Add it with:"
		say "    echo '$line' >> $rc"
	fi
}

post_install() {
	vakt=$bin_dir/vakt
	if [ "$opt_dry_run" = 1 ]; then
		[ "$opt_doctor" = 1 ] && printf '+ %s doctor\n' "$vakt"
		return 0
	fi
	if [ "$opt_doctor" = 1 ]; then
		say ""
		"$vakt" doctor || warn "vakt doctor reported problems (see above)"
	fi

	token_file=${HF_HOME:-$HOME/.cache/huggingface}/token
	if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$token_file" ]; then
		say ""
		warn "no Hugging Face token found. The DOM-0.8B model (vaktex/DOM-0.8B) is private:"
		say "    export HF_TOKEN=hf_...        # or: hf auth login"
	fi

	case $opt_summon in
	no) return 0 ;;
	yes) do_summon=1 ;;
	*) if ask "Download the DOM-0.8B model now (~1.7 GB)?"; then do_summon=1; else do_summon=0; fi ;;
	esac
	if [ "$do_summon" = 1 ]; then
		"$vakt" summon || warn "model download failed; run \`vakt summon\` later"
	else
		say "Run \`vakt summon\` to download the model, or it will download on first \`vakt patrol\`."
	fi
}

uninstall() {
	target=$bin_dir/vakt
	if [ -e "$target" ]; then
		if [ ! -w "$bin_dir" ] && [ "$opt_dry_run" = 0 ]; then
			ask "Remove $target with sudo?" || die "not removing $target without permission"
			setup_sudo || exit 1
		fi
		# shellcheck disable=SC2086
		run $sudo_cmd rm -f "$target"
		ok "removed $target"
	else
		info "vakt is not installed in $bin_dir"
	fi
	if [ "$(uname -s)" = Darwin ]; then
		cache=${VAKT_CACHE:-$HOME/Library/Caches/vakt}
	else
		cache=${VAKT_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/vakt}
	fi
	if [ -d "$cache" ]; then
		if ask "Also remove the model cache at $cache?"; then
			run rm -rf "$cache"
			ok "removed $cache"
		fi
	fi
}

main() {
	setup_colors
	parse_args "$@"
	header
	if [ "$opt_uninstall" = 1 ]; then
		uninstall
		return 0
	fi
	workdir=$(mktemp -d "${TMPDIR:-/tmp}/vakt-install.XXXXXX")
	trap cleanup EXIT
	trap 'cleanup; exit 130' INT
	trap 'cleanup; exit 143' TERM

	detect_platform
	resolve_version
	info "release: $VAKT_REPO $VAKT_VERSION, asset: $asset"
	download_and_verify
	install_binary
	path_advice
	post_install
	say ""
	if [ "$opt_dry_run" = 1 ]; then
		ok "dry run complete; nothing was changed"
	else
		ok "Vaktex OSS is ready. Try: ${c_bold}vakt patrol .${c_reset}"
	fi
}

main "$@"
