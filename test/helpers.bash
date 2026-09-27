# Stub commands are written in single quotes on purpose (they expand when run).
# shellcheck disable=SC2016
# Shared bats helpers: every external command the installer consults is
# replaced by a stub driven by STUB_* environment variables. Each stub logs
# its argv to $CALLS so tests can assert on what ran.

INSTALL_SH="$BATS_TEST_DIRNAME/../install.sh"

setup_stubs() {
	STUBS="$BATS_TEST_TMPDIR/stubs"
	CALLS="$BATS_TEST_TMPDIR/calls.log"
	ASSETS="$BATS_TEST_TMPDIR/assets"
	PREFIX="$BATS_TEST_TMPDIR/prefix"
	export STUBS CALLS ASSETS PREFIX BLAS_LIBS
	mkdir -p "$STUBS" "$ASSETS"
	: >"$CALLS"

	stub uname 'case "$1" in -s) echo "${STUB_KERNEL:-Linux}";; -m) echo "${STUB_MACHINE:-x86_64}";; *) echo "${STUB_KERNEL:-Linux}";; esac'
	stub sysctl 'echo "${STUB_TRANSLATED:-0}"'
	stub getconf 'if [ -n "${STUB_GLIBC-2.39}" ]; then echo "glibc ${STUB_GLIBC-2.39}"; else exit 1; fi'
	stub ldd 'echo "${STUB_LDD:-ldd (GNU libc) ${STUB_GLIBC-2.39}}"'
	stub ldconfig 'if [ -z "${STUB_NO_BLAS:-}" ]; then printf "%s" "$BLAS_LIBS"; fi; printf "%s" "${STUB_LDCONFIG:-}"'
	stub id 'if [ "$1" = -u ]; then echo "${STUB_UID:-1000}"; else /usr/bin/id "$@"; fi'
	stub sudo '"$@"'
	stub dpkg 'exit 0'
	stub apt-get 'if [ "$1" = install ] && [ -n "${STUB_APT_PROVIDES:-}" ]; then echo "$STUB_APT_PROVIDES" > "$STUBS/.ldconfig_after"; fi; exit 0'
	stub dnf 'exit 0'
	stub codesign 'exit 0'
	stub xattr 'exit 0'
	# curl: only serve files from $ASSETS; fail otherwise. <base>/latest
	# answers $STUB_LATEST (default v9.9.9), like get.vaktex.com does.
	stub curl '
out=""; url=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift ;;
    http*) url="$1" ;;
  esac
  shift
done
name=${url##*/}
if [ "$name" = latest ]; then printf "%s\n" "${STUB_LATEST:-v9.9.9}" >"$out"; exit 0; fi
[ -f "$ASSETS/$name" ] || exit 22
cp "$ASSETS/$name" "$out"'
	if [ -n "${STUB_NVIDIA:-}" ]; then
		stub nvidia-smi 'echo "$STUB_NVIDIA"'
	fi
	# A fake vakt binary for the release assets.
	printf '#!/bin/sh\necho "vakt stub $*"\n' >"$ASSETS/vakt-bin"
	chmod +x "$ASSETS/vakt-bin"
}

# ldconfig reflects libraries "installed" by the apt stub.
stub_ldconfig_dynamic() {
	stub ldconfig 'printf "%s" "$BLAS_LIBS"; if [ -f "$STUBS/.ldconfig_after" ]; then cat "$STUBS/.ldconfig_after"; else printf "%s" "${STUB_LDCONFIG:-}"; fi'
}

stub() {
	name=$1
	body=$2
	cat >"$STUBS/$name" <<EOF
#!/bin/sh
echo "$name \$*" >> "\$CALLS"
$body
EOF
	chmod +x "$STUBS/$name"
}

# publish <asset>: put the fake binary in $ASSETS under the asset name and
# write a correct SHA256SUMS.
publish() {
	cp "$ASSETS/vakt-bin" "$ASSETS/$1"
	(cd "$ASSETS" && shasum -a 256 "$1" >>SHA256SUMS)
}

os_release() {
	f="$BATS_TEST_TMPDIR/os-release"
	printf 'ID=%s\nVERSION_ID="%s"\n' "$1" "$2" >"$f"
	export VAKT_OS_RELEASE="$f"
}

run_installer() {
	# Minimal PATH: stubs first, then system dirs for sh/awk/sed/etc.
	PATH="$STUBS:/usr/bin:/bin:/usr/sbin:/sbin" HOME="$BATS_TEST_TMPDIR/home" NO_COLOR=1 \
		run "${VAKT_TEST_SHELL:-sh}" "$INSTALL_SH" --prefix "$PREFIX" --no-summon "$@" </dev/null
}

BLAS_LIBS='	libopenblas.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libopenblas.so.0
	liblapack.so.3 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/liblapack.so.3
'
# Used by install.bats (sourced).
# shellcheck disable=SC2034
CUDA_LIBS_ALL='	libcublas.so.13 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcublas.so.13
	libcublasLt.so.13 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcublasLt.so.13
	libnvrtc.so.13 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libnvrtc.so.13
	libcudnn.so.9 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudnn.so.9
'
