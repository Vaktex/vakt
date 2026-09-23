#!/bin/sh
# Release hygiene audit for a vakt prod binary.
#
# Fails if the binary leaks build-machine paths, our source layout, module
# metadata, debug info, or anything that looks like a credential; if its
# version/backend stamp is missing or wrong; or (on macOS) if its signature
# does not verify.
#
#   EXPECT_VERSION=v1.2.3 EXPECT_BACKEND=metal scripts/audit.sh dist/vakt-darwin-arm64
set -eu

bin=${1:?usage: audit.sh <binary>}
[ -f "$bin" ] || { echo "audit: no such file: $bin" >&2; exit 1; }

fail=0
pass() { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fail=1; }
skip() { printf '  skip  %s\n' "$1"; }

echo "audit: $bin ($(wc -c <"$bin" | tr -d ' ') bytes)"

# 1. Module metadata. garble -tiny + -trimpath leave no usable build info.
if ! command -v go >/dev/null 2>&1; then
	bad "go not found: cannot inspect build info"
elif go version -m "$bin" 2>/dev/null | grep -q 'github.com/vaktex'; then
	bad "module path visible in build info (go version -m)"
else
	pass "no module path in build info"
fi

# 2. Strings that must not appear. Print only the matching fragment.
leaks=$(strings -a "$bin" | grep -oE \
	-e 'github\.com/vaktex[^ ]{0,40}' \
	-e '/Users/[A-Za-z][^ ]{0,40}' \
	-e '/home/[A-Za-z][^ ]{0,40}' \
	-e '/root/[^ ]{0,40}' \
	-e '(^|[^A-Za-z0-9._-])/src/[A-Za-z][^ ]{0,40}' \
	-e '/Users/runner/work/[^r][^ ]{0,40}' \
	-e '/home/runner/work/[^r][^ ]{0,40}' \
	-e '/\.cargo/(registry|git)[^ ]{0,40}' \
	-e '/src/third_party[^ ]{0,40}' \
	-e 'vakt/(internal|cmd)/[^ ]{0,40}' \
	-e 'internal/(engine|pipeline|hub|tokenize|ast|walk|report|brand|core|labels)/[^ ]{0,20}' \
	-e '(^|[^A-Za-z0-9_])hf_[A-Za-z0-9]{34}([^A-Za-z0-9]|$)' \
	-e 'HF_TOKEN=[^ ]{0,10}' \
	| grep -vE '^/(Users|home)/runner/work/rust/rust/(build|library)/' \
	| sort -u | head -n 10 || true)
if [ -n "$leaks" ]; then
	bad "leaking strings:"
	printf '%s\n' "$leaks" | sed 's/^/          /'
else
	pass "no source paths, home dirs or tokens in strings"
fi

# 3. Debug information and signature.
case "$(uname -s)" in
Darwin)
	if otool -l "$bin" | grep -q '__DWARF'; then bad "DWARF segment present"; else pass "no DWARF"; fi
	if codesign -v "$bin" 2>/dev/null; then pass "codesign verifies"; else bad "codesign verification failed"; fi
	;;
Linux)
	if readelf -S "$bin" 2>/dev/null | grep -q '\.debug_'; then bad "debug sections present"; else pass "no debug sections"; fi
	# install.sh promises glibc >= MAX_GLIBC (Ubuntu 22.04). Building on a
	# newer glibc silently raises the floor via symbol versions.
	max_glibc=${MAX_GLIBC:-2.35}
	need=$(objdump -T "$bin" 2>/dev/null | grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' | sed 's/GLIBC_//' | sort -uV | tail -n 1)
	if [ -z "$need" ]; then
		pass "no glibc symbol versions (static?)"
	elif [ "$(printf '%s\n%s\n' "$need" "$max_glibc" | sort -V | tail -n 1)" = "$max_glibc" ]; then
		pass "glibc requirement $need <= $max_glibc"
	else
		bad "binary needs glibc $need > supported $max_glibc (build on an older base image)"
	fi
	# Every dynamic dependency must be something install.sh checks for or
	# installs (glibc, OpenBLAS/LAPACK, the CUDA 13 runtime) or ships with it.
	unexpected=$(readelf -d "$bin" 2>/dev/null | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | grep -vE \
		'^(libc|libm|libdl|librt|libpthread|libstdc\+\+|libgcc_s|ld-linux[-a-z0-9_]*)\.so|^lib(openblas|lapack|lapacke|gfortran|quadmath)\.so|^lib(cublas|cublasLt|nvrtc|cudnn|cuda|nccl)\.so' || true)
	if [ -n "$unexpected" ]; then
		bad "unexpected shared library dependencies: $(printf '%s' "$unexpected" | tr '\n' ' ')"
	else
		pass "shared library dependencies are all provisioned by install.sh"
	fi
	;;
*)
	bad "unsupported audit host: $(uname -s)"
	;;
esac

# 4. The version/backend stamp survived obfuscation (-X under garble). The
#    binary must run here; cross-built binaries are audited on their target.
if out=$("$bin" version 2>&1) && [ -n "$out" ]; then
	stamp=$(printf '%s\n' "$out" | grep -oE 'version=[^ ]+ commit=[^ ]+ backend=[^ ]+' | head -n 1 || true)
	if [ -z "$stamp" ]; then
		bad "version output has no 'version=… commit=… backend=…' stamp: $(printf '%s' "$out" | head -n 1)"
	else
		v=$(printf '%s' "$stamp" | sed 's/.*version=\([^ ]*\).*/\1/')
		c=$(printf '%s' "$stamp" | sed 's/.*commit=\([^ ]*\).*/\1/')
		b=$(printf '%s' "$stamp" | sed 's/.*backend=\([^ ]*\).*/\1/')
		stamp_ok=1
		sbad() { bad "$1"; stamp_ok=0; }
		case $v in dev|none|unknown|"") sbad "version not stamped ($stamp)" ;; esac
		case $c in none|unknown|"") sbad "commit not stamped ($stamp)" ;; esac
		case $b in dev|none|unknown|"") sbad "backend not stamped ($stamp)" ;; esac
		if [ -n "${EXPECT_VERSION:-}" ] && [ "$v" != "$EXPECT_VERSION" ]; then sbad "version $v != expected $EXPECT_VERSION"; fi
		if [ -n "${EXPECT_BACKEND:-}" ] && [ "$b" != "$EXPECT_BACKEND" ]; then sbad "backend $b != expected $EXPECT_BACKEND"; fi
		if [ "$b" = fake ] && [ "${EXPECT_BACKEND:-}" != fake ]; then sbad "fake-engine binary"; fi
		[ "$stamp_ok" -eq 1 ] && pass "stamp: $stamp"
	fi
else
	skip "version check (binary did not run on this host)"
fi

if [ "$fail" -ne 0 ]; then echo "audit: FAILED"; exit 1; fi
echo "audit: passed"
