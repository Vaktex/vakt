#!/bin/sh
# Release hygiene audit for a vakt prod binary.
#
# Fails if the binary leaks build-machine paths, our source layout, module
# metadata, debug info, or anything that looks like a credential, or (on
# macOS) if its signature does not verify.
set -eu

bin=${1:?usage: audit.sh <binary>}
[ -f "$bin" ] || { echo "audit: no such file: $bin" >&2; exit 1; }

fail=0
pass() { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fail=1; }

echo "audit: $bin ($(wc -c <"$bin" | tr -d ' ') bytes)"

# 1. Module metadata. garble -tiny + -trimpath leave no usable build info.
if go version -m "$bin" 2>/dev/null | grep -q 'github.com/vaktex'; then
	bad "module path visible in build info (go version -m)"
else
	pass "no module path in build info"
fi

# 2. Strings that must not appear.
leaks=$(strings -a "$bin" | grep -E \
	-e 'github\.com/vaktex' \
	-e '/Users/[A-Za-z]' \
	-e '/home/[A-Za-z]' \
	-e 'internal/(engine|pipeline|hub|tokenize|ast|walk|report|brand|core)' \
	-e 'hf_[A-Za-z0-9]{30,}' \
	-e 'HF_TOKEN=' \
	| head -n 5 || true)
if [ -n "$leaks" ]; then
	bad "leaking strings:"
	printf '%s\n' "$leaks" | sed 's/^/          /'
else
	pass "no source paths, home dirs or tokens in strings"
fi

# 3. Debug information.
case "$(uname -s)" in
Darwin)
	if otool -l "$bin" | grep -q '__DWARF'; then bad "DWARF segment present"; else pass "no DWARF"; fi
	if codesign -v "$bin" 2>/dev/null; then pass "codesign verifies"; else bad "codesign verification failed"; fi
	;;
Linux)
	if readelf -S "$bin" 2>/dev/null | grep -q '\.debug_'; then bad "debug sections present"; else pass "no debug sections"; fi
	;;
esac

# 4. Version stamp survived obfuscation (-X under garble). Only checked when
# the binary can run on this host.
if out=$("$bin" version 2>&1) && [ -n "$out" ]; then
	first=$(printf '%s\n' "$out" | head -n 1)
	case $out in
	*unknown*) bad "backend not stamped: $first" ;;
	*) pass "version stamped: $first" ;;
	esac
else
	printf '  skip  version check (binary did not run on this host)\n'
fi

if [ "$fail" -ne 0 ]; then echo "audit: FAILED"; exit 1; fi
echo "audit: passed"
