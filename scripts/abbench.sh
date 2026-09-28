#!/bin/sh
# Compare vakt configurations on the same machine state: runs each
# configuration in turn, round after round, so drift (heat, power mode,
# background load) hits every configuration alike, and prints the median
# tok/s of each. A single run per build is not enough: the same binary has
# varied by 15% between sessions.
#
#   scripts/abbench.sh [-n rounds] [-p path] [-b binary] [-a "vakt args"] CONFIG...
#
# CONFIG is a space-separated list of environment assignments, or - for the
# defaults. Example:
#
#   scripts/abbench.sh -n 5 - VAKT_CONV_STEPS=4 VAKT_OPEN_EARLY=1
set -eu

rounds=5
path=.
bin=./bin/vakt
args=
while getopts n:p:b:a: opt; do
	case "$opt" in
	n) rounds=$OPTARG ;;
	p) path=$OPTARG ;;
	b) bin=$OPTARG ;;
	a) args=$OPTARG ;;
	*)
		echo "usage: $0 [-n rounds] [-p path] [-b binary] [-a \"vakt args\"] CONFIG..." >&2
		exit 2
		;;
	esac
done
shift $((OPTIND - 1))
[ $# -gt 0 ] || set -- -

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

run() { # config -> tok/s
	cfg=$1
	[ "$cfg" = - ] && cfg=
	rm -f "$tmp/r.json"
	rc=0
	# shellcheck disable=SC2086 # CONFIG and args are deliberately word-split
	env $cfg "$bin" "$path" $args --no-cache -q --format json -o "$tmp/r.json" >/dev/null 2>"$tmp/err" || rc=$?
	# Exit status 1 only means findings were reported.
	if [ "$rc" -gt 1 ] || [ ! -s "$tmp/r.json" ]; then
		echo "vakt failed for config '$1' (exit $rc):" >&2
		cat "$tmp/err" >&2
		exit 1
	fi
	sed -n 's/.*"tokens_per_sec": *\([0-9.]*\).*/\1/p' "$tmp/r.json" | head -n 1
}

echo "warm-up run (not counted)..."
run - >/dev/null
r=1
while [ "$r" -le "$rounds" ]; do
	i=0
	for c in "$@"; do
		i=$((i + 1))
		v=$(run "$c")
		echo "$v" >>"$tmp/$i"
		printf 'round %d  %-40s %8.0f tok/s\n' "$r" "$c" "$v"
	done
	r=$((r + 1))
done

echo
printf '%-40s %8s %8s %8s\n' config median min max
i=0
for c in "$@"; do
	i=$((i + 1))
	sort -n "$tmp/$i" | awk -v c="$c" '{ v[NR] = $1 } END {
		m = (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2
		printf "%-40s %8.0f %8.0f %8.0f\n", c, m, v[1], v[NR] }'
done
