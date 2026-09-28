#!/bin/sh
# Publish a release to the vakt-releases R2 bucket and keep only the newest
# few. get.vaktex.com/oss-vakt streams downloads from that bucket; GitHub
# holds no release binaries.
#
#   publish-r2.sh publish <dist-dir> <tag>     upload, verify, point latest, prune
#   publish-r2.sh prune-plan <keep> <latest>   folder names on stdin -> ones to delete
#   publish-r2.sh is-prerelease <tag>          exit 0 for vX.Y.Z-pre
#   publish-r2.sh assets                       the files a publish would upload
#
# Bucket layout:
#   <tag>/vakt-darwin-arm64, <tag>/vakt-linux-{amd64,arm64}-{cpu,cuda13},
#   <tag>/install.sh, <tag>/SHA256SUMS
#   latest                                      the current final release tag
#
# publish needs R2_ACCOUNT_ID, R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY (an
# R2 API token scoped to this bucket, Object Read & Write). R2_BUCKET
# defaults to vakt-releases, R2_KEEP to 3.
set -eu

TAG_RE='^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$'
# Every binary a full release publishes. install.sh and SHA256SUMS are not in
# this list because they are never optional -- see assets() below.
KNOWN_BINARIES="vakt-darwin-arm64 vakt-linux-amd64-cuda13 vakt-linux-arm64-cuda13 vakt-linux-amd64-cpu vakt-linux-arm64-cpu"

die() { printf 'publish-r2: %s\n' "$*" >&2; exit 1; }

# assets: the files this run publishes. Defaults to every known binary, so a
# release that is missing a platform fails rather than shipping quietly. Set
# VAKT_ASSETS to publish a subset (e.g. a Linux-only release while a macOS
# runner is unavailable); unknown names are refused so a typo cannot become a
# release that silently lacks a platform.
assets() {
	binaries=${VAKT_ASSETS:-$KNOWN_BINARIES}
	for a in $binaries; do
		case " $KNOWN_BINARIES " in
		*" $a "*) ;;
		*) die "'$a' is not a known release asset (expected one of: $KNOWN_BINARIES)" ;;
		esac
	done
	# The installer and the integrity list always ship.
	printf '%s install.sh SHA256SUMS\n' "$binaries"
}

is_tag() { printf '%s\n' "$1" | grep -Eq "$TAG_RE"; }

is_prerelease() {
	is_tag "$1" || die "not a release tag: $1"
	case $1 in *-*) return 0 ;; *) return 1 ;; esac
}

# prune_plan <keep> <latest>: stdin is one bucket folder name per line.
# Prints (oldest first) the release folders beyond the newest <keep>,
# never including <latest>. Anything that is not a release tag is ignored.
prune_plan() {
	keep=$1 latest=$2
	grep -E "$TAG_RE" | awk -v keep="$keep" -v latest="$latest" '
	{
		t = $0; core = substr(t, 2); pre = ""
		i = index(core, "-")
		if (i > 0) { pre = substr(core, i + 1); core = substr(core, 1, i - 1) }
		split(core, v, ".")
		# A final release sorts after all of its pre-releases.
		key = sprintf("%010d.%010d.%010d.%d.%s", v[1], v[2], v[3], pre == "" ? 1 : 0, pre)
		keys[NR] = key; tags[key] = t; n = NR
	}
	END {
		# Insertion sort, newest first; n is a handful of folders.
		for (i = 2; i <= n; i++) {
			k = keys[i]; j = i - 1
			while (j >= 1 && keys[j] < k) { keys[j + 1] = keys[j]; j-- }
			keys[j + 1] = k
		}
		m = 0
		for (i = keep + 1; i <= n; i++) if (tags[keys[i]] != latest) drop[++m] = tags[keys[i]]
		for (i = m; i >= 1; i--) print drop[i]
	}'
}

# ---------------------------------------------------------------- R2 (S3 API)

s3() {
	aws --endpoint-url "https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com" "$@"
}

content_type() {
	case $1 in
	install.sh) echo 'text/x-shellscript; charset=utf-8' ;;
	SHA256SUMS) echo 'text/plain; charset=utf-8' ;;
	*) echo 'application/octet-stream' ;;
	esac
}

publish() {
	dist=$1 tag=$2
	: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID is not set}" "${R2_ACCESS_KEY_ID:?}" "${R2_SECRET_ACCESS_KEY:?}"
	bucket=${R2_BUCKET:-vakt-releases}
	keep=${R2_KEEP:-3}
	export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
	export AWS_DEFAULT_REGION=auto
	# R2 does not accept every checksum header newer AWS CLIs send by default.
	export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required

	is_tag "$tag" || die "not a release tag: $tag"
	ASSETS=$(assets)
	for f in $ASSETS; do [ -f "$dist/$f" ] || die "missing $dist/$f"; done
	(cd "$dist" && sha256sum -c --quiet SHA256SUMS) || die "SHA256SUMS does not match $dist"

	# A published tag is immutable: get.vaktex.com caches it forever.
	existing=$(s3 s3api list-objects-v2 --bucket "$bucket" --prefix "$tag/" --max-keys 1 --query 'KeyCount' --output text)
	[ "$existing" = 0 ] || die "$tag is already published; release a new tag instead"

	# SHA256SUMS last: a folder with SHA256SUMS is a complete upload.
	for f in $ASSETS; do
		echo "upload $tag/$f"
		s3 s3api put-object --bucket "$bucket" --key "$tag/$f" --body "$dist/$f" \
			--content-type "$(content_type "$f")" >/dev/null
	done

	# Read everything back and check it against SHA256SUMS before any
	# installer can be pointed at it.
	check=$(mktemp -d)
	trap 'rm -rf "$check"' EXIT
	for f in $ASSETS; do
		s3 s3api get-object --bucket "$bucket" --key "$tag/$f" "$check/$f" >/dev/null
	done
	(cd "$check" && sha256sum -c --quiet SHA256SUMS) || die "read-back of $tag does not match SHA256SUMS; latest not updated"
	echo "verified $tag in r2://$bucket"

	if is_prerelease "$tag"; then
		echo "$tag is a pre-release: latest is unchanged"
	else
		printf '%s\n' "$tag" >"$check/latest"
		s3 s3api put-object --bucket "$bucket" --key latest --body "$check/latest" \
			--content-type 'text/plain; charset=utf-8' --cache-control no-store >/dev/null
		echo "latest -> $tag"
	fi

	latest=""
	if s3 s3api get-object --bucket "$bucket" --key latest "$check/current" >/dev/null 2>&1; then
		latest=$(tr -d '\r\n ' <"$check/current")
	fi
	folders=$(s3 s3api list-objects-v2 --bucket "$bucket" --delimiter / --query 'CommonPrefixes[].Prefix' --output text |
		tr '\t' '\n' | sed 's#/$##')
	for old in $(printf '%s\n' "$folders" | prune_plan "$keep" "$latest"); do
		is_tag "$old" || continue
		echo "prune $old"
		s3 s3 rm --recursive --only-show-errors "s3://$bucket/$old/"
	done
}

case ${1:-} in
assets) [ $# -eq 1 ] || die "usage: publish-r2.sh assets"; assets ;;
publish) [ $# -eq 3 ] || die "usage: publish <dist-dir> <tag>"; publish "$2" "$3" ;;
prune-plan) [ $# -eq 3 ] || die "usage: prune-plan <keep> <latest>"; prune_plan "$2" "$3" ;;
is-prerelease) [ $# -eq 2 ] || die "usage: is-prerelease <tag>"; is_prerelease "$2" ;;
*) die "usage: publish-r2.sh publish|prune-plan|is-prerelease|assets ..." ;;
esac
