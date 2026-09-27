#!/usr/bin/env bats
# publish-r2.sh prune-plan: which release folders to delete from the bucket.
# Pure: reads folder names on stdin, prints the ones to remove.

SCRIPT="$BATS_TEST_DIRNAME/../.github/scripts/publish-r2.sh"

plan() {
	# plan <keep> <latest> <tags...>
	keep=$1 latest=$2
	shift 2
	printf '%s\n' "$@" | sh "$SCRIPT" prune-plan "$keep" "$latest"
}

@test "keeps the newest three versions and prunes the rest" {
	run plan 3 v0.5.0 v0.1.0 v0.2.0 v0.3.0 v0.4.0 v0.5.0
	[ "$status" -eq 0 ]
	[ "$output" = "$(printf 'v0.1.0\nv0.2.0')" ] || false
}

@test "orders numerically, not lexically (v0.10.0 is newer than v0.9.0)" {
	run plan 3 v0.10.0 v0.8.0 v0.9.0 v0.10.0 v0.7.0
	[ "$status" -eq 0 ]
	[ "$output" = "v0.7.0" ] || false
}

@test "a pre-release is older than its final release" {
	run plan 3 v1.0.0 v1.0.0 v1.0.0-rc1 v1.0.0-rc2 v0.9.0
	[ "$status" -eq 0 ]
	[ "$output" = "v0.9.0" ] || false
}

@test "never prunes the version latest points to, even behind newer pre-releases" {
	run plan 3 v0.3.0 v0.2.0 v0.3.0 v0.4.0-rc1 v0.4.0-rc2 v0.4.0-rc3
	[ "$status" -eq 0 ]
	[ "$output" = "v0.2.0" ] || false
}

@test "three or fewer versions prunes nothing" {
	run plan 3 v0.2.0 v0.1.0 v0.2.0
	[ "$status" -eq 0 ]
	[ -z "$output" ]
}

@test "names that are not release folders are ignored, never deleted" {
	run plan 3 v0.4.0 latest v0.1.0 v0.2.0 v0.3.0 v0.4.0 'junk' '../x' 'v1/../../y'
	[ "$status" -eq 0 ]
	[ "$output" = "v0.1.0" ] || false
}

@test "the published asset list can be narrowed, but never silently" {
	# A release that quietly lacks a platform is worse than one that refuses to
	# publish, so the set is explicit: VAKT_ASSETS overrides it, and anything
	# outside the known assets is refused.
	run sh "$SCRIPT" assets
	[ "$status" -eq 0 ]
	[[ "$output" == *"vakt-darwin-arm64"* ]] || false
	[[ "$output" == *"vakt-linux-amd64-cuda13"* ]] || false
	[[ "$output" == *"install.sh"* ]] || false
	[[ "$output" == *"SHA256SUMS"* ]] || false

	run env VAKT_ASSETS="vakt-linux-amd64-cpu vakt-linux-arm64-cpu" sh "$SCRIPT" assets
	[ "$status" -eq 0 ]
	[[ "$output" == *"vakt-linux-amd64-cpu"* ]] || false
	[[ "$output" != *"vakt-darwin-arm64"* ]] || false
	# install.sh and SHA256SUMS are always published: the installer and the
	# integrity list are not optional.
	[[ "$output" == *"install.sh"* ]] || false
	[[ "$output" == *"SHA256SUMS"* ]] || false
}

@test "an unknown asset name is refused" {
	run env VAKT_ASSETS="vakt-linux-amd64-cpu vakt-openbsd-sparc64" sh "$SCRIPT" assets
	[ "$status" -ne 0 ]
	[[ "$output" == *"not a known release asset"* ]] || false
}

@test "is-prerelease distinguishes final and pre-release tags" {
	run sh "$SCRIPT" is-prerelease v1.2.3
	[ "$status" -eq 1 ]
	run sh "$SCRIPT" is-prerelease v1.2.3-rc.1
	[ "$status" -eq 0 ]
}
