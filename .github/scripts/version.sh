# shellcheck shell=sh
# Sourced by build steps: sets VAKT_VERSION from the git ref without letting
# arbitrary ref text reach make/ldflags. Tags must be strict semver.
case ${GITHUB_REF:-} in
refs/tags/*)
	tag=${GITHUB_REF_NAME:-}
	if printf '%s' "$tag" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$'; then
		VAKT_VERSION=$tag
	else
		echo "::error::tag '$tag' is not a valid release version (vMAJOR.MINOR.PATCH[-pre])"
		exit 1
	fi
	;;
*)
	VAKT_VERSION="dev-$(printf '%s' "${GITHUB_SHA:-unknown}" | cut -c1-7)"
	;;
esac
export VAKT_VERSION
