#!/bin/sh
# Download and verify the parity fixture bundle (mock checkpoint + reference
# outputs). Sets the step output available=true|false; never fails the job
# when the repository variables are unset.
set -eu

out=${GITHUB_OUTPUT:-/dev/stdout}

if [ -z "${FIXTURES_URL:-}" ] || [ -z "${FIXTURES_SHA256:-}" ]; then
	echo "::notice::PARITY_FIXTURES_URL / PARITY_FIXTURES_SHA256 not set; skipping parity"
	echo "available=false" >>"$out"
	exit 0
fi
case $FIXTURES_URL in
https://*) ;;
*) echo "::error::PARITY_FIXTURES_URL must be https"; exit 1 ;;
esac
case $FIXTURES_SHA256 in
*[!0-9a-f]*) echo "::error::PARITY_FIXTURES_SHA256 must be lowercase hex"; exit 1 ;;
esac
[ ${#FIXTURES_SHA256} -eq 64 ] || { echo "::error::PARITY_FIXTURES_SHA256 must be 64 hex chars"; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$tmp/fixtures.tar.zst" "$FIXTURES_URL"
if command -v sha256sum >/dev/null; then
	got=$(sha256sum "$tmp/fixtures.tar.zst" | awk '{print $1}')
else
	got=$(shasum -a 256 "$tmp/fixtures.tar.zst" | awk '{print $1}')
fi
if [ "$got" != "$FIXTURES_SHA256" ]; then
	echo "::error::parity fixture checksum mismatch (got $got)"
	exit 1
fi
# The bundle may only contain testdata/{models,parity}/...
tar --zstd -tf "$tmp/fixtures.tar.zst" | while IFS= read -r entry; do
	case $entry in
	testdata/|testdata/models/*|testdata/parity/*) ;;
	*) echo "::error::unexpected path in fixture bundle: $entry"; exit 1 ;;
	esac
	case $entry in *..*) echo "::error::path traversal in fixture bundle: $entry"; exit 1 ;; esac
done
tar --zstd -xf "$tmp/fixtures.tar.zst" --no-same-owner --no-same-permissions
echo "available=true" >>"$out"
