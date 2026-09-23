#!/bin/sh
# Notarize a Developer ID-signed binary. Requires a notarytool keychain
# profile: xcrun notarytool store-credentials "$NOTARY_PROFILE" ...
set -eu

bin=${1:?usage: notarize.sh <binary>}
profile=${NOTARY_PROFILE:?set NOTARY_PROFILE to a notarytool keychain profile}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ditto -c -k --keepParent "$bin" "$tmp/vakt.zip"
xcrun notarytool submit "$tmp/vakt.zip" --keychain-profile "$profile" --wait
echo "notarized: $bin (standalone binaries cannot be stapled; Gatekeeper checks online)"
