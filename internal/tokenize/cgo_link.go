package tokenize

// github.com/daulet/tokenizers links with a bare "-ltokenizers" and no search
// path. cgo LDFLAGS from every package are merged into the final link, so
// adding the -L here lets a plain `go build` find the static library that
// third_party/tokenizers/build.sh installs. Nothing else is needed from C.

/*
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/../../third_party/tokenizers/lib/darwin_arm64
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/../../third_party/tokenizers/lib/linux_amd64
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/../../third_party/tokenizers/lib/linux_arm64
*/
import "C"
