package safetensors

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const hashChunk = 4 << 20

// SHA256File streams path through SHA-256 and returns the lowercase hex
// digest. It checks ctx between chunks, so cancellation takes effect within
// one chunk read.
func SHA256File(ctx context.Context, path string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return "", fmt.Errorf("safetensors: sha256: %w", err)
	}
	defer f.Close() //nolint:errcheck // read-only
	return sha256Reader(ctx, f)
}

func sha256Reader(ctx context.Context, r io.Reader) (string, error) {
	h := sha256.New()
	buf := make([]byte, hashChunk)
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, rerr := r.Read(buf)
		if n > 0 {
			h.Write(buf[:n]) //nolint:errcheck // hash.Hash.Write never fails
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return "", fmt.Errorf("safetensors: sha256: %w", rerr)
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
