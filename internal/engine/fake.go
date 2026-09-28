package engine

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"

	"github.com/vaktex/vakt/internal/core"
)

// Fake is a deterministic engine: scores are a hash of the token ids. It
// exercises every path around the model without weights.
type Fake struct{}

func (Fake) Info() core.EngineInfo {
	return core.EngineInfo{Backend: "fake", Device: "none", Precision: "fp32", ModelSHA: "fake"}
}

func (Fake) Score(ctx context.Context, batch [][]int32) ([]core.Scores, error) {
	out := make([]core.Scores, len(batch))
	for i, ids := range batch {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		h := sha256.New()
		var b [4]byte
		for _, id := range ids {
			binary.LittleEndian.PutUint32(b[:], uint32(id)) // #nosec G115 -- bit reinterpretation for hashing is intended
			h.Write(b[:])
		}
		sum := h.Sum(nil)
		out[i].Severity = unit(sum[0:2])
		for f := 0; f < core.NumFamilies; f++ {
			j := (2 + f) % len(sum)
			out[i].Families[f] = unit(sum[j : j+1])
		}
	}
	return out, nil
}

func (Fake) Close() error { return nil }

func unit(b []byte) float32 {
	var v uint32
	for _, x := range b {
		v = v<<8 | uint32(x)
	}
	return float32(float64(v) / (math.Pow(256, float64(len(b))) - 1))
}
