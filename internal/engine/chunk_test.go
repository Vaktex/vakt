//go:build mlx

package engine

import (
	"math"
	"runtime"
	"testing"

	"github.com/vaktex/vakt/internal/engine/mlx"
)

// TestDeltaChunkedMatchesScan checks the portable chunked algorithm against
// the per-token reference on random inputs, across chunk boundaries, B>1.
func TestDeltaChunkedMatchesScan(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	s := mlx.TestStream()
	defer s.Free()
	x := mlx.NewCtx(s)
	defer x.Free()
	for _, T := range []int{1, 17, 64, 65, 130} {
		B := 2
		rnd := func(n int, seed, scale float64) []float32 {
			v := make([]float32, n)
			for i := range v {
				v[i] = float32(math.Sin(float64(i)*0.731+seed) * scale)
			}
			return v
		}
		q := l2norm(x, x.FromFloat32(rnd(B*T*linHeads*linKeyDim, 1, 1), B, T, linHeads, linKeyDim))
		k := l2norm(x, x.FromFloat32(rnd(B*T*linHeads*linKeyDim, 2, 1), B, T, linHeads, linKeyDim))
		v := x.FromFloat32(rnd(B*T*linHeads*linValDim, 3, 1), B, T, linHeads, linValDim)
		g := x.Negative(x.Softplus(x.FromFloat32(rnd(B*T*linHeads, 4, 2), B, T, linHeads)))
		beta := x.Sigmoid(x.FromFloat32(rnd(B*T*linHeads, 5, 2), B, T, linHeads))
		a, err := x.Float32s(deltaChunked(x, q, k, v, g, beta, B, T))
		if err != nil {
			t.Fatal(err)
		}
		b, err := x.Float32s(deltaScanOps(x, q, k, v, g, beta, B, T))
		if err != nil {
			t.Fatal(err)
		}
		var m float64
		for i := range a {
			m = math.Max(m, math.Abs(float64(a[i]-b[i])))
		}
		t.Logf("T=%d: chunked vs scan max|Δ| = %.3g", T, m)
		if m > 1e-4 {
			t.Fatalf("T=%d: max|Δ| %.3g", T, m)
		}
	}
}
