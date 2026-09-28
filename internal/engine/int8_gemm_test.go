//go:build mlx

package engine

// Tests and benchmarks for the int8 GEMM (int8.go) behind --precision int8.
//
//	go test -tags mlx -run Int8Gemm -bench Int8Gemm -benchtime 20x -v ./internal/engine

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/vaktex/vakt/internal/engine/mlx"
)

type int8GemmCfg struct{ TM, TN, NSG int }

func (c int8GemmCfg) String() string { return fmt.Sprintf("tile%dx%d_sg%d", c.TM, c.TN, c.NSG) }

// The int32 tile lives in threadgroup memory (32 KB max): TM*TN <= 8192.
var int8GemmCfgs = []int8GemmCfg{{64, 64, 4}, {64, 64, 8}, {128, 64, 4}, {64, 128, 4}}

func int8Gemm(x *mlx.Ctx, k *int8Kernels, c int8GemmCfg, xq, wq, sx, sw *mlx.Array, M, N int) *mlx.Array {
	return k.gemmTile(x, xq, sx, q8w{q: wq, s: sw}, M, N, mlx.Float16, c.TM, c.TN, c.NSG)
}

// int8Inputs makes random int8 operands (values in [-127, 127]) and scales.
func int8Inputs(x *mlx.Ctx, M, K, N int, seed int64) (xq, wq, sx, sw *mlx.Array, xv, wv []int32, sxv, swv []float32) {
	r := rand.New(rand.NewSource(seed)) // #nosec G404 -- test data
	xv = make([]int32, M*K)
	wv = make([]int32, N*K)
	for i := range xv {
		xv[i] = int32(r.Intn(255) - 127) // #nosec G115
	}
	for i := range wv {
		wv[i] = int32(r.Intn(255) - 127) // #nosec G115
	}
	sxv = make([]float32, M)
	swv = make([]float32, N)
	for i := range sxv {
		sxv[i] = 0.001 + r.Float32()*0.01
	}
	for i := range swv {
		swv[i] = 0.0001 + r.Float32()*0.001
	}
	xq = x.AsType(x.FromInt32(xv, M, K), mlx.Int8)
	wq = x.AsType(x.FromInt32(wv, N, K), mlx.Int8)
	return xq, wq, x.FromFloat32(sxv, M), x.FromFloat32(swv, N), xv, wv, sxv, swv
}

func needNAX(t testing.TB) *mlx.Stream {
	if testDevice() == "cpu" || !gpuOK() || gpuBackend != "metal" {
		t.Skip("int8 GEMM prototype needs a Metal GPU")
	}
	return mlx.GPU()
}

// TestInt8GemmCorrect checks every tile configuration against an exact
// reference on a small odd-shaped problem (partial tiles on both edges).
func TestInt8GemmCorrect(t *testing.T) {
	s := needNAX(t)
	defer s.Free()
	k := newInt8Kernels()
	defer k.free()
	const M, K, N = 200, 320, 136
	x := mlx.NewCtx(s)
	defer x.Free()
	xq, wq, sx, sw, xv, wv, sxv, swv := int8Inputs(x, M, K, N, 1)
	for _, c := range int8GemmCfgs {
		got, err := x.Float32s(x.AsType(int8Gemm(x, k, c, xq, wq, sx, sw, M, N), mlx.Float32))
		if err != nil {
			t.Fatalf("%s: %v", c, err)
		}
		var maxRel float64
		for m := 0; m < M; m++ {
			for n := 0; n < N; n++ {
				var acc int64
				for kk := 0; kk < K; kk++ {
					acc += int64(xv[m*K+kk]) * int64(wv[n*K+kk])
				}
				want := float64(acc) * float64(sxv[m]) * float64(swv[n])
				d := math.Abs(float64(got[m*N+n])-want) / math.Max(math.Abs(want), 1e-3)
				maxRel = math.Max(maxRel, d)
			}
		}
		t.Logf("%s: max relative error %.3g", c, maxRel)
		if maxRel > 2e-3 { // fp16 output rounding is ~5e-4
			t.Errorf("%s: max relative error %.3g", c, maxRel)
		}
	}
}

// BenchmarkInt8Gemm times the prototype against MLX's fp16 matmul (what the
// engine runs today) on the model's two largest matmul shapes at the
// default 4096-token batch: gate|up (1024 -> 7168) and down (3584 -> 1024).
func BenchmarkInt8Gemm(b *testing.B) {
	s := needNAX(b)
	defer s.Free()
	k := newInt8Kernels()
	defer k.free()
	for _, sh := range []struct {
		name    string
		M, K, N int
	}{{"gate_up", 4096, 1024, 7168}, {"down", 4096, 3584, 1024}} {
		x := mlx.NewCtx(s)
		xq, wq, sx, sw, _, _, _, _ := int8Inputs(x, sh.M, sh.K, sh.N, 2)
		// fp16 operands in the engine's layout: activations [M,K], weights [K,N].
		xh := x.AsType(xq, mlx.Float16)
		wh := x.Contiguous(x.Transpose(x.AsType(wq, mlx.Float16), 1, 0))
		if err := x.Eval(xq, wq, sx, sw, xh, wh); err != nil {
			b.Fatal(err)
		}
		flops := 2 * float64(sh.M) * float64(sh.K) * float64(sh.N)
		run := func(name string, f func(y *mlx.Ctx) *mlx.Array) {
			b.Run(sh.name+"/"+name, func(b *testing.B) {
				y := mlx.NewCtx(s)
				defer y.Free()
				if err := y.Eval(f(y)); err != nil { // warm-up (kernel JIT)
					b.Fatal(err)
				}
				b.ResetTimer()
				start := time.Now()
				for range b.N {
					z := mlx.NewCtx(s)
					if err := z.Eval(f(z)); err != nil {
						b.Fatal(err)
					}
					z.Free()
				}
				el := time.Since(start)
				b.ReportMetric(flops*float64(b.N)/el.Seconds()/1e12, "TFLOPS")
			})
		}
		run("fp16_mlx", func(y *mlx.Ctx) *mlx.Array { return y.Matmul(xh, wh) })
		for _, c := range int8GemmCfgs {
			run("int8_"+c.String(), func(y *mlx.Ctx) *mlx.Array { return int8Gemm(y, k, c, xq, wq, sx, sw, sh.M, sh.N) })
		}
		x.Free()
	}
}

// TestQuantizeWeight checks the load-time weight quantization on any device:
// per output channel, every dequantized weight is within half a step.
func TestQuantizeWeight(t *testing.T) {
	s := mlx.TestStream()
	defer s.Free()
	x := mlx.NewCtx(s)
	defer x.Free()
	const K, N = 96, 40
	r := rand.New(rand.NewSource(3)) // #nosec G404 -- test data
	wv := make([]float32, K*N)
	for i := range wv {
		wv[i] = float32(r.NormFloat64()) * float32(1+i%N) * 0.01
	}
	q := quantizeWeight(x, x.FromFloat32(wv, K, N)) // [K,N] -> q [N,K], s [N]
	qv, err := x.Float32s(x.AsType(q.q, mlx.Float32))
	if err != nil {
		t.Fatal(err)
	}
	sv, err := x.Float32s(q.s)
	if err != nil {
		t.Fatal(err)
	}
	for n := 0; n < N; n++ {
		for k := 0; k < K; k++ {
			got := qv[n*K+k] * sv[n]
			if d := math.Abs(float64(got - wv[k*N+n])); d > float64(sv[n])/2+1e-7 {
				t.Fatalf("w[%d,%d] = %v, dequantized %v (scale %v)", k, n, wv[k*N+n], got, sv[n])
			}
			if qv[n*K+k] < -127 || qv[n*K+k] > 127 {
				t.Fatalf("q out of range: %v", qv[n*K+k])
			}
		}
	}
}
