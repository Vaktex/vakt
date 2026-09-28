//go:build mlx

package engine

// Prototype: an int8 x int8 -> int32 GEMM on the M5-class GPU matmul units
// (Metal 4 tensor ops, the API MLX's NAX kernels and llama.cpp use), with a
// fused epilogue that applies per-row activation scales and per-column weight
// scales and writes fp16. MLX 0.31 has no int8 matmul on Metal (matmul
// rejects integer types and qqmm is "NYI for the general case"), so this is
// the only way to try int8 compute. It lives in a test until the benchmark
// below says it beats MLX's fp16 matmul.
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

const int8GemmHeader = `
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
`

// int8GemmSource computes y[m,n] = half(sx[m] * sw[n] * sum_k x[m,k]*w[n,k])
// for int8 x [M,K], w [N,K] (both K-contiguous), float32 sx [M], sw [N].
// One threadgroup of NSG SIMD groups per TM x TN output tile; the matmul op
// streams K from device memory itself. The int32 tile goes through
// threadgroup memory for the epilogue.
const int8GemmSource = `
    const uint tid = thread_index_in_threadgroup;
    const int M = x_shape[0];
    const int K = x_shape[1];
    const int N = w_shape[0];
    const int m0 = int(threadgroup_position_in_grid.y) * TM;
    const int n0 = int(threadgroup_position_in_grid.x) * TN;
    if (m0 >= M || n0 >= N) { return; }

    auto tX = metal::tensor((device int8_t*)(x + size_t(m0) * K),
        metal::dextents<int32_t, 2>(K, metal::min(TM, M - m0)), metal::array<int, 2>({1, K}));
    auto tW = metal::tensor((device int8_t*)(w + size_t(n0) * K),
        metal::dextents<int32_t, 2>(K, metal::min(TN, N - n0)), metal::array<int, 2>({1, K}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(TM, TN, static_cast<int>(metal::dynamic_extent), false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        metal::execution_simdgroups<NSG>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), int32_t>();
    mm.run(tX, tW, cT);

    threadgroup int32_t sc[TM * TN];
    auto tS = metal::tensor<threadgroup int32_t, metal::dextents<int32_t, 2>, metal::tensor_inline>(
        sc, metal::dextents<int32_t, 2>(TN, TM));
    cT.store(tS);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (int i = int(tid); i < TM * TN; i += NSG * 32) {
        const int r = i / TN;
        const int c = i % TN;
        const int m = m0 + r;
        const int n = n0 + c;
        if (m < M && n < N) {
            y[size_t(m) * N + n] = static_cast<half>(float(sc[i]) * sx[m] * sw[n]);
        }
    }
`

type int8GemmCfg struct{ TM, TN, NSG int }

func (c int8GemmCfg) String() string { return fmt.Sprintf("tile%dx%d_sg%d", c.TM, c.TN, c.NSG) }

// The int32 tile lives in threadgroup memory (32 KB max): TM*TN <= 8192.
var int8GemmCfgs = []int8GemmCfg{{64, 64, 4}, {64, 64, 8}, {128, 64, 4}, {64, 128, 4}}

func int8Gemm(x *mlx.Ctx, k *mlx.Kernel, c int8GemmCfg, xq, wq, sx, sw *mlx.Array, M, N int) *mlx.Array {
	return x.Apply(k, []*mlx.Array{xq, wq, sx, sw}, mlx.KernelLaunch{
		Grid:         [3]int{(N + c.TN - 1) / c.TN * 32 * c.NSG, (M + c.TM - 1) / c.TM, 1},
		ThreadGroup:  [3]int{32 * c.NSG, 1, 1},
		Outputs:      []mlx.KernelOutput{{Shape: []int{M, N}, Dtype: mlx.Float16}},
		TemplateInts: map[string]int{"TM": c.TM, "TN": c.TN, "NSG": c.NSG},
	})[0]
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
	k := mlx.NewKernel("vakt_int8_gemm_proto", []string{"x", "w", "sx", "sw"}, []string{"y"}, int8GemmSource, int8GemmHeader)
	defer k.Free()
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
	k := mlx.NewKernel("vakt_int8_gemm_proto", []string{"x", "w", "sx", "sw"}, []string{"y"}, int8GemmSource, int8GemmHeader)
	defer k.Free()
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
