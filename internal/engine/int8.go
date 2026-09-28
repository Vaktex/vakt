//go:build mlx

package engine

// --precision int8: the backbone's matmuls run as int8 x int8 -> int32 on
// the M5-class GPU matmul units (Metal 4 tensor ops), with per-output-channel
// weight scales (fixed at load) and per-token activation scales (computed
// right before each matmul). Everything else runs as in fp16 mode. MLX 0.31
// has no int8 matmul on Metal, hence the custom kernels.
//
// Measured on an M5 Pro against MLX's fp16 matmul at a 4096-token batch:
// gate|up (1024 -> 7168) 45.3 vs 26.0 TFLOPS, down (3584 -> 1024) 39.3 vs
// 22.9 TFLOPS (BenchmarkInt8Gemm).

import (
	"fmt"
	"os"
	"strings"

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
// threadgroup memory for the epilogue (TM*TN*4 bytes, 32 KB max).
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
            y[size_t(m) * N + n] = static_cast<OutT>(float(sc[i]) * sx[m] * sw[n]);
        }
    }
`

// quantRowsSource quantizes each row of a [R,K] to int8 with a symmetric
// per-row scale: s = max|a|/127, q = clamp(rint(a/s), -127, 127). One
// threadgroup of 256 threads per row.
const quantRowsSource = `
    const uint tid = thread_position_in_threadgroup.x;
    const size_t row = threadgroup_position_in_grid.y;
    const int K = a_shape[1];
    const device InT* ar = a + row * K;
    threadgroup float part[8];
    float mx = 0.0f;
    for (int i = int(tid); i < K; i += 256) { mx = metal::max(mx, metal::abs(float(ar[i]))); }
    mx = simd_max(mx);
    if (thread_index_in_simdgroup == 0) { part[simdgroup_index_in_threadgroup] = mx; }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float amax = 0.0f;
    for (int i = 0; i < 8; ++i) { amax = metal::max(amax, part[i]); }
    const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / sc;
    device int8_t* qr = q + row * K;
    for (int i = int(tid); i < K; i += 256) {
        qr[i] = static_cast<int8_t>(metal::clamp(metal::rint(float(ar[i]) * inv), -127.0f, 127.0f));
    }
    if (tid == 0) { s[row] = sc; }
`

// int8MatmulKinds are the backbone matmuls --precision int8 can run in int8.
var int8MatmulKinds = []string{"in_proj", "out_proj", "gate_up", "down", "qkv", "o"}

// int8Kinds is the set of matmul kinds that run in int8: VAKT_INT8 as a
// comma-separated list of int8MatmulKinds, or all of them. The others stay
// fp16, so accuracy can be traded against speed per kind.
func int8Kinds() map[string]bool {
	out := map[string]bool{}
	v := strings.TrimSpace(os.Getenv("VAKT_INT8"))
	if v == "" {
		for _, k := range int8MatmulKinds {
			out[k] = true
		}
		return out
	}
	for _, k := range strings.Split(v, ",") {
		out[strings.TrimSpace(k)] = true
	}
	return out
}

// q8w is a weight quantized for the int8 GEMM: q int8 [N,K], s float32 [N].
type q8w struct{ q, s *mlx.Array }

type int8Kernels struct{ gemm, quant *mlx.Kernel }

func newInt8Kernels() *int8Kernels {
	return &int8Kernels{
		gemm:  mlx.NewKernel("vakt_int8_gemm", []string{"x", "w", "sx", "sw"}, []string{"y"}, int8GemmSource, int8GemmHeader),
		quant: mlx.NewKernel("vakt_quant_rows", []string{"a"}, []string{"q", "s"}, quantRowsSource, ""),
	}
}

func (k *int8Kernels) free() {
	if k != nil {
		k.gemm.Free()
		k.quant.Free()
	}
}

// int8Tile picks the GEMM tile: 64x64 for wide outputs, 64x128 for narrow
// ones (best of the configurations BenchmarkInt8Gemm measured).
func int8Tile(N int) (tm, tn, nsg int) {
	if N <= 2048 {
		return 64, 128, 4
	}
	return 64, 64, 4
}

// quantRows quantizes a [R,K] to int8 rows with float32 scales [R].
func (k *int8Kernels) quantRows(x *mlx.Ctx, a *mlx.Array, R, K int) (q, s *mlx.Array) {
	o := x.Apply(k.quant, []*mlx.Array{a}, mlx.KernelLaunch{
		Grid:        [3]int{256, R, 1},
		ThreadGroup: [3]int{256, 1, 1},
		Outputs: []mlx.KernelOutput{
			{Shape: []int{R, K}, Dtype: mlx.Int8},
			{Shape: []int{R}, Dtype: mlx.Float32},
		},
		TemplateDtypes: map[string]mlx.DType{"InT": a.Dtype()},
	})
	return o[0], o[1]
}

// gemm is y [M,N] (out dtype) = (xq [M,K] * sx) @ (w.q [N,K] * w.s)^T.
func (k *int8Kernels) gemmOp(x *mlx.Ctx, xq, sx *mlx.Array, w q8w, M, N int, out mlx.DType) *mlx.Array {
	tm, tn, nsg := int8Tile(N)
	return k.gemmTile(x, xq, sx, w, M, N, out, tm, tn, nsg)
}

func (k *int8Kernels) gemmTile(x *mlx.Ctx, xq, sx *mlx.Array, w q8w, M, N int, out mlx.DType, tm, tn, nsg int) *mlx.Array {
	return x.Apply(k.gemm, []*mlx.Array{xq, w.q, sx, w.s}, mlx.KernelLaunch{
		Grid:           [3]int{(N + tn - 1) / tn * 32 * nsg, (M + tm - 1) / tm, 1},
		ThreadGroup:    [3]int{32 * nsg, 1, 1},
		Outputs:        []mlx.KernelOutput{{Shape: []int{M, N}, Dtype: out}},
		TemplateInts:   map[string]int{"TM": tm, "TN": tn, "NSG": nsg},
		TemplateDtypes: map[string]mlx.DType{"OutT": out},
	})[0]
}

// quantizeWeight turns a compute-layout weight [K,N] into int8 [N,K] with
// symmetric per-output-channel scales [N].
func quantizeWeight(x *mlx.Ctx, wKN *mlx.Array) q8w {
	w := x.Contiguous(x.Transpose(x.AsType(wKN, mlx.Float32), 1, 0)) // [N,K]
	amax := x.Max(x.Abs(w), true, 1)                                 // [N,1]
	sc := x.Divide(x.Maximum(amax, x.Scalar(1e-12)), x.Scalar(127))
	q := x.Round(x.Divide(w, sc))
	q = x.Minimum(x.Maximum(q, x.Scalar(-127)), x.Scalar(127))
	return q8w{q: x.Contiguous(x.AsType(q, mlx.Int8)), s: x.Reshape(sc, -1)}
}

// matmul is h @ W for a backbone weight W [K,N], as an int8 GEMM when the
// engine runs --precision int8 (h is [..., K]; the result keeps h's leading
// dims), else MLX's matmul.
func (m *model) matmul(x *mlx.Ctx, h, W *mlx.Array) *mlx.Array {
	w, ok := m.w.q8[W]
	if !ok || m.i8 == nil {
		return x.Matmul(h, W)
	}
	shape := h.Shape()
	K := shape[len(shape)-1]
	R := h.Size() / K
	N := w.s.Size()
	q, s := m.i8.quantRows(x, x.Reshape(h, R, K), R, K)
	y := m.i8.gemmOp(x, q, s, w, R, N, m.w.compute)
	out := append(append([]int(nil), shape[:len(shape)-1]...), N)
	return x.Reshape(y, out...)
}

// probeInt8 runs one small int8 GEMM so a GPU or OS without Metal 4 tensor
// ops fails at Open with a clear message instead of mid-scan.
func probeInt8(s *mlx.Stream, k *int8Kernels) error {
	x := mlx.NewCtx(s)
	defer x.Free()
	a := x.AsType(x.Add(x.Zeros(mlx.Float32, 64, 128), x.Scalar(0.5)), mlx.Float16)
	w := quantizeWeight(x, x.AsType(x.Add(x.Zeros(mlx.Float32, 128, 64), x.Scalar(0.25)), mlx.Float16))
	q, sc := k.quantRows(x, a, 64, 128)
	y := k.gemmOp(x, q, sc, w, 64, 64, mlx.Float16)
	v, err := x.Float32s(x.AsType(y, mlx.Float32))
	if err != nil {
		return err
	}
	if d := v[0] - 16; d > 0.05 || d < -0.05 { // 128 * 0.5 * 0.25
		return fmt.Errorf("int8 GEMM self-test gave %v, want 16", v[0])
	}
	return nil
}
