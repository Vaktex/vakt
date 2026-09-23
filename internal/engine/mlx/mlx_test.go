//go:build mlx

package mlx

import (
	"math"
	"os"
	"strings"
	"testing"
)

func newCtx(t *testing.T) *Ctx {
	t.Helper()
	s := TestStream()
	t.Cleanup(s.Free)
	x := NewCtx(s)
	t.Cleanup(x.Free)
	return x
}

func vals(t *testing.T, x *Ctx, a *Array) []float32 {
	t.Helper()
	v, err := x.Float32s(a)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func near(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol || math.IsNaN(float64(got[i])) {
			t.Fatalf("%s[%d] = %v want %v", name, i, got[i], want[i])
		}
	}
}

func TestMatmulAndElementwise(t *testing.T) {
	x := newCtx(t)
	a := x.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := x.FromFloat32([]float32{1, 0, 0, 1, 1, 1}, 3, 2)
	near(t, "matmul", vals(t, x, x.Matmul(a, b)), []float32{4, 5, 10, 11}, 1e-5)
	near(t, "add", vals(t, x, x.Add(a, a)), []float32{2, 4, 6, 8, 10, 12}, 1e-6)
	near(t, "sigmoid", vals(t, x, x.Sigmoid(x.FromFloat32([]float32{0}, 1))), []float32{0.5}, 1e-6)
	sp := vals(t, x, x.Softplus(x.FromFloat32([]float32{-30, 0, 30}, 3)))
	near(t, "softplus", sp, []float32{float32(math.Log1p(math.Exp(-30))), float32(math.Ln2), 30}, 1e-5)
	near(t, "sum", vals(t, x, x.Sum(a, false, 1)), []float32{6, 15}, 1e-5)
	near(t, "mean", vals(t, x, x.Mean(a, false, 0)), []float32{2.5, 3.5, 4.5}, 1e-5)
	if s := x.Transpose(a, 1, 0).Shape(); s[0] != 3 || s[1] != 2 {
		t.Fatalf("transpose shape %v", s)
	}
}

func TestSplitConcatTakeWhere(t *testing.T) {
	x := newCtx(t)
	a := x.FromFloat32([]float32{0, 1, 2, 3, 4, 5}, 6)
	parts := x.SplitAt(a, 0, 2, 5)
	near(t, "p0", vals(t, x, parts[0]), []float32{0, 1}, 0)
	near(t, "p1", vals(t, x, parts[1]), []float32{2, 3, 4}, 0)
	near(t, "p2", vals(t, x, parts[2]), []float32{5}, 0)
	near(t, "concat", vals(t, x, x.Concatenate(0, parts[2], parts[0])), []float32{5, 0, 1}, 0)
	emb := x.FromFloat32([]float32{10, 11, 20, 21, 30, 31}, 3, 2)
	near(t, "take", vals(t, x, x.Take(emb, x.FromInt32([]int32{2, 0}, 2), 0)), []float32{30, 31, 10, 11}, 0)
	cond := x.FromBool([]bool{true, false, true}, 3)
	nan := x.FromFloat32([]float32{float32(math.NaN()), 7, 8}, 3)
	// where must not propagate NaN from the unselected branch.
	near(t, "where", vals(t, x, x.Where(cond, x.FromFloat32([]float32{1, 2, 3}, 3), nan)), []float32{1, 7, 3}, 0)
}

func TestRMSNorm(t *testing.T) {
	x := newCtx(t)
	in := []float32{1, 2, 3, 4}
	w := []float32{0.5, 1, 1.5, 2}
	var ms float64
	for _, v := range in {
		ms += float64(v * v)
	}
	r := 1 / math.Sqrt(ms/4+1e-6)
	want := make([]float32, 4)
	for i := range in {
		want[i] = float32(float64(in[i]) * r * float64(w[i]))
	}
	got := vals(t, x, x.RMSNorm(x.FromFloat32(in, 1, 4), x.FromFloat32(w, 4), 1e-6))
	near(t, "rms_norm", got, want, 1e-5)
}

// RoPE (non-traditional = rotate_half) on the first `dims` features.
func TestRoPE(t *testing.T) {
	x := newCtx(t)
	const T, D, dims = 3, 8, 4
	const base = 10000.0
	in := make([]float32, T*D)
	for i := range in {
		in[i] = float32(i%7) - 3
	}
	got := vals(t, x, x.RoPE(x.FromFloat32(in, 1, 1, T, D), dims, false, base, 1, 0))
	want := make([]float32, T*D)
	copy(want, in)
	half := dims / 2
	for p := 0; p < T; p++ {
		for i := 0; i < half; i++ {
			theta := float64(p) / math.Pow(base, float64(2*i)/dims)
			c, s := math.Cos(theta), math.Sin(theta)
			x1, x2 := float64(in[p*D+i]), float64(in[p*D+i+half])
			want[p*D+i] = float32(x1*c - x2*s)
			want[p*D+i+half] = float32(x2*c + x1*s)
		}
	}
	near(t, "rope", got, want, 1e-4)
}

// Causal SDPA with GQA against a manual softmax attention.
func TestSDPACausalGQA(t *testing.T) {
	x := newCtx(t)
	const Hq, Hk, T, D = 4, 2, 5, 8
	mk := func(n int, seed float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(math.Sin(float64(i)*0.37 + seed))
		}
		return v
	}
	q, k, v := mk(Hq*T*D, 1), mk(Hk*T*D, 2), mk(Hk*T*D, 3)
	scale := float32(1 / math.Sqrt(D))
	got := vals(t, x, x.SDPA(x.FromFloat32(q, 1, Hq, T, D), x.FromFloat32(k, 1, Hk, T, D), x.FromFloat32(v, 1, Hk, T, D), scale, "causal", nil))
	want := make([]float32, Hq*T*D)
	for h := 0; h < Hq; h++ {
		kh := h / (Hq / Hk)
		for i := 0; i < T; i++ {
			sc := make([]float64, i+1)
			mx := math.Inf(-1)
			for j := 0; j <= i; j++ {
				var d float64
				for e := 0; e < D; e++ {
					d += float64(q[(h*T+i)*D+e]) * float64(k[(kh*T+j)*D+e])
				}
				sc[j] = d * float64(scale)
				mx = math.Max(mx, sc[j])
			}
			var z float64
			for j := range sc {
				sc[j] = math.Exp(sc[j] - mx)
				z += sc[j]
			}
			for e := 0; e < D; e++ {
				var o float64
				for j := range sc {
					o += sc[j] / z * float64(v[(kh*T+j)*D+e])
				}
				want[(h*T+i)*D+e] = float32(o)
			}
		}
	}
	near(t, "sdpa", got, want, 1e-5)
}

// Depthwise causal conv: left-pad K-1, groups = channels.
func TestConv1dDepthwise(t *testing.T) {
	x := newCtx(t)
	const L, C, K = 6, 3, 4
	in := make([]float32, L*C)
	for i := range in {
		in[i] = float32(i%5) - 2
	}
	w := make([]float32, C*K) // [C, K, 1]
	for i := range w {
		w[i] = float32(i%4)*0.25 - 0.3
	}
	out := x.Conv1d(x.FromFloat32(in, 1, L, C), x.FromFloat32(w, C, K, 1), 1, K-1, 1, C)
	// padding K-1 on both sides => L+K-1 outputs; causal = first L.
	out = x.Slice(out, []int{0, 0, 0}, []int{1, L, C}, nil)
	got := vals(t, x, out)
	want := make([]float32, L*C)
	for tt := 0; tt < L; tt++ {
		for c := 0; c < C; c++ {
			var s float64
			for k := 0; k < K; k++ {
				src := tt - (K - 1) + k
				if src >= 0 {
					s += float64(in[src*C+c]) * float64(w[c*K+k])
				}
			}
			want[tt*C+c] = float32(s)
		}
	}
	near(t, "conv1d", got, want, 1e-5)
}

func TestErrorsDoNotCrash(t *testing.T) {
	x := newCtx(t)
	a := x.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := x.FromFloat32([]float32{1, 2, 3, 4}, 2, 2)
	c := x.Matmul(a, b) // shape mismatch
	d := x.Add(c, c)    // after failure: no-op
	if err := x.Eval(d); err == nil || !strings.Contains(err.Error(), "matmul") {
		t.Fatalf("err = %v, want matmul error", err)
	}
	y := newCtx(t)
	_ = y.FromFloat32([]float32{1, 2}, 3)
	if y.Err() == nil {
		t.Fatal("bad shape accepted")
	}
}

func TestNoLeak(t *testing.T) {
	s := TestStream()
	defer s.Free()
	run := func() {
		x := NewCtx(s)
		defer x.Free()
		a := x.FromFloat32(make([]float32, 64*64), 64, 64)
		if _, err := x.Float32s(x.Matmul(a, a)); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 50; i++ {
		run()
	}
	ClearCache()
	base := ActiveMemory()
	for i := 0; i < 1000; i++ {
		run()
	}
	ClearCache()
	if grew := int64(ActiveMemory()) - int64(base); grew > 1<<20 {
		t.Fatalf("active memory grew by %d bytes over 1000 iterations", grew)
	}
}

func TestLoadSafetensors(t *testing.T) {
	const p = "../../../../../testdata/parity/layers.safetensors"
	if _, err := os.Stat(p); err != nil {
		t.Skip("parity layers not present")
	}
	x := newCtx(t)
	m, err := x.LoadSafetensors(p)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, a := range m {
			a.Free()
		}
	}()
	for _, k := range []string{"input_ids", "embeddings", "layers.00", "layers.23", "norm", "pooled"} {
		if m[k] == nil {
			t.Fatalf("missing %s", k)
		}
	}
	if s := m["norm"].Shape(); len(s) != 2 || s[1] != 1024 {
		t.Fatalf("norm shape %v", s)
	}
	// mean(norm) == pooled (the fixture's own invariant).
	near(t, "pooled", vals(t, x, x.Mean(m["norm"], false, 0)), vals(t, x, m["pooled"]), 1e-5)
}

// Float32s must return logical order for views (transpose/strided slice).
func TestFloat32sOfViews(t *testing.T) {
	x := newCtx(t)
	a := x.FromFloat32([]float32{0, 1, 2, 3, 4, 5}, 2, 3)
	near(t, "transpose", vals(t, x, x.Transpose(a, 1, 0)), []float32{0, 3, 1, 4, 2, 5}, 0)
	near(t, "strided", vals(t, x, x.Slice(a, []int{0, 0}, []int{2, 3}, []int{1, 2})), []float32{0, 2, 3, 5}, 0)
	near(t, "column", vals(t, x, x.Slice(a, []int{0, 1}, []int{2, 2}, nil)), []float32{1, 4}, 0)
}
