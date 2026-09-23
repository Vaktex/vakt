//go:build mlx

package engine

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/engine/mlx"
)

var testdata = func() string {
	if d := os.Getenv("VAKT_TESTDATA"); d != "" {
		return d
	}
	// .worktrees/<stream>/internal/engine -> Harness/testdata, or
	// Harness/internal/engine -> Harness/testdata.
	for _, rel := range []string{"../../../../testdata", "../../testdata"} {
		if _, err := os.Stat(filepath.Join(rel, "parity")); err == nil {
			return rel
		}
	}
	return "../../testdata"
}()

func mockPath(t testing.TB, variant string) string {
	p := filepath.Join(testdata, "models", variant, "model.safetensors")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("%s not present", p)
	}
	return p
}

func testDevice() string {
	if os.Getenv("VAKT_TEST_DEVICE") == "cpu" {
		return "cpu"
	}
	return "auto"
}

type fixtures struct {
	ModelSHA string `json:"model_sha256"`
	Samples  []struct {
		Name     string    `json:"name"`
		IDs      []int32   `json:"ids"`
		Severity float64   `json:"severity"`
		Families []float64 `json:"families"`
		Pooled   []float64 `json:"pooled"`
	} `json:"samples"`
}

func loadFixtures(t testing.TB) *fixtures {
	b, err := os.ReadFile(filepath.Join(testdata, "parity", "fixtures.json"))
	if err != nil {
		t.Skip("fixtures.json not present")
	}
	var f fixtures
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	return &f
}

var sharedEngine = map[string]*mlxEngine{}

func openMock(t testing.TB, variant, prec string) *mlxEngine {
	key := variant + "/" + prec + "/" + testDevice() + "/" + os.Getenv("VAKT_DELTANET")
	if e := sharedEngine[key]; e != nil {
		return e
	}
	e, err := Open(Options{ModelPath: mockPath(t, variant), Precision: prec, Device: testDevice()})
	if err != nil {
		t.Fatal(err)
	}
	sharedEngine[key] = e.(*mlxEngine)
	return e.(*mlxEngine)
}

func cosine(a []float32, b []float64) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * b[i]
		aa += float64(a[i]) * float64(a[i])
		bb += b[i] * b[i]
	}
	return ab / math.Sqrt(aa*bb)
}

// TestParityLayers compares the final-norm hidden states for the layer-dump
// sample; on failure it reports the max error, which points at the layer to
// bisect with testdata/parity/layers.safetensors.
func TestParityLayers(t *testing.T) {
	e := openMock(t, "mock-dom-0.8b", "fp32")
	p := filepath.Join(testdata, "parity", "layers.safetensors")
	if _, err := os.Stat(p); err != nil {
		t.Skip("layers.safetensors not present")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	s := mlx.CPU()
	defer s.Free()
	x := mlx.NewCtx(s)
	defer x.Free()
	ref, err := x.LoadSafetensors(p)
	if err != nil {
		t.Fatal(err)
	}
	idsF, err := x.Float32s(ref["input_ids"])
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]int32, len(idsF))
	for i, v := range idsF {
		ids[i] = int32(v)
	}
	want, err := x.Float32s(ref["norm"])
	if err != nil {
		t.Fatal(err)
	}
	pooled, got, err := e.pooledForTest(ids)
	if err != nil {
		t.Fatal(err)
	}
	var maxErr float64
	for i := range got {
		maxErr = math.Max(maxErr, math.Abs(float64(got[i]-want[i])))
	}
	wp, _ := x.Float32s(ref["pooled"])
	wp64 := make([]float64, len(wp))
	for i, v := range wp {
		wp64[i] = float64(v)
	}
	c := cosine(pooled, wp64)
	t.Logf("final norm max|Δ| = %.3g, pooled cosine = %.8f (T=%d)", maxErr, c, len(ids))
	if c < 0.9999 {
		t.Fatalf("pooled cosine %.6f < 0.9999", c)
	}
}

// TestParityFixtures is the release gate: every fixture's severity and
// family probabilities within 1e-3 (fp32) and pooled cosine > 0.9999.
func TestParityFixtures(t *testing.T) {
	f := loadFixtures(t)
	e := openMock(t, "mock-dom-0.8b", "fp32")
	if e.info.ModelSHA != f.ModelSHA {
		t.Fatalf("model sha %s != fixtures %s", e.info.ModelSHA, f.ModelSHA)
	}
	var maxS, maxP float64
	minCos := 1.0
	for _, s := range f.Samples {
		if testing.Short() && len(s.IDs) > 2048 {
			continue
		}
		got, err := e.Score(context.Background(), [][]int32{s.IDs})
		if err != nil {
			t.Fatalf("%s: %v", s.Name, err)
		}
		ds := math.Abs(float64(got[0].Severity) - s.Severity)
		maxS = math.Max(maxS, ds)
		for j := range core.NumFamilies {
			maxP = math.Max(maxP, math.Abs(float64(got[0].Families[j])-s.Families[j]))
		}
		if len(s.IDs) <= 4096 {
			pooled, _, err := e.pooledForTest(s.IDs)
			if err != nil {
				t.Fatal(err)
			}
			minCos = math.Min(minCos, cosine(pooled, s.Pooled))
		}
		if ds > 1e-3 {
			t.Errorf("%s (T=%d): severity %.6f want %.6f", s.Name, len(s.IDs), got[0].Severity, s.Severity)
		}
	}
	t.Logf("fp32 parity over %d samples: max|Δs| = %.3g, max|Δp| = %.3g, min cosine = %.8f", len(f.Samples), maxS, maxP, minCos)
	if maxP > 1e-3 {
		t.Errorf("max family |Δ| %.3g > 1e-3", maxP)
	}
	if minCos < 0.9999 {
		t.Errorf("min pooled cosine %.6f < 0.9999", minCos)
	}
}

// TestBatchInvariance: a sequence scored alone equals the same sequence in a
// padded mixed-length batch.
func TestBatchInvariance(t *testing.T) {
	f := loadFixtures(t)
	e := openMock(t, "mock-dom-0.8b", "fp32")
	var batch [][]int32
	for _, s := range f.Samples {
		if len(s.IDs) < 600 && len(batch) < 6 {
			batch = append(batch, s.IDs)
		}
	}
	together, err := e.Score(context.Background(), batch)
	if err != nil {
		t.Fatal(err)
	}
	for i, ids := range batch {
		alone, err := e.Score(context.Background(), [][]int32{ids})
		if err != nil {
			t.Fatal(err)
		}
		if d := math.Abs(float64(alone[0].Severity - together[i].Severity)); d > 1e-4 {
			t.Errorf("seq %d (T=%d): batch vs alone |Δs| = %.3g", i, len(ids), d)
		}
	}
}

// TestDeltaKernelMatchesScan checks the Metal kernel against the portable
// scan on real activations.
func TestDeltaKernelMatchesScan(t *testing.T) {
	if testDevice() == "cpu" || !gpuOK() {
		t.Skip("Metal kernel needs the GPU")
	}
	f := loadFixtures(t)
	ek := openMock(t, "mock-dom-0.8b", "fp32")
	t.Setenv("VAKT_DELTANET", "scan")
	es := openMock(t, "mock-dom-0.8b", "fp32")
	if es.m.delta != deltaScan || ek.m.delta != deltaKernel {
		t.Fatal("engines did not pick the expected delta modes")
	}
	for _, s := range f.Samples[:8] {
		a, err := ek.Score(context.Background(), [][]int32{s.IDs})
		if err != nil {
			t.Fatal(err)
		}
		b, err := es.Score(context.Background(), [][]int32{s.IDs})
		if err != nil {
			t.Fatal(err)
		}
		if d := math.Abs(float64(a[0].Severity - b[0].Severity)); d > 1e-4 {
			t.Errorf("%s: kernel vs scan |Δs| = %.3g", s.Name, d)
		}
	}
}

// TestParityBF16 loads the bf16-backbone mock and reports its drift from the
// fp32 reference (looser bound).
func TestParityBF16(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	f := loadFixtures(t)
	e := openMock(t, "mock-dom-0.8b-bf16", "fp32")
	var maxS float64
	for _, s := range f.Samples {
		if len(s.IDs) > 2048 {
			continue
		}
		got, err := e.Score(context.Background(), [][]int32{s.IDs})
		if err != nil {
			t.Fatal(err)
		}
		maxS = math.Max(maxS, math.Abs(float64(got[0].Severity)-s.Severity))
	}
	t.Logf("bf16 weights vs fp32 reference: max|Δs| = %.3g", maxS)
	if maxS > 2e-2 {
		t.Errorf("bf16 max|Δs| %.3g > 2e-2", maxS)
	}
}

func TestRejectsBadInput(t *testing.T) {
	e := openMock(t, "mock-dom-0.8b", "fp32")
	for _, b := range [][][]int32{{{}}, {make([]int32, core.MaxTokens+1)}, {{-1}}, {{core.VocabSize}}} {
		if _, err := e.Score(context.Background(), b); err == nil {
			t.Errorf("accepted bad batch of len %d", len(b[0]))
		}
	}
}

func BenchmarkScore(b *testing.B) {
	e := openMock(b, "mock-dom-0.8b", "fp32")
	for _, c := range []struct{ T, B int }{{512, 16}, {2048, 4}, {16384, 1}} {
		batch := make([][]int32, c.B)
		for i := range batch {
			batch[i] = make([]int32, c.T)
			for j := range batch[i] {
				batch[i][j] = int32((i*7919 + j*104729) % 150000)
			}
		}
		b.Run(func() string { return "T" + itoa(c.T) + "xB" + itoa(c.B) }(), func(b *testing.B) {
			if _, err := e.Score(context.Background(), batch); err != nil { // warm-up (kernel JIT)
				b.Fatal(err)
			}
			b.ResetTimer()
			for range b.N {
				if _, err := e.Score(context.Background(), batch); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(c.T*c.B*b.N)/b.Elapsed().Seconds(), "tokens/s")
		})
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d []byte
	for n > 0 {
		d = append([]byte{byte('0' + n%10)}, d...)
		n /= 10
	}
	return string(d)
}
