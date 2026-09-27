package labels

import (
	"testing"

	"github.com/vaktex/vakt/internal/core"
)

func TestDefaultThresholdsValid(t *testing.T) {
	doc := []byte(`{"families":{`)
	for i, n := range Families {
		if i > 0 {
			doc = append(doc, ',')
		}
		doc = append(doc, []byte(`"`+n+`":0.3`)...)
	}
	doc = append(doc, []byte(`}}`)...)
	got, err := ParseThresholds(doc)
	if err != nil || got[0] != 0.3 {
		t.Fatalf("%v %v", got, err)
	}
	for i, v := range DefaultThresholds {
		if !(v > 0 && v < 1) {
			t.Errorf("%s: %v", Families[i], v)
		}
	}
	if _, err := ParseThresholds([]byte(`{"families":{"access_control":0.3}}`)); err == nil {
		t.Error("partial thresholds accepted")
	}
}

// Top picks the family furthest over its cut-off, not the largest raw
// probability: error_handling at 0.10 (cut 0.03) beats data_neutralization
// at 0.45 (cut 0.40).
func TestTopUsesCutoffs(t *testing.T) {
	var p [core.NumFamilies]float64
	p[Index("data_neutralization")] = 0.45
	p[Index("error_handling")] = 0.10
	i, prob := Top(p, DefaultThresholds)
	if Families[i] != "error_handling" || prob != 0.10 {
		t.Fatalf("top = %s %.2f", Families[i], prob)
	}
}
