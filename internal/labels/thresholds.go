package labels

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/vaktex/vakt/internal/core"
)

// DefaultThresholds are the per-family cut-offs published with the pinned
// weights (thresholds.json at brand.ModelCommit): F1-optimal on held-out
// validation data, in head order. Rare families sit well below 0.5, so a
// single 0.5 cut-off would leave several of them never firing.
var DefaultThresholds = [core.NumFamilies]float64{
	0.2944398275106302,   // access_control
	0.24805404387057053,  // authentication
	0.21452892647728386,  // authorization
	0.20471088261263026,  // communication_security
	0.11190292114179588,  // credentials_and_secrets
	0.1970535984366234,   // cryptographic_issues
	0.39922511987750714,  // data_neutralization
	0.18418992012083565,  // data_processing
	0.03141314444955876,  // error_handling
	0.2276440516540958,   // file_and_path
	0.06971980836669046,  // initialization_and_cleanup
	0.2473448405165064,   // memory_safety
	0.23279998674652683,  // resource_management
	0.25443853571770425,  // serialization_and_parsing
	0.16634039820258426,  // session_management
	0.02862530178199642,  // synchronization_and_concurrency
	0.19506198965689228,  // web_security
	0.049114563030317776, // other
}

// ParseThresholds reads a thresholds.json document ({"families": {name: cut}}).
// Every family must be present with a cut-off in (0, 1).
func ParseThresholds(data []byte) ([core.NumFamilies]float64, error) {
	var out [core.NumFamilies]float64
	var doc struct {
		Families map[string]float64 `json:"families"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return out, fmt.Errorf("thresholds: %w", err)
	}
	for i, name := range Families {
		v, ok := doc.Families[name]
		if !ok {
			return out, fmt.Errorf("thresholds: family %q missing", name)
		}
		if math.IsNaN(v) || v <= 0 || v >= 1 {
			return out, fmt.Errorf("thresholds: %q cut-off %v is outside (0, 1)", name, v)
		}
		out[i] = v
	}
	if len(doc.Families) != len(Families) {
		return out, fmt.Errorf("thresholds: %d families, want %d", len(doc.Families), len(Families))
	}
	return out, nil
}

// Top returns the family that most clears its cut-off (the largest p/t) and
// its probability. Families differ widely in base rate, so the raw largest
// probability over-reports common families; the card routes on the
// per-family cut-offs, and so does this.
func Top(p [core.NumFamilies]float64, t [core.NumFamilies]float64) (int, float64) {
	best, bestRatio := 0, -1.0
	for i := range p {
		if t[i] <= 0 {
			continue
		}
		if r := p[i] / t[i]; r > bestRatio {
			best, bestRatio = i, r
		}
	}
	return best, p[best]
}
