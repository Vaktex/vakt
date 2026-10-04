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
	0.3066038843350505,   // access_control
	0.2928365884080407,   // authentication
	0.2845438533372397,   // authorization
	0.22084363956391176,  // communication_security
	0.3395179034181917,   // credentials_and_secrets
	0.17693129806181332,  // cryptographic_issues
	0.21428010703490563,  // data_neutralization
	0.14725222836749954,  // data_processing
	0.11061918228642885,  // error_handling
	0.17032466822271736,  // file_and_path
	0.12121400925811497,  // initialization_and_cleanup
	0.2832721713234602,   // memory_safety
	0.24380708177967444,  // resource_management
	0.14046872348128342,  // serialization_and_parsing
	0.31703013639211025,  // session_management
	0.05352505482257432,  // synchronization_and_concurrency
	0.3382818172097932,   // web_security
	0.07727904300183096,  // other
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
