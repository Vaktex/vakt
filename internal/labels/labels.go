// Package labels mirrors experiment/labels.py. The order is the order of the
// auxiliary head's outputs and must never change independently of the model.
package labels

import "github.com/vaktex/vakt/internal/core"

// Families are CWE_FAMILY_NAMES: the CWE_FAMILIES keys in insertion order,
// followed by "other".
var Families = [core.NumFamilies]string{
	"access_control",
	"authentication",
	"authorization",
	"communication_security",
	"credentials_and_secrets",
	"cryptographic_issues",
	"data_neutralization",
	"data_processing",
	"error_handling",
	"file_and_path",
	"initialization_and_cleanup",
	"memory_safety",
	"resource_management",
	"serialization_and_parsing",
	"session_management",
	"synchronization_and_concurrency",
	"web_security",
	"other",
}

// Index returns the head index of a family name, or -1.
func Index(name string) int {
	for i, f := range Families {
		if f == name {
			return i
		}
	}
	return -1
}
