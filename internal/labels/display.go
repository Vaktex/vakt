package labels

import "strings"

// Issue is the user-facing name for a CWE family, with an optional CWE id.
type Issue struct {
	Title string // plain-language issue, e.g. "SQL injection"
	CWE   string // e.g. "CWE-89"; empty when no single id fits
}

// familyIssues maps each model family to the most representative issue name.
// The model only predicts a family, not a specific CWE, so these are the
// canonical labels for terminal output. Keep one entry per Families value.
var familyIssues = map[string]Issue{
	"access_control":                  {Title: "Access control flaw", CWE: "CWE-284"},
	"authentication":                  {Title: "Broken authentication", CWE: "CWE-287"},
	"authorization":                   {Title: "Missing authz check", CWE: "CWE-862"},
	"communication_security":          {Title: "Insecure communication", CWE: "CWE-319"},
	"credentials_and_secrets":         {Title: "Hardcoded secret", CWE: "CWE-798"},
	"cryptographic_issues":            {Title: "Weak cryptography", CWE: "CWE-327"},
	"data_neutralization":             {Title: "SQL injection", CWE: "CWE-89"},
	"data_processing":                 {Title: "Unsafe data processing", CWE: "CWE-20"},
	"error_handling":                  {Title: "Information leak in errors", CWE: "CWE-209"},
	"file_and_path":                   {Title: "Path traversal", CWE: "CWE-22"},
	"initialization_and_cleanup":      {Title: "Unsafe init or cleanup", CWE: "CWE-665"},
	"memory_safety":                   {Title: "Buffer overflow", CWE: "CWE-120"},
	"resource_management":             {Title: "Resource leak", CWE: "CWE-400"},
	"serialization_and_parsing":       {Title: "Unsafe deserialization", CWE: "CWE-502"},
	"session_management":              {Title: "Session fixation", CWE: "CWE-384"},
	"synchronization_and_concurrency": {Title: "Race condition", CWE: "CWE-362"},
	"web_security":                    {Title: "Cross-site scripting", CWE: "CWE-79"},
	"other":                           {Title: "Security issue", CWE: ""},
}

// IssueFor returns the display issue for a family. Unknown families fall back
// to a title-cased family name with no CWE.
func IssueFor(family string) Issue {
	if iss, ok := familyIssues[family]; ok {
		return iss
	}
	return Issue{Title: titleCaseFamily(family)}
}

// DisplayTitle is Title, plus " (CWE-…)" when a CWE id is known.
func (iss Issue) DisplayTitle() string {
	if iss.CWE == "" {
		return iss.Title
	}
	return iss.Title + " (" + iss.CWE + ")"
}

func titleCaseFamily(family string) string {
	if family == "" {
		return "Unknown issue"
	}
	parts := strings.Split(family, "_")
	for i, p := range parts {
		if p == "" {
			continue
		}
		parts[i] = strings.ToUpper(p[:1]) + p[1:]
	}
	return strings.Join(parts, " ")
}
