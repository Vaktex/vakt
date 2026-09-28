package labels

import "testing"

func TestIssueFor(t *testing.T) {
	iss := IssueFor("data_neutralization")
	if iss.Title != "SQL injection" || iss.CWE != "CWE-89" {
		t.Fatalf("%+v", iss)
	}
	if got := iss.DisplayTitle(); got != "SQL injection (CWE-89)" {
		t.Fatalf("DisplayTitle = %q", got)
	}
	iss = IssueFor("totally_unknown_family")
	if iss.Title != "Totally Unknown Family" || iss.CWE != "" {
		t.Fatalf("fallback %+v", iss)
	}
	for _, f := range Families {
		if IssueFor(f).Title == "" {
			t.Errorf("%s: empty title", f)
		}
	}
}
