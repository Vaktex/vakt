package ast

import (
	"bytes"
	"os"
	"strconv"
	"syscall"
)

// processRSS returns a process's resident set size in bytes (VmRSS).
func processRSS(pid int) (int64, bool) {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/status")
	if err != nil {
		return 0, false
	}
	i := bytes.Index(b, []byte("VmRSS:"))
	if i < 0 {
		return 0, false
	}
	f := bytes.Fields(b[i+6:])
	if len(f) < 1 {
		return 0, false
	}
	kb, err := strconv.ParseInt(string(f[0]), 10, 64)
	if err != nil {
		return 0, false
	}
	return kb << 10, true
}

// limitSelf applies the worker's own limits. On Linux RLIMIT_AS makes an
// over-budget allocation fail (tree-sitter then aborts, which the parent
// contains); the budget is doubled over the RSS limit because the Go
// runtime reserves address space it never touches. RLIMIT_CPU is a
// backstop for a parent that died.
func limitSelf() {
	_ = syscall.Setrlimit(syscall.RLIMIT_CPU, &syscall.Rlimit{Cur: 600, Max: 600})
	if mem, err := strconv.ParseInt(os.Getenv("VAKT_AST_WORKER_MEM"), 10, 64); err == nil && mem > 0 {
		as := uint64(mem)*2 + 4<<30 // #nosec G115 -- positive
		_ = syscall.Setrlimit(syscall.RLIMIT_AS, &syscall.Rlimit{Cur: as, Max: as})
	}
}
