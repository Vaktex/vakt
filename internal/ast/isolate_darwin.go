package ast

/*
#include <libproc.h>
#include <sys/resource.h>

static long long vakt_rss(int pid) {
	struct proc_taskinfo ti;
	int n = proc_pidinfo(pid, PROC_PIDTASKINFO, 0, &ti, sizeof ti);
	if (n != (int)sizeof ti) return -1;
	return (long long)ti.pti_resident_size;
}
*/
import "C"

import (
	"math"
	"syscall"
)

// processRSS returns a process's resident set size in bytes.
func processRSS(pid int) (int64, bool) {
	if pid <= 0 || pid > math.MaxInt32 {
		return 0, false
	}
	v := int64(C.vakt_rss(C.int(pid))) // #nosec G115 -- range checked above
	return v, v >= 0
}

// limitSelf applies the worker's own limits. macOS does not enforce
// RLIMIT_AS/RLIMIT_DATA, so memory is policed by the parent (processRSS);
// the CPU limit is a backstop for a parent that died.
func limitSelf() {
	_ = syscall.Setrlimit(syscall.RLIMIT_CPU, &syscall.Rlimit{Cur: 600, Max: 600})
}
