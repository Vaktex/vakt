package ast

/*
#include <libproc.h>
#include <sys/resource.h>

static long long vakt_rss(int pid) {
	// Physical footprint counts compressed and swapped-out dirty memory,
	// which resident size misses once macOS starts compressing.
	struct rusage_info_v4 ri;
	if (proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&ri) == 0) {
		return (long long)ri.ri_phys_footprint;
	}
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

// limitSelf applies the worker's own limits. macOS enforces neither
// RLIMIT_AS/RLIMIT_DATA nor, reliably, RLIMIT_CPU, so memory and time are
// policed by the parent, and a worker whose parent dies exits on its own
// (MaybeServeWorker watches the parent pid). The CPU limit is kept for
// platforms that do honour it.
func limitSelf() {
	_ = syscall.Setrlimit(syscall.RLIMIT_CPU, &syscall.Rlimit{Cur: 600, Max: 600})
}
