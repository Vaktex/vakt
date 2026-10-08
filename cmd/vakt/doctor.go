package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/hub"
)

// doctorBench runs a short engine benchmark and returns a one-line result.
// The integrator sets it in the engine-linked build; nil or the default
// means no engine is linked.
var doctorBench = func(ctx context.Context) (string, error) {
	return "", errEngineNotLinked
}

var errEngineNotLinked = errors.New("engine not linked")

// minDriver is the NVIDIA driver major version CUDA 13 needs.
const minDriver = 580

// cudaLibs are the shared libraries the CUDA build loads.
var cudaLibs = []string{"libcublas.so.13", "libcublasLt.so.13", "libnvrtc.so.13", "libcudnn.so.9"}

// commandOutput runs a fixed diagnostic command with a timeout. Tests replace it.
var commandOutput = func(ctx context.Context, name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output() // #nosec G204 -- fixed command names and args
	return strings.TrimSpace(string(out)), err
}

// lookPath finds a binary. Tests replace it.
var lookPath = exec.LookPath

type doctor struct {
	w     io.Writer
	warns int
}

func (d *doctor) row(k, v string)  { fmt.Fprintf(d.w, "  %-16s %s\n", k, v) }
func (d *doctor) section(s string) { fmt.Fprintf(d.w, "\n%s\n", s) }
func (d *doctor) warn(k, v string) { d.warns++; d.row(k, "! "+v) }

func newDoctorCmd() *cobra.Command {
	var noBench, strict bool
	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Check this machine, the engine and the model cache",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			warns, err := runDoctorCount(cmd.Context(), cmd.OutOrStdout(), runtime.GOOS, !noBench)
			if err == nil && strict && warns > 0 {
				return &exitCodeError{code: exitError} // message already printed
			}
			return err
		},
	}
	cmd.Flags().BoolVar(&noBench, "no-bench", false, "skip the engine benchmark")
	cmd.Flags().BoolVar(&strict, "strict", false, "exit 1 if any check needs attention (for scripts and CI)")
	return cmd
}

func runDoctor(ctx context.Context, w io.Writer, goos string, bench bool) error {
	_, err := runDoctorCount(ctx, w, goos, bench)
	return err
}

// runDoctorCount runs the checks and returns how many need attention.
func runDoctorCount(ctx context.Context, w io.Writer, goos string, bench bool) (int, error) {
	d := &doctor{w: w}
	fmt.Fprintf(w, "%s %s doctor\n", brand.Product, brand.Version)

	d.section("System")
	d.row("os/arch", goos+"/"+runtime.GOARCH)
	d.row("cpu", cpuName(ctx, goos))
	d.row("cpus", strconv.Itoa(runtime.NumCPU()))

	d.section("Engine")
	switch brand.Backend {
	case "none", "", "unknown":
		d.warn("backend", brand.Backend+" (this build cannot scan; install a release build with Metal or CUDA)")
	default:
		d.row("backend", brand.Backend)
	}
	if goos == "linux" {
		checkNvidia(ctx, d)
	}
	if bench {
		res, err := doctorBench(ctx)
		switch {
		case errors.Is(err, errEngineNotLinked):
			d.row("benchmark", "engine not linked")
		case err != nil:
			d.warn("benchmark", "failed: "+oneLine(err.Error()))
		default:
			d.row("benchmark", oneLine(res))
		}
	}

	d.section("Model")
	d.row("repo", brand.ModelRepo)
	d.row("cache", hub.CacheDir())
	d.row("revision", brand.ModelCommit)
	// Check the exact pinned commit vakt uses (summon/patrol resolve
	// brand.ModelCommit), not refs/main: a stale `main` ref can point at a
	// different snapshot and make doctor report "cached" while summon 404s.
	if path, sha, err := hub.Cached(hub.Options{Repo: brand.ModelRepo, Revision: brand.ModelCommit}); err == nil {
		d.row("status", "cached")
		d.row("path", path)
		d.row("sha256", sha)
	} else {
		d.warn("status", "not cached (run `"+brand.Binary+" summon`)")
	}
	if _, ok := hub.ResolveToken(hub.Options{}); ok {
		src := "token file"
		if os.Getenv("HF_TOKEN") != "" {
			src = "HF_TOKEN"
		} else if os.Getenv("HUGGING_FACE_HUB_TOKEN") != "" {
			src = "HUGGING_FACE_HUB_TOKEN"
		}
		d.row("hf token", "yes ("+src+")")
	} else {
		d.row("hf token", "no (needed to download the gated model; set HF_TOKEN or run `hf auth login`)")
	}

	fmt.Fprintln(w)
	if d.warns == 0 {
		fmt.Fprintln(w, "All checks passed.")
	} else {
		fmt.Fprintf(w, "%d check%s need attention.\n", d.warns, plural(d.warns))
	}
	return d.warns, nil
}

func cpuName(ctx context.Context, goos string) string {
	switch goos {
	case "darwin":
		if s, err := commandOutput(ctx, "sysctl", "-n", "machdep.cpu.brand_string"); err == nil && s != "" {
			return oneLine(s)
		}
	case "linux":
		if f, err := os.Open("/proc/cpuinfo"); err == nil {
			defer f.Close()
			sc := bufio.NewScanner(f)
			for sc.Scan() {
				k, v, ok := strings.Cut(sc.Text(), ":")
				if ok && strings.TrimSpace(k) == "model name" {
					return oneLine(strings.TrimSpace(v))
				}
			}
		}
	}
	return "unknown"
}

func checkNvidia(ctx context.Context, d *doctor) {
	if _, err := lookPath("nvidia-smi"); err != nil {
		d.warn("nvidia-smi", "not found (no NVIDIA driver; the CUDA build needs driver >= "+strconv.Itoa(minDriver)+")")
	} else {
		d.row("nvidia-smi", "found")
		out, err := commandOutput(ctx, "nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader")
		if err != nil || out == "" {
			d.warn("gpus", "nvidia-smi failed")
		} else {
			driver := ""
			for i, line := range strings.Split(out, "\n") {
				ver, name, _ := strings.Cut(line, ",")
				ver, name = strings.TrimSpace(ver), strings.TrimSpace(name)
				if driver == "" {
					driver = ver
				}
				d.row(fmt.Sprintf("gpu %d", i), oneLine(name))
			}
			major, _ := strconv.Atoi(strings.SplitN(driver, ".", 2)[0])
			if major >= minDriver {
				d.row("driver", oneLine(driver))
			} else {
				d.warn("driver", fmt.Sprintf("%s (CUDA 13 needs >= %d; upgrade the NVIDIA driver)", oneLine(driver), minDriver))
			}
		}
	}
	ldconfig, err := commandOutput(ctx, "ldconfig", "-p")
	if err != nil {
		ldconfig, err = commandOutput(ctx, "/sbin/ldconfig", "-p")
	}
	if err != nil {
		d.warn("cuda libs", "cannot run ldconfig -p")
		return
	}
	var missing []string
	for _, lib := range cudaLibs {
		if !strings.Contains(ldconfig, lib+" ") {
			missing = append(missing, lib)
		}
	}
	if len(missing) == 0 {
		d.row("cuda libs", "found ("+strings.Join(cudaLibs, ", ")+")")
	} else {
		d.warn("cuda libs", "missing "+strings.Join(missing, ", ")+" (install CUDA 13 and cuDNN 9, or add them to the loader path)")
	}
}

// oneLine keeps diagnostic strings printable and single-line.
func oneLine(s string) string {
	return strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f || (r >= 0x80 && r < 0xa0) {
			return ' '
		}
		return r
	}, s)
}
