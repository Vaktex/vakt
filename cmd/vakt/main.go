// Command vakt is the Vaktex OSS scanner: it scores every function in a
// codebase with DOM-0.8B and reports the ones that look vulnerable.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/vaktex/vakt/internal/ast"
	"github.com/vaktex/vakt/internal/brand"
)

// Exit codes.
const (
	exitOK       = 0   // scan succeeded with no findings at or above threshold
	exitFindings = 1   // scan succeeded and findings were at or above threshold
	exitError    = 2   // usage error, I/O failure, or scan failure
	exitSignals  = 130 // interrupted
)

// exitCodeError carries a specific exit code. A nil err means the message was
// already printed.
type exitCodeError struct {
	code int
	err  error
}

func (e *exitCodeError) Error() string {
	if e.err == nil {
		return fmt.Sprintf("exit %d", e.code)
	}
	return e.err.Error()
}

func (e *exitCodeError) Unwrap() error { return e.err }

func main() {
	// Parse workers are this binary re-executed; serve and exit if we are one.
	ast.MaybeServeWorker()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	code := run(ctx, os.Args[1:], os.Stdin, os.Stdout, os.Stderr)
	stop()
	os.Exit(code)
}

// run executes the CLI and returns the process exit code.
func run(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	root := newRootCmd(stdin, stdout, stderr)
	root.SetArgs(defaultToPatrol(root, args))
	err := root.ExecuteContext(ctx)
	if err == nil {
		return exitOK
	}
	// An interrupt wins over whatever error the cancellation surfaced as
	// (e.g. "scoring: context canceled"): scripts expect 130.
	if ctx.Err() != nil {
		fmt.Fprintf(stderr, "%s: interrupted\n", brand.Binary)
		return exitSignals
	}
	var ee *exitCodeError
	if errors.As(err, &ee) {
		if ee.err != nil {
			fmt.Fprintf(stderr, "%s: %v\n", brand.Binary, ee.err)
		}
		return ee.code
	}
	fmt.Fprintf(stderr, "%s: %v\n", brand.Binary, err)
	return exitError
}

// defaultToPatrol makes patrol the default command, so `vakt .` and
// `vakt --format json src/` work. Help and version flags are left alone.
func defaultToPatrol(root *cobra.Command, args []string) []string {
	if len(args) == 0 {
		return args
	}
	first := args[0]
	switch first {
	case "-h", "--help", "--version", "help", "__complete", "__completeNoDesc":
		return args
	}
	if !strings.HasPrefix(first, "-") {
		for _, c := range root.Commands() {
			if c.Name() == first || c.HasAlias(first) {
				return args
			}
		}
	}
	return append([]string{"patrol"}, args...)
}

func newRootCmd(stdin io.Reader, stdout, stderr io.Writer) *cobra.Command {
	root := &cobra.Command{
		Use:           brand.Binary + " [path]",
		Short:         brand.Product + ": scan code for likely vulnerabilities",
		Long:          brand.Product + " (" + brand.Binary + ")",
		Version:       versionString(),
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.SetIn(stdin)
	root.SetOut(stdout)
	root.SetErr(stderr)
	root.SetVersionTemplate("{{.Version}}\n")
	root.CompletionOptions.HiddenDefaultCmd = true
	root.AddCommand(
		newPatrolCmd(),
		newSummonCmd(),
		newDoctorCmd(),
		newReportCmd(),
		newShowCmd(),
		newMCPCmd(),
		newVersionCmd(),
	)
	return root
}
