package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/mcp"
	"github.com/vaktex/vakt/internal/report"
)

func newMCPCmd() *cobra.Command {
	var rootDir string
	cmd := &cobra.Command{
		Use:   "mcp",
		Short: "Serve " + brand.Product + " to a coding agent over the Model Context Protocol",
		Long: "mcp turns " + brand.Binary + " into an MCP server on stdin/stdout, so an agent can\n" +
			"scan code and read the findings as structured data instead of parsing the\n" +
			"terminal report.\n\n" +
			"It exposes four tools: scan (run a scan), findings (list what it flagged),\n" +
			"finding (one result with its source and CWE) and families (the taxonomy).\n\n" +
			"The agent passes the project directory (dir) on every call, and each call\n" +
			"reads and writes only inside it. --root limits which directories the agent\n" +
			"may pass, for example to your home directory or one projects folder.\n" +
			"Scanning stays local: no code leaves the machine.\n\n" +
			"Register it with an agent, for example in .cursor/mcp.json or\n" +
			"claude_desktop_config.json:\n\n" +
			"  {\n" +
			"    \"mcpServers\": {\n" +
			"      \"" + brand.Binary + "\": {\n" +
			"        \"command\": \"" + brand.Binary + "\",\n" +
			"        \"args\": [\"mcp\"]\n" +
			"      }\n" +
			"    }\n" +
			"  }",
		Example: "  " + brand.Binary + " mcp\n" +
			"  " + brand.Binary + " mcp --root ~\n" +
			"  " + brand.Binary + " mcp --root ~/work",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			root := ""
			if rootDir != "" {
				r, err := resolveRoot(rootDir)
				if err != nil {
					return &exitCodeError{code: exitError, err: err}
				}
				root = r
			}
			stderr := cmd.ErrOrStderr()
			srv := mcp.New(mcp.Tools(mcp.Deps{
				Scan:       mcpScan(stderr),
				ReportName: defaultOut,
				Root:       root,
			}), func(format string, args ...any) {
				// Diagnostics go to stderr: stdout carries framed JSON only,
				// and a stray byte there breaks the session.
				fmt.Fprintf(stderr, format+"\n", args...)
			})

			// Announce on stderr so a user running this by hand sees that it
			// is a server waiting on stdin, not a hung command.
			scope := "any directory"
			if root != "" {
				scope = "directories under " + root
			}
			fmt.Fprintf(stderr, "%s mcp: serving %s on stdio (%s)\n", brand.Binary, brand.Product, scope)
			if err := srv.Serve(cmd.Context(), cmd.InOrStdin(), cmd.OutOrStdout()); err != nil {
				return &exitCodeError{code: exitError, err: err}
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&rootDir, "root", "", "only let the agent work in directories under this one (default: any directory)")
	return cmd
}

// resolveRoot returns the absolute, symlink-resolved --root directory.
func resolveRoot(dir string) (string, error) {
	if strings.HasPrefix(dir, "~") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		dir = filepath.Join(home, strings.TrimPrefix(dir, "~"))
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	st, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("--root %s: %w", dir, err)
	}
	if !st.IsDir() {
		return "", fmt.Errorf("--root %s: not a directory", dir)
	}
	if real, err := filepath.EvalSymlinks(abs); err == nil {
		abs = real
	}
	return abs, nil
}

// mcpScan adapts the CLI's scan to the MCP tool.
//
// runScan is only bound in a build with a native engine; without one this
// returns the same explanatory error the patrol command does, which the tool
// passes to the agent.
func mcpScan(stderr io.Writer) mcp.ScanFunc {
	return func(ctx context.Context, req mcp.ScanRequest) (*report.Report, error) {
		o := ScanOptions{
			Root:         req.Root,
			Format:       "json",
			Out:          req.Out,
			Top:          defaultTop,
			Quiet:        true, // no progress bar: stdout is the protocol stream
			Threshold:    req.Threshold,
			MinTokens:    defaultMinTokens,
			BatchTokens:  defaultBatchTokens,
			Jobs:         runtime.NumCPU(),
			Include:      req.Include,
			Exclude:      req.Exclude,
			MaxFileBytes: defaultMaxFileBytes,
			TopLevel:     req.TopLevel,
			Precision:    defaultPrecision,
			Device:       "auto",
		}
		spec, err := parseModelSpec(defaultModel, "")
		if err != nil {
			return nil, err
		}
		o.Model, o.ModelPath, o.ModelRepo, o.ModelRevision = defaultModel, spec.Path, spec.Repo, spec.Revision

		prog := &report.Progress{}
		rep, err := runScan(ctx, o, prog)
		if err != nil {
			return nil, err
		}
		// Persist the report so the read-only tools, and the user's own
		// `vakt show`, can reach the same findings afterwards.
		if req.Out != "" {
			if err := report.WriteFile(req.Out, rep); err != nil {
				fmt.Fprintf(stderr, "%s mcp: could not write %s: %v\n", brand.Binary, req.Out, err)
			}
		}
		return rep, nil
	}
}
