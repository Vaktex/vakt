package main

import (
	"errors"
	"fmt"
	"os"
	"strconv"

	"github.com/spf13/cobra"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/report"
)

func newShowCmd() *cobra.Command {
	var (
		reportPath string
		threshold  float64
		noColor    bool
	)
	cmd := &cobra.Command{
		Use:   "show <n>",
		Short: "Print full details for finding n from a report",
		Long: "show prints the full finding numbered n in a report (severity order),\n" +
			"including score, confidence, family, CWE, explanation and a code snippet.\n\n" +
			"By default it reads ./" + brand.Binary + "-report.json written by the last scan.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			n, err := strconv.Atoi(args[0])
			if err != nil || n < 1 {
				return &exitCodeError{code: exitError, err: errors.New("finding number must be a positive integer")}
			}
			if cmd.Flags().Changed("threshold") && !(threshold > 0 && threshold <= 1) {
				return &exitCodeError{code: exitError, err: errors.New("--threshold must be in (0, 1]")}
			}
			if !cmd.Flags().Changed("threshold") {
				threshold = 0
			}
			path := reportPath
			if path == "" {
				path = defaultOut
			}
			f, err := os.Open(path) // #nosec G304 -- user-supplied report path
			if err != nil {
				return &exitCodeError{code: exitError, err: fmt.Errorf("%s: %w", path, err)}
			}
			defer f.Close()
			rep, err := report.ReadJSON(f)
			if err != nil {
				return &exitCodeError{code: exitError, err: fmt.Errorf("%s: %w", path, err)}
			}
			u, err := report.FlaggedAt(rep, n, threshold)
			if err != nil {
				return &exitCodeError{code: exitError, err: err}
			}
			out := cmd.OutOrStdout()
			return report.ShowFinding(out, rep, u, report.ShowOptions{
				Color: useColor(out, noColor),
				Width: termWidth(out),
			})
		},
	}
	f := cmd.Flags()
	f.StringVarP(&reportPath, "report", "r", "", "report JSON path (default: "+defaultOut+")")
	f.Float64Var(&threshold, "threshold", defaultThreshold, "flag threshold (default: the report's)")
	f.BoolVar(&noColor, "no-color", false, "disable colour (also NO_COLOR)")
	return cmd
}
