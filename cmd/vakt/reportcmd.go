package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/vaktex/vakt/internal/report"
)

func newReportCmd() *cobra.Command {
	var (
		top           int
		threshold     float64
		minConfidence float64
		families      []string
		format        string
		noColor       bool
		quiet         bool
		verbose       bool
	)
	cmd := &cobra.Command{
		Use:   "report <report.json>",
		Short: "Re-render a saved JSON report",
		Long: "report reads a JSON report written by patrol and prints it again, optionally\n" +
			"with a different threshold or only some CWE families. Use - for stdin.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if top < 1 {
				return errors.New("--top must be at least 1")
			}
			if cmd.Flags().Changed("threshold") && !(threshold > 0 && threshold <= 1) {
				return errors.New("--threshold must be in (0, 1]")
			}
			if !cmd.Flags().Changed("threshold") {
				threshold = 0
			}
			if err := report.ValidateFamilies(families); err != nil {
				return fmt.Errorf("--family: %w", err)
			}
			in := cmd.InOrStdin()
			if args[0] != "-" {
				f, err := os.Open(args[0])
				if err != nil {
					return err
				}
				defer f.Close()
				in = f
			}
			rep, err := report.ReadJSON(in)
			if err != nil {
				return fmt.Errorf("%s: %w", args[0], err)
			}
			if cmd.Flags().Changed("min-confidence") {
				if !(minConfidence >= 0 && minConfidence <= 1) {
					return errors.New("--min-confidence must be in [0, 1]")
				}
				report.ApplyConfidence(rep, minConfidence)
			}
			rep = report.Filter(rep, threshold, families)
			out := cmd.OutOrStdout()
			switch format {
			case "json":
				return report.WriteJSON(out, rep)
			case "pretty":
				return report.Pretty(out, rep, report.PrettyOptions{
					Top: top, Families: families, Quiet: quiet, Verbose: verbose,
					Color: useColor(out, noColor), Hyperlinks: useHyperlinks(out),
					Width: termWidth(out),
				})
			default:
				return fmt.Errorf("--format must be pretty or json (got %q)", format)
			}
		},
	}
	f := cmd.Flags()
	f.IntVar(&top, "top", defaultTop, "number of findings to print")
	f.Float64Var(&threshold, "threshold", defaultThreshold, "flag functions with severity >= this (default: the report's)")
	f.Float64Var(&minConfidence, "min-confidence", 0, "minimum selected family score (default: the report's; 0 disables)")
	f.StringArrayVar(&families, "family", nil, "only show findings whose top family is this (repeatable)")
	f.StringVar(&format, "format", "pretty", "output format: pretty or json")
	f.BoolVar(&noColor, "no-color", false, "disable colour (also NO_COLOR)")
	f.BoolVarP(&quiet, "quiet", "q", false, "print only the summary lines")
	f.BoolVarP(&verbose, "verbose", "v", false, "show model/backend stats and per-finding scores")
	return cmd
}
