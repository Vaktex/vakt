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
		top       int
		threshold float64
		families  []string
		format    string
		noColor   bool
		quiet     bool
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
			rep = report.Filter(rep, threshold, families)
			out := cmd.OutOrStdout()
			switch format {
			case "json":
				return report.WriteJSON(out, rep)
			case "pretty":
				return report.Pretty(out, rep, report.PrettyOptions{
					Top: top, Families: families, Quiet: quiet,
					Color: useColor(out, noColor), Width: termWidth(out),
				})
			default:
				return fmt.Errorf("--format must be pretty or json (got %q)", format)
			}
		},
	}
	f := cmd.Flags()
	f.IntVar(&top, "top", defaultTop, "number of findings in the top table")
	f.Float64Var(&threshold, "threshold", defaultThreshold, "flag units with severity >= this (default: the report's)")
	f.StringArrayVar(&families, "family", nil, "only show units whose top family is this (repeatable)")
	f.StringVar(&format, "format", "pretty", "output format: pretty or json")
	f.BoolVar(&noColor, "no-color", false, "disable colour (also NO_COLOR)")
	f.BoolVarP(&quiet, "quiet", "q", false, "print only the summary line")
	return cmd
}
