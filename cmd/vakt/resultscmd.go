package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/hub"
	"github.com/vaktex/vakt/internal/report"
	"github.com/vaktex/vakt/internal/results"
)

func newResultsCmd() *cobra.Command {
	var threshold, minConfidence float64
	var maxFileBytes int64
	cmd := &cobra.Command{
		Use:   "results [path]",
		Short: "Print stored results for files unchanged since their last scan",
		Long: "results reads what `" + brand.Binary + " patrol --save-results` and `" + brand.Binary + " serve` stored in\n" +
			results.Dir + " under a directory and prints it as a JSON report, without loading the\n" +
			"model. Only files whose contents still hash to what was scored are included;\n" +
			"changed or deleted files are left out until they are scanned again.\n\n" +
			"Results are signed with a key in your " + brand.Binary + " cache, so results a repository\n" +
			"ships in its own " + results.Dir + " directory are ignored.",
		Example: "  " + brand.Binary + " results .\n" +
			"  " + brand.Binary + " results src | jq '.units | length'",
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			root := "."
			if len(args) == 1 {
				root = args[0]
			}
			switch {
			case !(threshold > 0 && threshold <= 1):
				return errors.New("--threshold must be in (0, 1]")
			case !(minConfidence >= 0 && minConfidence <= 1):
				return errors.New("--min-confidence must be in [0, 1]")
			}
			abs, err := filepath.Abs(root)
			if err != nil {
				return err
			}
			if st, err := os.Stat(abs); err != nil || !st.IsDir() {
				return fmt.Errorf("cannot read results for %s: not a directory", root)
			}
			key, err := results.Key(hub.RootDir())
			if err != nil {
				return err
			}
			d, err := results.Load(abs, key)
			if err != nil {
				return err
			}
			units, err := results.Fresh(abs, d, maxFileBytes)
			if err != nil {
				return err
			}
			rep := &report.Report{
				SchemaVersion: report.SchemaVersion,
				Tool:          report.Tool{Name: brand.Product, Binary: brand.Binary, Version: brand.Version, Commit: brand.Commit},
				Model:         d.Model,
				Units:         units,
				Scan:          report.Scan{Root: abs, Units: len(units)},
				Summary:       report.Summary{Threshold: threshold},
			}
			report.ApplyConfidence(rep, minConfidence)
			rep.Scan.Files = len(rep.Files)
			return report.WriteJSON(cmd.OutOrStdout(), rep)
		},
	}
	f := cmd.Flags()
	f.Float64Var(&threshold, "threshold", defaultThreshold, "flag functions with severity >= this score")
	f.Float64Var(&minConfidence, "min-confidence", defaultMinConfidence, "minimum selected family score for findings")
	f.Int64Var(&maxFileBytes, "max-file-bytes", defaultMaxFileBytes, "skip files larger than this")
	return cmd
}
