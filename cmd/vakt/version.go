package main

import (
	"fmt"
	"runtime"

	"github.com/spf13/cobra"

	"github.com/vaktex/vakt/internal/brand"
)

func versionString() string {
	// Line 1 is a machine-readable stamp; scripts/audit.sh checks it in
	// release builds, so keep the key=value format stable.
	return fmt.Sprintf("%s (%s) version=%s commit=%s backend=%s\nmodel    %s (%s)\ngo       %s %s/%s",
		brand.Product, brand.Binary, brand.Version, brand.Commit, brand.Backend,
		brand.ModelRepo, brand.ModelFile, runtime.Version(), runtime.GOOS, runtime.GOARCH)
}

func newVersionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print the version, commit, backend and model",
		Args:  cobra.NoArgs,
		Run: func(cmd *cobra.Command, _ []string) {
			fmt.Fprintln(cmd.OutOrStdout(), versionString())
		},
	}
}
