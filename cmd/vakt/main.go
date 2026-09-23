package main

import (
	"fmt"
	"os"

	"github.com/vaktex/vakt/internal/brand"
)

// Placeholder CLI until the report stream's cobra CLI is merged; the
// `version` output format is the contract scripts/audit.sh checks.
func main() {
	fmt.Printf("%s (%s) version=%s commit=%s backend=%s\n", brand.Product, brand.Binary, brand.Version, brand.Commit, brand.Backend)
	_ = os.Args
}
