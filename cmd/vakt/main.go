package main

import (
	"fmt"
	"os"

	"github.com/vaktex/vakt/internal/brand"
)

func main() {
	fmt.Fprintf(os.Stderr, "%s (%s) %s\n", brand.Product, brand.Binary, brand.Version)
}
