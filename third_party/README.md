# third_party

Grammars that have no usable Go module upstream, vendored as local cgo
packages. Each directory keeps its upstream LICENSE; the C sources are
unmodified generated files.

| Directory | Upstream | Version | License |
| --- | --- | --- | --- |
| `tree-sitter-vba` | https://github.com/harumiWeb/tree-sitter-vba (`bindings/go`, parser.c sha256 1ec8fc0c44c49fd15da2174a4db92824c4be048c074689e8e0b8962a2ab81e0a; upstream gitignores `vba/src/parser.c`, and its Go binding lives in a nested module not tagged for `go get`) | commit a67fd2d | MIT |
| `tree-sitter-solidity` | https://github.com/JoranHonig/tree-sitter-solidity (`src`) | v1.2.13 (no Go binding upstream) | MIT |
