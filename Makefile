# Vaktex OSS (vakt) build.
#
#   make dev                 unobfuscated build into bin/vakt
#   make prod                garble-obfuscated release build into dist/
#   make deps                native libraries (MLX, tokenizers) for BACKEND
#   make test lint parity bench audit checksums clean help
#
# BACKEND: auto | metal | cuda | cpu | fake
#   auto  -> metal on darwin, cuda on linux when nvcc is present, else cpu
#   fake  -> no native model engine (deterministic fake scores); for CI lint
#            and pipeline development. tree-sitter still uses cgo.

SHELL := /bin/sh
.DEFAULT_GOAL := help

GO      ?= go
GARBLE  ?= $(shell command -v garble 2>/dev/null || echo $(HOME)/go/bin/garble)
GOOS    := $(shell $(GO) env GOOS)
GOARCH  := $(shell $(GO) env GOARCH)

VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT  ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo none)

BACKEND ?= auto
ifeq ($(BACKEND),auto)
  ifeq ($(GOOS),darwin)
    override BACKEND := metal
  else ifneq ($(shell command -v nvcc 2>/dev/null),)
    override BACKEND := cuda
  else
    override BACKEND := cpu
  endif
endif

ifeq ($(filter $(BACKEND),metal cuda cpu fake),)
  $(error BACKEND must be one of auto, metal, cuda, cpu, fake (got '$(BACKEND)'))
endif
ifeq ($(GOOS)-$(BACKEND),darwin-cuda)
  $(error CUDA is not available on macOS; use BACKEND=metal or cpu)
endif
ifeq ($(GOOS)-$(BACKEND),linux-metal)
  $(error Metal is macOS-only; use BACKEND=cuda or cpu)
endif

# Build tags per backend.
TAGS_metal := mlx
TAGS_cpu   := mlx
TAGS_cuda  := mlx cuda
TAGS_fake  :=
TAGS       := $(TAGS_$(BACKEND))

# Release asset name. Non-release backends get a suffix so they can never be
# mistaken for (or overwrite) a published asset.
ifeq ($(BACKEND),fake)
  ASSET := vakt-$(GOOS)-$(GOARCH)-fake
else ifeq ($(GOOS)-$(BACKEND),darwin-cpu)
  ASSET := vakt-darwin-$(GOARCH)-cpu
else ifeq ($(GOOS),darwin)
  ASSET := vakt-darwin-$(GOARCH)
else ifeq ($(BACKEND),cuda)
  ASSET := vakt-$(GOOS)-$(GOARCH)-cuda13
else
  ASSET := vakt-$(GOOS)-$(GOARCH)-$(BACKEND)
endif

BRAND   := github.com/vaktex/vakt/internal/brand
XFLAGS  := -X $(BRAND).Version=$(VERSION) -X $(BRAND).Commit=$(COMMIT) -X $(BRAND).Backend=$(BACKEND)

# The tokenizers static lib is linked through CGO_LDFLAGS so that the Go
# package does not need a machine-specific path baked into a #cgo directive.
TOK_LIB := $(CURDIR)/third_party/tokenizers/lib/$(GOOS)_$(GOARCH)
export CGO_ENABLED := 1
export CGO_LDFLAGS := $(CGO_LDFLAGS) -L$(TOK_LIB)

# garble: obfuscate our packages and all dependencies. Packages that fail
# under garble can be excluded by narrowing GOGARBLE (see docs/BUILD.md).
export GOGARBLE ?= *

BIN_DIR  := bin
DIST_DIR := dist
PKG      := ./cmd/vakt

# Native-engine packages that exist in this checkout (pipeline lands in Phase 2).
NATIVE_PKGS = $(shell for d in ./internal/engine ./internal/pipeline; do [ -d "$$d" ] && echo "$$d/..."; done)

# Keep build-machine paths out of native objects (C/C++ __FILE__, Rust panics).
export CGO_CFLAGS   := $(CGO_CFLAGS) -ffile-prefix-map=$(CURDIR)=.
export CGO_CXXFLAGS := $(CGO_CXXFLAGS) -ffile-prefix-map=$(CURDIR)=.
export RUSTFLAGS    := $(RUSTFLAGS) --remap-path-prefix=$(CURDIR)=. --remap-path-prefix=$(HOME)=~

.PHONY: help deps deps-mlx deps-tokenizers dev prod sign test parity bench lint audit checksums clean print-%

help: ## Show targets
	@printf 'Vaktex OSS build (BACKEND=%s, %s/%s)\n\n' '$(BACKEND)' '$(GOOS)' '$(GOARCH)'
	@grep -E '^[a-z%-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | awk -F'\t' '{printf "  %-12s %s\n", $$1, $$2}'

print-%: ## Print a make variable, e.g. make print-ASSET
	@echo '$($*)'

deps: deps-tokenizers deps-mlx ## Build native deps for BACKEND

deps-mlx:
ifneq ($(BACKEND),fake)
	@[ -x third_party/mlx/build.sh ] || { echo 'deps: third_party/mlx/build.sh is missing (it lands with the engine)'; exit 1; }
	./third_party/mlx/build.sh $(BACKEND)
endif

deps-tokenizers:
	@if [ -x third_party/tokenizers/build.sh ]; then ./third_party/tokenizers/build.sh; \
	else echo 'tokenizers: build script not present yet, skipping'; fi

dev: ## Fast unobfuscated build into bin/vakt
	@mkdir -p $(BIN_DIR)
	$(GO) build -tags '$(TAGS)' -ldflags '$(XFLAGS)' -o $(BIN_DIR)/vakt $(PKG)
	@echo "built $(BIN_DIR)/vakt ($(BACKEND))"

prod: ## Obfuscated, stripped release build into dist/$(ASSET)
	@mkdir -p $(DIST_DIR)
	@if [ '$(BACKEND)' = fake ]; then echo 'note: BACKEND=fake produces $(ASSET), which is not a release asset'; fi
	@test -x '$(GARBLE)' || { echo 'garble not found: go install mvdan.cc/garble@v0.18.0'; exit 1; }
	$(GARBLE) -literals -tiny -seed=random build -trimpath -buildvcs=false \
		-tags '$(TAGS)' -ldflags '-s -w $(XFLAGS)' -o $(DIST_DIR)/$(ASSET) $(PKG)
	@$(MAKE) --no-print-directory sign ASSET=$(ASSET)
	@ls -lh $(DIST_DIR)/$(ASSET) | awk '{print "built " $$9 " (" $$5 ")"}'

sign: ## Strip local symbols and codesign (darwin)
ifeq ($(GOOS),darwin)
	strip -x $(DIST_DIR)/$(ASSET)
	codesign --force --sign "$${CODESIGN_IDENTITY:--}" \
		$(if $(CODESIGN_IDENTITY),--timestamp --options runtime,--timestamp=none) \
		$(DIST_DIR)/$(ASSET)
	@if [ "$${NOTARIZE:-0}" = 1 ]; then ./scripts/notarize.sh $(DIST_DIR)/$(ASSET); fi
else
	strip --strip-unneeded $(DIST_DIR)/$(ASSET) 2>/dev/null || true
endif

test: ## Unit tests (race); native engine tests when BACKEND != fake
	$(GO) test -race ./...
ifneq ($(BACKEND),fake)
	$(GO) test -tags '$(TAGS)' $(NATIVE_PKGS)
endif

parity: ## Numerical parity against the PyTorch reference fixtures
	$(GO) test -tags '$(TAGS)' -run Parity -count=1 -v ./internal/engine/...

bench: ## Engine and pipeline benchmarks
	$(GO) test -tags '$(TAGS)' -run '^$$' -bench . -benchtime 3x $(NATIVE_PKGS)

lint: ## vet, staticcheck, gosec, govulncheck, shellcheck
	$(GO) vet ./...
	staticcheck ./...
	gosec -quiet -exclude-dir=third_party -exclude-dir=.worktrees ./...
	govulncheck ./...
	@if [ -f install.sh ]; then shellcheck -s sh install.sh; fi
	shellcheck scripts/*.sh scripts/docker/*.sh

audit: ## Release hygiene checks on dist/$(ASSET)
	EXPECT_VERSION='$(VERSION)' EXPECT_BACKEND='$(BACKEND)' ./scripts/audit.sh $(DIST_DIR)/$(ASSET)

checksums: ## Write dist/SHA256SUMS (release assets + install.sh)
	@if [ -f install.sh ]; then cp install.sh $(DIST_DIR)/install.sh; fi
	cd $(DIST_DIR) && files=$$(ls vakt-* install.sh 2>/dev/null | grep -v -- '-fake$$') && \
		{ if command -v sha256sum >/dev/null; then sha256sum $$files; else shasum -a 256 $$files; fi; } > SHA256SUMS
	@cat $(DIST_DIR)/SHA256SUMS

clean: ## Remove build outputs
	rm -rf $(BIN_DIR) $(DIST_DIR)
