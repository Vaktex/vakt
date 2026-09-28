# Contributing

Bug reports and pull requests are welcome. For anything bigger than a fix,
open an issue first so we can agree on the approach before you spend time on
it.

## Getting set up

You don't need a GPU or the model to work on most of `vakt`. The walker,
parser, tokenizer, pipeline, report and CLI all build and test without the
native engine:

```sh
make dev BACKEND=fake   # builds bin/vakt with no model backend
go test ./...
```

Engine work needs the native libraries (`make deps`, then `make test`); see
[docs/BUILD.md](docs/BUILD.md) for toolchains. Parity tests need mock
checkpoints that are too large for git and skip when they are absent.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes how the packages fit
together and the rules each one keeps. It's worth reading before changing
anything in `internal/pipeline` or `internal/engine`.

## Before you send a pull request

- `make lint` and `go test -race ./...` pass.
- New behaviour has a test. Bug fixes have a test that failed before the fix.
- Anything that reads from the scanned tree treats it as hostile (see
  [docs/SECURITY-DESIGN.md](docs/SECURITY-DESIGN.md)).
- Commit messages follow the existing style: `area: what changed`, e.g.
  `walk: skip .jj directories`.

By contributing you agree that Vaktex may use your contributions under the
[LICENSE](LICENSE) and under any other terms, including commercial licenses
it offers for `vakt`.
