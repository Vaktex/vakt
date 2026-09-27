# Reporting a vulnerability

Please don't open a public issue for a security bug. Email
security@vaktex.com, or use GitHub's private vulnerability reporting on this
repository (Security tab, "Report a vulnerability").

Include what you ran, what happened, and a proof of concept if you have one.
We'll acknowledge within three working days and keep you updated until it is
fixed.

`vakt` reads code you may not trust, so bugs in these areas are in scope and
we especially want to hear about them:

- a scanned repository escaping the scan root (symlinks, ignore files, paths
  in reports), or hiding files from the scan without it being reported;
- parser or tokenizer input that crashes, hangs or exhausts memory beyond the
  documented budgets;
- a model file that gets past the safetensors header checks;
- the MCP server reading or writing outside `--dir`;
- `install.sh` running anything it did not verify.

A model score being wrong (a missed vulnerability or a false positive) is not
a security bug; open a normal issue with the code if you can share it.

How the scanner handles untrusted input, and the static-analysis findings we
have accepted, are written up in [docs/SECURITY-DESIGN.md](docs/SECURITY-DESIGN.md).
