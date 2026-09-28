package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/hub"
)

// DOM-0.8B is a gated model: a download that fails for lack of
// credentials (no token, a rejected one, or one without access) is fixed by
// signing in with the Hugging Face CLI and, if needed, requesting access on
// the model page. withSignIn turns that failure into a prompt.

// Test hooks.
var (
	stdinIsTTY = func() bool { return isTTY(os.Stdin) }
	runHFLogin = func(ctx context.Context, stdin io.Reader, stdout, stderr io.Writer) error {
		name, args, ok := hfLoginCommand()
		if !ok {
			return errNoHFCLI
		}
		cmd := exec.CommandContext(ctx, name, args...) // #nosec G204 -- fixed binary name and arguments
		cmd.Stdin, cmd.Stdout, cmd.Stderr = stdin, stdout, stderr
		return cmd.Run()
	}
)

var errNoHFCLI = errors.New("the Hugging Face CLI is not installed")

// hfLoginCommand finds `hf auth login`, falling back to the older
// `huggingface-cli login`.
func hfLoginCommand() (string, []string, bool) {
	if _, err := lookPath("hf"); err == nil {
		return "hf", []string{"auth", "login"}, true
	}
	if _, err := lookPath("huggingface-cli"); err == nil {
		return "huggingface-cli", []string{"login"}, true
	}
	return "", nil, false
}

// signInHelp explains how to get access, for non-interactive runs and after
// a declined prompt.
func signInHelp(repo string) string {
	return fmt.Sprintf(`%[1]s is a gated model on Hugging Face. To download it:
  1. request access at https://huggingface.co/%[1]s
  2. sign in:  hf auth login     (install: pip install -U "huggingface_hub[cli]")
     or set HF_TOKEN to a token with read access
  3. run:      %[2]s summon`, repo, brand.Binary)
}

// withSignIn runs download; if it fails with an authentication error it
// offers `hf auth login` on a terminal and retries once after a successful
// sign-in. Non-interactive runs get instructions instead of a prompt.
func withSignIn[T any](ctx context.Context, repo string, stdin io.Reader, stderr io.Writer, download func() (T, error)) (T, error) {
	v, err := download()
	if err == nil || !errors.Is(err, hub.ErrAuth) {
		return v, err
	}
	var zero T
	if !stdinIsTTY() {
		return zero, fmt.Errorf("%w\n\n%s", err, signInHelp(repo))
	}
	fmt.Fprintf(stderr, "%s: %v\n\n", brand.Binary, err)
	fmt.Fprintf(stderr, "%s is a gated model: you need a Hugging Face account with access to it.\n", repo)
	fmt.Fprintf(stderr, "Request access at https://huggingface.co/%s if you have not already.\n\n", repo)
	name, args, ok := hfLoginCommand()
	if !ok {
		return zero, fmt.Errorf("%w; %s", errNoHFCLI, signInHelp(repo))
	}
	fmt.Fprintf(stderr, "Sign in now with `%s %s`? [Y/n] ", name, strings.Join(args, " "))
	answer, _ := bufio.NewReader(stdin).ReadString('\n')
	if a := strings.ToLower(strings.TrimSpace(answer)); a != "" && a != "y" && a != "yes" {
		return zero, fmt.Errorf("%w\n\n%s", err, signInHelp(repo))
	}
	if lerr := runHFLogin(ctx, stdin, stderr, stderr); lerr != nil {
		return zero, fmt.Errorf("sign-in failed: %w\n\n%s", lerr, signInHelp(repo))
	}
	fmt.Fprintln(stderr)
	v, err = download()
	if err != nil && errors.Is(err, hub.ErrAuth) {
		// Signed in but still refused: the account has no access yet.
		return zero, fmt.Errorf("%w\n\nYou are signed in, but this account cannot download %s yet.\nRequest access at https://huggingface.co/%s, then run `%s summon`",
			err, repo, repo, brand.Binary)
	}
	return v, err
}
