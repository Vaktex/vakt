package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/vaktex/vakt/internal/hub"
)

func stubLogin(t *testing.T, tty bool, haveCLI bool, login func() error) *int {
	t.Helper()
	oldTTY, oldRun, oldLook := stdinIsTTY, runHFLogin, lookPath
	t.Cleanup(func() { stdinIsTTY, runHFLogin, lookPath = oldTTY, oldRun, oldLook })
	stdinIsTTY = func() bool { return tty }
	calls := 0
	runHFLogin = func(context.Context, io.Reader, io.Writer, io.Writer) error { calls++; return login() }
	lookPath = func(name string) (string, error) {
		if haveCLI && name == "hf" {
			return "/usr/local/bin/hf", nil
		}
		return "", errors.New("not found")
	}
	return &calls
}

var errAuthStub = fmt.Errorf("%w: %w", hub.ErrAuth, hub.ErrPrivate)

func TestSignInPromptThenRetry(t *testing.T) {
	signedIn := false
	calls := stubLogin(t, true, true, func() error { signedIn = true; return nil })
	var stderr bytes.Buffer
	got, err := withSignIn(context.Background(), "vaktex/dom-oss-0.8b", strings.NewReader("\n"), &stderr, func() (string, error) {
		if !signedIn {
			return "", errAuthStub
		}
		return "ok", nil
	})
	if err != nil || got != "ok" || *calls != 1 {
		t.Fatalf("got %q err %v calls %d", got, err, *calls)
	}
	out := stderr.String()
	for _, want := range []string{"gated model", "https://huggingface.co/vaktex/dom-oss-0.8b", "hf auth login", "[Y/n]"} {
		if !strings.Contains(out, want) {
			t.Errorf("prompt missing %q:\n%s", want, out)
		}
	}
}

func TestSignInDeclined(t *testing.T) {
	calls := stubLogin(t, true, true, func() error { return nil })
	_, err := withSignIn(context.Background(), "r/m", strings.NewReader("n\n"), io.Discard, func() (string, error) { return "", errAuthStub })
	if err == nil || *calls != 0 || !strings.Contains(err.Error(), "hf auth login") {
		t.Fatalf("err %v calls %d", err, *calls)
	}
}

func TestSignInNonInteractiveGivesInstructions(t *testing.T) {
	calls := stubLogin(t, false, true, func() error { return nil })
	_, err := withSignIn(context.Background(), "r/m", strings.NewReader(""), io.Discard, func() (string, error) { return "", errAuthStub })
	if err == nil || *calls != 0 {
		t.Fatalf("err %v calls %d", err, *calls)
	}
	for _, want := range []string{"request access at https://huggingface.co/r/m", "hf auth login", "HF_TOKEN", "summon"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("missing %q in %v", want, err)
		}
	}
}

func TestSignInStillNoAccess(t *testing.T) {
	stubLogin(t, true, true, func() error { return nil })
	_, err := withSignIn(context.Background(), "r/m", strings.NewReader("y\n"), io.Discard, func() (string, error) { return "", errAuthStub })
	if err == nil || !strings.Contains(err.Error(), "signed in, but this account cannot download") {
		t.Fatalf("err %v", err)
	}
}

func TestSignInNoCLI(t *testing.T) {
	stubLogin(t, true, false, func() error { return nil })
	_, err := withSignIn(context.Background(), "r/m", strings.NewReader("y\n"), io.Discard, func() (string, error) { return "", errAuthStub })
	if !errors.Is(err, errNoHFCLI) || !strings.Contains(err.Error(), "pip install") {
		t.Fatalf("err %v", err)
	}
}

func TestSignInIgnoresOtherErrors(t *testing.T) {
	calls := stubLogin(t, true, true, func() error { return nil })
	boom := errors.New("disk full")
	_, err := withSignIn(context.Background(), "r/m", strings.NewReader("y\n"), io.Discard, func() (string, error) { return "", boom })
	if !errors.Is(err, boom) || *calls != 0 {
		t.Fatalf("err %v calls %d", err, *calls)
	}
}
