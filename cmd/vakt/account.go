package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/spf13/cobra"

	"github.com/vaktex/vakt/internal/account"
	"github.com/vaktex/vakt/internal/brand"
)

// `vakt login`, `vakt logout` and `vakt account` manage the Vaktex account.
// Signing in sends only this machine's host name; no code leaves the machine.
// (login.go is the unrelated Hugging Face sign-in for the gated model.)

// Test hooks.
var (
	newAccountClient = account.NewClient
	openBrowser      = func(url string) error {
		var cmd *exec.Cmd
		switch runtime.GOOS {
		case "darwin":
			cmd = exec.Command("open", url) // #nosec G204 -- fixed binary, URL from our server
		case "windows":
			cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url) // #nosec G204
		default:
			cmd = exec.Command("xdg-open", url) // #nosec G204
		}
		if err := cmd.Start(); err != nil {
			return err
		}
		go func() { _ = cmd.Wait() }()
		return nil
	}
	hostname = os.Hostname
)

const exitNotSignedIn = 1

func deviceName() string {
	h, err := hostname()
	if err != nil {
		return ""
	}
	h = strings.TrimSpace(h)
	for len(h) > 64 {
		_, size := utf8.DecodeLastRuneInString(h)
		h = h[:len(h)-size]
	}
	return h
}

func planName(p string) string {
	switch p {
	case "free":
		return "Free"
	case "paid":
		return "Paid"
	case "":
		return "unknown plan"
	}
	return strings.ToUpper(p[:1]) + p[1:]
}

func newLoginCmd() *cobra.Command {
	var force, noBrowser bool
	cmd := &cobra.Command{
		Use:   "login",
		Short: "Sign in to your Vaktex account",
		Long: "Sign in to your Vaktex account in the browser. Only this machine's name is sent;\n" +
			"no code leaves the machine.",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return runLogin(cmd.Context(), cmd.OutOrStdout(), cmd.ErrOrStderr(), force, noBrowser)
		},
	}
	cmd.Flags().BoolVar(&force, "force", false, "sign in again even if already signed in")
	cmd.Flags().BoolVar(&noBrowser, "no-browser", false, "print the link without opening a browser")
	return cmd
}

func runLogin(ctx context.Context, stdout, stderr io.Writer, force, noBrowser bool) error {
	old, err := account.Load()
	switch {
	case err == nil && !force:
		fmt.Fprintf(stdout, "Already signed in as %s. Run %s logout first.\n", old.Email, brand.Binary)
		return nil
	case err != nil && !errors.Is(err, account.ErrNotSignedIn) && !force:
		return err
	}
	base := account.BaseURL()
	c, err := newAccountClient(base)
	if err != nil {
		return err
	}
	dc, err := c.RequestCode(ctx, deviceName())
	if err != nil {
		return err
	}
	fmt.Fprintf(stdout, "Open %s and enter %s\n", dc.VerificationURI, dc.UserCode)
	tty := stdinIsTTY()
	if tty && !noBrowser && dc.VerificationURIComplete != "" {
		_ = openBrowser(dc.VerificationURIComplete)
	}
	fmt.Fprint(stderr, "Waiting for you to confirm in the browser...")
	tok, err := c.WaitForToken(ctx, dc)
	fmt.Fprintln(stderr)
	if err != nil {
		return err
	}
	if err := account.Save(&account.Credentials{
		Token: tok.Token, Email: tok.Email, Plan: tok.Plan, BaseURL: base, CreatedAt: time.Now().UTC(),
	}); err != nil {
		return fmt.Errorf("saving credentials: %w", err)
	}
	// Replaced a still-valid sign-in (--force): retire the old token.
	if old != nil && old.Token != tok.Token {
		_ = revokeQuietly(ctx, old)
	}
	fmt.Fprintf(stdout, "Signed in as %s (%s).\n", tok.Email, planName(tok.Plan))
	return nil
}

// revokeQuietly revokes c's token with a short timeout and reports whether
// the server confirmed it.
func revokeQuietly(ctx context.Context, c *account.Credentials) error {
	base := c.BaseURL
	if base == "" {
		base = account.BaseURL()
	}
	cl, err := newAccountClient(base)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	return cl.Revoke(ctx, c.Token)
}

func newLogoutCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "logout",
		Short: "Sign out and remove the stored token",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return runLogout(cmd.Context(), cmd.OutOrStdout(), cmd.ErrOrStderr())
		},
	}
}

func runLogout(ctx context.Context, stdout, stderr io.Writer) error {
	c, err := account.Load()
	if errors.Is(err, account.ErrNotSignedIn) {
		fmt.Fprintln(stdout, "Not signed in.")
		return nil
	}
	if err == nil {
		if rerr := revokeQuietly(ctx, c); rerr != nil && ctx.Err() == nil {
			fmt.Fprintf(stderr, "%s: could not revoke the token on the server (%v); removing it from this machine.\n", brand.Binary, rerr)
		}
	}
	// A corrupt file is removed too: that is the way out of it.
	if _, err := account.Delete(); err != nil {
		return fmt.Errorf("removing credentials: %w", err)
	}
	fmt.Fprintln(stdout, "Signed out.")
	return nil
}

func newAccountCmd() *cobra.Command {
	var asJSON bool
	cmd := &cobra.Command{
		Use:   "account",
		Short: "Show your Vaktex account and usage",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return runAccount(cmd.Context(), cmd.OutOrStdout(), cmd.ErrOrStderr(), asJSON)
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "print the account as JSON")
	return cmd
}

func runAccount(ctx context.Context, stdout, stderr io.Writer, asJSON bool) error {
	c, err := account.Load()
	if errors.Is(err, account.ErrNotSignedIn) {
		fmt.Fprintf(stderr, "Not signed in. Run %s login.\n", brand.Binary)
		return &exitCodeError{code: exitNotSignedIn}
	}
	if err != nil {
		return err
	}
	base := c.BaseURL
	if base == "" {
		base = account.BaseURL()
	}
	cl, err := newAccountClient(base)
	if err != nil {
		return err
	}
	a, raw, err := cl.Account(ctx, c.Token)
	if errors.Is(err, account.ErrUnauthorized) {
		return &exitCodeError{code: exitNotSignedIn, err: err}
	}
	if err != nil {
		return err
	}
	if asJSON {
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			return err
		}
		m["email"] = c.Email
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(m)
	}
	plan := a.Plan
	if plan == "" {
		plan = c.Plan
	}
	fmt.Fprintf(stdout, "Signed in as %s · %s\n", c.Email, planName(plan))
	if m := a.Month; m != nil {
		// What was spent, never the allowance: the cap stays on the server.
		line := fmt.Sprintf("Cloud review: %s used this month", usd(m.SpentUSD))
		if d := shortDate(m.ResetsAt); d != "" {
			line += " (resets " + d + ")"
		}
		fmt.Fprintln(stdout, line)
	}
	if t := a.Today; t != nil {
		fmt.Fprintf(stdout, "Today: %s of %s requests\n", thousands(t.Requests), thousands(t.Limit))
	} else {
		fmt.Fprintln(stdout, "Verify your email to get the full free account.")
	}
	return nil
}

// usd formats whole dollars without cents: $5, $4.12.
func usd(v float64) string {
	if v == float64(int64(v)) {
		return "$" + thousands(int(v))
	}
	return fmt.Sprintf("$%.2f", v)
}

func thousands(n int) string {
	s := strconv.Itoa(n)
	neg := strings.HasPrefix(s, "-")
	s = strings.TrimPrefix(s, "-")
	var b strings.Builder
	for i, r := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(r)
	}
	if neg {
		return "-" + b.String()
	}
	return b.String()
}

// shortDate turns an RFC 3339 time into "1 Nov" (UTC, the server's month).
func shortDate(s string) string {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return ""
	}
	return t.UTC().Format("2 Jan")
}

// accountStatus is the doctor line; it reads only the local file.
func accountStatus() (string, bool) {
	c, err := account.Load()
	switch {
	case errors.Is(err, account.ErrNotSignedIn):
		return "not signed in (optional; run `" + brand.Binary + " login`)", true
	case err != nil:
		return "unreadable credentials: " + oneLine(err.Error()), false
	}
	return "signed in as " + c.Email + " (" + planName(c.Plan) + ")", true
}
