package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vaktex/vakt/internal/account"
)

// fakeCloud is a Vaktex account API. tokenReplies are served in order by
// /device/token ("ok" means approve); account is the /account body, or ""
// for a 401.
type fakeCloud struct {
	t            *testing.T
	mu           sync.Mutex
	tokenReplies []string
	polls        int
	account      string
	revoked      []string
	sleeps       []time.Duration
	opened       []string
}

func (f *fakeCloud) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	switch r.URL.Path {
	case "/api/vakt/device/code":
		_, _ = w.Write([]byte(`{"device_code":"vkd_1","user_code":"WDJB-MJHT","verification_uri":"https://vaktex.com/activate","verification_uri_complete":"https://vaktex.com/activate?code=WDJB-MJHT","expires_in":600,"interval":5}`))
	case "/api/vakt/device/token":
		reply := f.tokenReplies[min(f.polls, len(f.tokenReplies)-1)]
		f.polls++
		if reply == "ok" {
			_, _ = w.Write([]byte(`{"token":"vkx_new","email":"a@b.c","plan":"free"}`))
			return
		}
		w.WriteHeader(400)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": reply})
	case "/api/vakt/account":
		if f.account == "" || r.Header.Get("Authorization") != "Bearer vkx_new" {
			w.WriteHeader(401)
			return
		}
		_, _ = w.Write([]byte(f.account))
	case "/api/vakt/auth/revoke":
		f.revoked = append(f.revoked, strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
		_, _ = w.Write([]byte(`{"ok":true}`))
	default:
		f.t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		w.WriteHeader(404)
	}
}

func startCloud(t *testing.T, tokenReplies ...string) (*fakeCloud, string) {
	t.Helper()
	isolate(t)
	f := &fakeCloud{t: t, tokenReplies: tokenReplies}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	t.Setenv("VAKT_CLOUD_URL", srv.URL+"/")

	oldNew, oldOpen, oldTTY := newAccountClient, openBrowser, stdinIsTTY
	t.Cleanup(func() { newAccountClient, openBrowser, stdinIsTTY = oldNew, oldOpen, oldTTY })
	newAccountClient = func(base string) (*account.Client, error) {
		c, err := oldNew(base)
		if err != nil {
			return nil, err
		}
		c.Sleep = func(ctx context.Context, d time.Duration) error {
			f.mu.Lock()
			f.sleeps = append(f.sleeps, d)
			f.mu.Unlock()
			return ctx.Err()
		}
		return c, nil
	}
	openBrowser = func(u string) error { f.opened = append(f.opened, u); return nil }
	stdinIsTTY = func() bool { return true }
	return f, srv.URL
}

func signIn(t *testing.T, base string) {
	t.Helper()
	if err := account.Save(&account.Credentials{Token: "vkx_new", Email: "a@b.c", Plan: "free", BaseURL: base}); err != nil {
		t.Fatal(err)
	}
}

func TestLoginHappyPath(t *testing.T) {
	f, base := startCloud(t, "authorization_pending", "slow_down", "ok")
	code, stdout, stderr := runCLI(t, "", "login")
	if code != exitOK {
		t.Fatalf("code %d stderr %q", code, stderr)
	}
	for _, want := range []string{"Open https://vaktex.com/activate and enter WDJB-MJHT", "Signed in as a@b.c (Free)."} {
		if !strings.Contains(stdout, want) {
			t.Errorf("stdout missing %q:\n%s", want, stdout)
		}
	}
	if strings.Count(stderr, "Waiting") != 1 {
		t.Errorf("want one waiting line: %q", stderr)
	}
	if want := []time.Duration{5 * time.Second, 5 * time.Second, 10 * time.Second}; len(f.sleeps) != 3 ||
		f.sleeps[0] != want[0] || f.sleeps[1] != want[1] || f.sleeps[2] != want[2] {
		t.Errorf("sleeps %v, want %v", f.sleeps, want)
	}
	if len(f.opened) != 1 || f.opened[0] != "https://vaktex.com/activate?code=WDJB-MJHT" {
		t.Errorf("opened %v", f.opened)
	}
	c, err := account.Load()
	if err != nil || c.Token != "vkx_new" || c.Email != "a@b.c" || c.Plan != "free" || c.BaseURL != base || c.CreatedAt.IsZero() {
		t.Fatalf("saved %+v err %v", c, err)
	}
}

func TestLoginNoBrowser(t *testing.T) {
	f, _ := startCloud(t, "ok")
	if code, _, stderr := runCLI(t, "", "login", "--no-browser"); code != exitOK {
		t.Fatalf("code %d %s", code, stderr)
	}
	if len(f.opened) != 0 {
		t.Errorf("opened %v", f.opened)
	}
}

func TestLoginStops(t *testing.T) {
	for reply, want := range map[string]string{
		"access_denied": "Sign-in was denied in the browser.",
		"expired_token": "The code expired. Run vakt login again.",
	} {
		startCloud(t, "authorization_pending", reply)
		code, _, stderr := runCLI(t, "", "login")
		if code == exitOK || !strings.Contains(stderr, want) {
			t.Errorf("%s: code %d stderr %q", reply, code, stderr)
		}
		if _, err := account.Load(); err == nil {
			t.Errorf("%s: credentials saved", reply)
		}
	}
}

func TestLoginAlreadySignedIn(t *testing.T) {
	f, base := startCloud(t, "ok")
	signIn(t, base)
	code, stdout, _ := runCLI(t, "", "login")
	if code != exitOK || !strings.Contains(stdout, "Already signed in as a@b.c. Run vakt logout first.") || f.polls != 0 {
		t.Fatalf("code %d polls %d stdout %q", code, f.polls, stdout)
	}
}

func TestLogoutRevokesAndDeletes(t *testing.T) {
	f, base := startCloud(t, "ok")
	signIn(t, base)
	code, stdout, stderr := runCLI(t, "", "logout")
	if code != exitOK || stdout != "Signed out.\n" || stderr != "" {
		t.Fatalf("code %d stdout %q stderr %q", code, stdout, stderr)
	}
	if len(f.revoked) != 1 || f.revoked[0] != "vkx_new" {
		t.Errorf("revoked %v", f.revoked)
	}
	if _, err := account.Load(); err == nil {
		t.Error("credentials still present")
	}
	if code, stdout, _ := runCLI(t, "", "logout"); code != exitOK || stdout != "Not signed in.\n" {
		t.Errorf("second logout: %d %q", code, stdout)
	}
}

func TestLogoutOffline(t *testing.T) {
	_, _ = startCloud(t, "ok")
	signIn(t, "http://127.0.0.1:1") // nothing listens on port 1
	code, stdout, stderr := runCLI(t, "", "logout")
	if code != exitOK || stdout != "Signed out.\n" || !strings.Contains(stderr, "could not revoke") {
		t.Fatalf("code %d stdout %q stderr %q", code, stdout, stderr)
	}
	if _, err := account.Load(); err == nil {
		t.Error("credentials still present")
	}
}

func TestAccountOutput(t *testing.T) {
	f, base := startCloud(t, "ok")
	if code, _, stderr := runCLI(t, "", "account"); code != 1 || stderr != "Not signed in. Run vakt login.\n" {
		t.Errorf("not signed in: %d %q", code, stderr)
	}
	signIn(t, base)

	f.account = `{"plan":"free","emailVerified":true,"month":{"month":"2026-10","spentUsd":4.12,"resetsAt":"2026-11-01T00:00:00Z"},"today":{"requests":212,"limit":1500,"resetsAt":"2026-10-14T00:00:00Z"}}`
	code, stdout, _ := runCLI(t, "", "account")
	want := "Signed in as a@b.c · Free\nCloud review: $4.12 used this month (resets 1 Nov)\nToday: 212 of 1,500 requests\n"
	if code != exitOK || stdout != want {
		t.Errorf("verified:\n%s\nwant:\n%s", stdout, want)
	}

	f.account = `{"plan":"free","emailVerified":false,"month":{"month":"2026-10","spentUsd":0,"resetsAt":"2026-11-01T00:00:00Z"},"today":null}`
	_, stdout, _ = runCLI(t, "", "account")
	if !strings.Contains(stdout, "Verify your email to get the full free account.") || strings.Contains(stdout, "Today:") {
		t.Errorf("unverified:\n%s", stdout)
	}

	_, stdout, _ = runCLI(t, "", "account", "--json")
	var m map[string]any
	if err := json.Unmarshal([]byte(stdout), &m); err != nil || m["email"] != "a@b.c" || m["emailVerified"] != false || m["today"] != nil {
		t.Errorf("json %v: %s", err, stdout)
	}

	f.account = ""
	code, _, stderr := runCLI(t, "", "account")
	if code != 1 || !strings.Contains(stderr, "Run vakt login") {
		t.Errorf("401: %d %q", code, stderr)
	}
}

func TestDoctorAccountLine(t *testing.T) {
	_, base := startCloud(t, "ok")
	signIn(t, base)
	if got, ok := accountStatus(); !ok || got != "signed in as a@b.c (Free)" {
		t.Errorf("%q %v", got, ok)
	}
}

func TestDeviceName(t *testing.T) {
	old := hostname
	t.Cleanup(func() { hostname = old })
	hostname = func() (string, error) { return " " + strings.Repeat("h", 80) + "\n", nil }
	if got := deviceName(); got != strings.Repeat("h", 64) {
		t.Errorf("%q", got)
	}
}
