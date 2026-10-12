package account

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vaktex/vakt/internal/brand"
)

func TestCredentialsRoundTripAndPerms(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "cfg")
	t.Setenv("VAKT_CONFIG_DIR", dir)
	if _, err := Load(); !errors.Is(err, ErrNotSignedIn) {
		t.Fatalf("empty: %v", err)
	}
	want := &Credentials{Token: "vkx_secret", Email: "a@b.c", Plan: "free", BaseURL: "https://vaktex.com",
		CreatedAt: time.Date(2026, 10, 13, 9, 0, 0, 0, time.UTC)}
	if err := Save(want); err != nil {
		t.Fatal(err)
	}
	got, err := Load()
	if err != nil || *got != *want {
		t.Fatalf("got %+v err %v", got, err)
	}
	p, _ := Path()
	if runtime.GOOS != "windows" {
		if fi, _ := os.Stat(p); fi.Mode().Perm() != 0o600 {
			t.Errorf("file mode %v", fi.Mode().Perm())
		}
		if fi, _ := os.Stat(dir); fi.Mode().Perm() != 0o700 {
			t.Errorf("dir mode %v", fi.Mode().Perm())
		}
	}
	b, _ := os.ReadFile(p)
	for _, k := range []string{`"token"`, `"email"`, `"plan"`, `"base_url"`, `"created_at"`} {
		if !strings.Contains(string(b), k) {
			t.Errorf("file missing %s: %s", k, b)
		}
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Errorf("temp files left behind: %v", entries)
	}
	if ok, err := Delete(); !ok || err != nil {
		t.Fatalf("delete: %v %v", ok, err)
	}
	if ok, err := Delete(); ok || err != nil {
		t.Fatalf("second delete: %v %v", ok, err)
	}
	if _, err := Load(); !errors.Is(err, ErrNotSignedIn) {
		t.Fatalf("after delete: %v", err)
	}
}

func TestBaseURL(t *testing.T) {
	t.Setenv("VAKT_CLOUD_URL", "")
	if BaseURL() != DefaultBaseURL {
		t.Error(BaseURL())
	}
	t.Setenv("VAKT_CLOUD_URL", "https://staging.vaktex.com//")
	if BaseURL() != "https://staging.vaktex.com" {
		t.Error(BaseURL())
	}
	if _, err := NewClient("http://vaktex.com"); err == nil {
		t.Error("accepted plain http")
	}
	if _, err := NewClient("http://127.0.0.1:8080/"); err != nil {
		t.Error(err)
	}
}

// fakeClock advances on Sleep so polling tests run instantly.
type fakeClock struct {
	mu     sync.Mutex
	now    time.Time
	sleeps []time.Duration
}

func (f *fakeClock) Now() time.Time { f.mu.Lock(); defer f.mu.Unlock(); return f.now }
func (f *fakeClock) Sleep(ctx context.Context, d time.Duration) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.sleeps = append(f.sleeps, d)
	f.now = f.now.Add(d)
	return ctx.Err()
}

func testClient(t *testing.T, h http.HandlerFunc) (*Client, *fakeClock) {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	c, err := NewClient(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	fc := &fakeClock{now: time.Unix(1_800_000_000, 0)}
	c.Now, c.Sleep = fc.Now, fc.Sleep
	return c, fc
}

func tokenServer(t *testing.T, replies ...string) http.HandlerFunc {
	var mu sync.Mutex
	n := 0
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-vakt-client") != "cli/"+brand.Version {
			t.Errorf("client header %q", r.Header.Get("x-vakt-client"))
		}
		mu.Lock()
		defer mu.Unlock()
		reply := replies[min(n, len(replies)-1)]
		n++
		if reply == "ok" {
			_, _ = w.Write([]byte(`{"token":"vkx_1","email":"a@b.c","plan":"free"}`))
			return
		}
		if reply == "500" {
			w.WriteHeader(500)
			_, _ = w.Write([]byte(`{"error":"internal"}`))
			return
		}
		w.WriteHeader(400)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": reply})
	}
}

func TestWaitForTokenIntervalAndSlowDown(t *testing.T) {
	c, fc := testClient(t, tokenServer(t, "authorization_pending", "slow_down", "ok"))
	tok, err := c.WaitForToken(context.Background(), &DeviceCode{DeviceCode: "vkd_1", Interval: 5, ExpiresIn: 600})
	if err != nil || tok.Email != "a@b.c" {
		t.Fatalf("tok %+v err %v", tok, err)
	}
	want := []time.Duration{5 * time.Second, 5 * time.Second, 10 * time.Second}
	if len(fc.sleeps) != len(want) {
		t.Fatalf("sleeps %v", fc.sleeps)
	}
	for i := range want {
		if fc.sleeps[i] != want[i] {
			t.Errorf("sleeps %v, want %v", fc.sleeps, want)
		}
	}
}

func TestWaitForTokenStops(t *testing.T) {
	for reply, want := range map[string]error{"access_denied": ErrDenied, "expired_token": ErrExpired} {
		c, _ := testClient(t, tokenServer(t, "authorization_pending", reply))
		_, err := c.WaitForToken(context.Background(), &DeviceCode{DeviceCode: "x", Interval: 1, ExpiresIn: 60})
		if !errors.Is(err, want) {
			t.Errorf("%s: %v", reply, err)
		}
	}
}

func TestWaitForTokenExpiresLocally(t *testing.T) {
	c, _ := testClient(t, tokenServer(t, "authorization_pending"))
	_, err := c.WaitForToken(context.Background(), &DeviceCode{DeviceCode: "x", Interval: 5, ExpiresIn: 30})
	if !errors.Is(err, ErrExpired) {
		t.Fatal(err)
	}
}

func TestWaitForTokenRetriesThenFails(t *testing.T) {
	c, fc := testClient(t, tokenServer(t, "500", "ok"))
	if _, err := c.WaitForToken(context.Background(), &DeviceCode{DeviceCode: "x", Interval: 1, ExpiresIn: 600}); err != nil {
		t.Fatalf("one transient failure: %v", err)
	}
	if fc.sleeps[1] != 2*time.Second {
		t.Errorf("no backoff: %v", fc.sleeps)
	}
	c, _ = testClient(t, tokenServer(t, "500"))
	_, err := c.WaitForToken(context.Background(), &DeviceCode{DeviceCode: "x", Interval: 1, ExpiresIn: 600})
	if err == nil || !strings.Contains(err.Error(), "internal") {
		t.Fatalf("err %v", err)
	}
}

func TestRequestCodeSendsOnlyDeviceName(t *testing.T) {
	c, _ := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if r.URL.Path != "/api/vakt/device/code" || len(body) != 1 || body["device_name"] != "box" ||
			r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("%s %v %v", r.URL.Path, body, r.Header)
		}
		_, _ = w.Write([]byte(`{"device_code":"vkd_1","user_code":"WDJB-MJHT","verification_uri":"https://vaktex.com/activate","expires_in":600,"interval":5}`))
	})
	dc, err := c.RequestCode(context.Background(), "box")
	if err != nil || dc.UserCode != "WDJB-MJHT" {
		t.Fatalf("%+v %v", dc, err)
	}
}

func TestAccountUnauthorized(t *testing.T) {
	c, _ := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer vkx_old" {
			t.Errorf("auth %q", r.Header.Get("Authorization"))
		}
		w.WriteHeader(401)
	})
	if _, _, err := c.Account(context.Background(), "vkx_old"); !errors.Is(err, ErrUnauthorized) {
		t.Fatal(err)
	}
}
