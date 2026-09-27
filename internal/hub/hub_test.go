package hub

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vaktex/vakt/internal/brand"
)

const (
	testToken  = "hf_TESTSECRETtoken1234567890"
	testRepo   = "vaktex/DOM-0.8B"
	testCommit = "0123456789abcdef0123456789abcdef01234567"
)

type fakeHub struct {
	t        *testing.T
	blob     []byte
	sha      string // advertised
	hub, cdn *httptest.Server

	mu          sync.Mutex
	hubAuth     []string
	cdnAuth     []string
	cdnRanges   []string
	cdnBytes    int
	status      int    // if set, the hub answers every request with it
	getRedirect string // if set, GET redirects here instead of the CDN
	private     bool   // 401 without the right token
}

func newFakeHub(t *testing.T, size int) *fakeHub {
	t.Helper()
	blob := make([]byte, size)
	_, _ = rand.Read(blob)
	sum := sha256.Sum256(blob)
	f := &fakeHub{t: t, blob: blob, sha: hex.EncodeToString(sum[:])}
	f.cdn = httptest.NewServer(http.HandlerFunc(f.serveCDN))
	f.hub = httptest.NewServer(http.HandlerFunc(f.serveHub))
	t.Cleanup(func() { f.hub.Close(); f.cdn.Close() })
	return f
}

func (f *fakeHub) serveHub(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.hubAuth = append(f.hubAuth, r.Header.Get("Authorization"))
	status, redirect, private := f.status, f.getRedirect, f.private
	f.mu.Unlock()
	if status != 0 {
		w.WriteHeader(status)
		return
	}
	if private && r.Header.Get("Authorization") != "Bearer "+testToken {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	want := "/" + testRepo + "/resolve/"
	if !strings.HasPrefix(r.URL.Path, want) || !strings.HasSuffix(r.URL.Path, "/"+brand.ModelFile) {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	loc := f.cdn.URL + "/blob?sig=signedCDNquery"
	if r.Method == http.MethodGet && redirect != "" {
		loc = redirect
	}
	w.Header().Set("X-Repo-Commit", testCommit)
	w.Header().Set("X-Linked-Etag", `"`+f.sha+`"`)
	w.Header().Set("X-Linked-Size", strconv.Itoa(len(f.blob)))
	w.Header().Set("Location", loc)
	w.WriteHeader(http.StatusFound)
}

func (f *fakeHub) serveCDN(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.cdnAuth = append(f.cdnAuth, r.Header.Get("Authorization"))
	f.cdnRanges = append(f.cdnRanges, r.Header.Get("Range"))
	f.mu.Unlock()
	start := 0
	if rg := r.Header.Get("Range"); rg != "" {
		n, err := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(rg, "bytes="), "-"))
		if err != nil || n >= len(f.blob) {
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return
		}
		start = n
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", n, len(f.blob)-1, len(f.blob)))
		w.WriteHeader(http.StatusPartialContent)
	}
	n, _ := w.Write(f.blob[start:])
	f.mu.Lock()
	f.cdnBytes += n
	f.mu.Unlock()
}

func (f *fakeHub) opts(t *testing.T) Options {
	return Options{Repo: testRepo, Revision: "main", CacheDir: t.TempDir(), Token: testToken, Endpoint: f.hub.URL}
}

func clearEnv(t *testing.T) {
	for _, k := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_ENDPOINT", "HF_HUB_OFFLINE", "VAKT_CACHE"} {
		t.Setenv(k, "")
	}
	t.Setenv("HF_HOME", t.TempDir()) // no token file
}

func TestFreshDownload(t *testing.T) {
	clearEnv(t)
	f := newFakeHub(t, 3<<20+17)
	o := f.opts(t)
	var last, total int64
	o.Progress = func(d, tot int64) { last, total = d, tot }
	path, sha, err := Resolve(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	if sha != f.sha {
		t.Errorf("sha = %s, want %s", sha, f.sha)
	}
	got, err := os.ReadFile(path) // #nosec G304 -- test
	if err != nil || !bytes.Equal(got, f.blob) {
		t.Fatalf("content mismatch (%v)", err)
	}
	repoDir := filepath.Join(o.CacheDir, "models--vaktex--DOM-0.8B")
	if want := filepath.Join(repoDir, "snapshots", testCommit, brand.ModelFile); path != want {
		t.Errorf("path = %s, want %s", path, want)
	}
	if fi, err := os.Lstat(path); err != nil || fi.Mode()&os.ModeSymlink == 0 {
		t.Errorf("snapshot is not a symlink: %v", err)
	}
	ref, _ := os.ReadFile(filepath.Join(repoDir, "refs", "main")) // #nosec G304 -- test
	if string(ref) != testCommit {
		t.Errorf("refs/main = %q", ref)
	}
	blobInfo, _ := os.Stat(filepath.Join(repoDir, "blobs", f.sha))
	if blobInfo == nil || blobInfo.Mode().Perm() != 0o600 {
		t.Errorf("blob perms = %v", blobInfo)
	}
	for _, d := range []string{filepath.Join(repoDir, "blobs"), filepath.Join(repoDir, "snapshots", testCommit), filepath.Join(repoDir, "refs")} {
		if st, err := os.Stat(d); err != nil || st.Mode().Perm() != 0o700 {
			t.Errorf("%s perms = %v (%v)", d, st.Mode().Perm(), err)
		}
	}
	if last != int64(len(f.blob)) || total != int64(len(f.blob)) {
		t.Errorf("progress = %d/%d", last, total)
	}
	// The hub gets the token, the CDN (another host) must not.
	for _, a := range f.hubAuth {
		if a != "Bearer "+testToken {
			t.Errorf("hub auth = %q", a)
		}
	}
	for _, a := range f.cdnAuth {
		if a != "" {
			t.Errorf("CDN received Authorization %q", a)
		}
	}
	// Second call: cached, no new download.
	before := f.cdnBytes
	p2, s2, err := Resolve(context.Background(), o)
	if err != nil || p2 != path || s2 != sha || f.cdnBytes != before {
		t.Errorf("second resolve: %v %v %v, cdn bytes %d -> %d", p2, s2, err, before, f.cdnBytes)
	}
}

func TestResumePartial(t *testing.T) {
	clearEnv(t)
	f := newFakeHub(t, 2<<20)
	o := f.opts(t)
	blobs := filepath.Join(o.CacheDir, "models--vaktex--DOM-0.8B", "blobs")
	if err := os.MkdirAll(blobs, 0o700); err != nil {
		t.Fatal(err)
	}
	half := len(f.blob) / 2
	if err := os.WriteFile(filepath.Join(blobs, f.sha+".incomplete"), f.blob[:half], 0o600); err != nil {
		t.Fatal(err)
	}
	_, sha, err := Resolve(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	if sha != f.sha {
		t.Fatal("sha mismatch")
	}
	if len(f.cdnRanges) != 1 || f.cdnRanges[0] != fmt.Sprintf("bytes=%d-", half) {
		t.Errorf("ranges = %v", f.cdnRanges)
	}
	if f.cdnBytes != len(f.blob)-half {
		t.Errorf("cdn sent %d bytes, want %d", f.cdnBytes, len(f.blob)-half)
	}
	if _, err := os.Stat(filepath.Join(blobs, f.sha+".incomplete")); !os.IsNotExist(err) {
		t.Error(".incomplete left behind")
	}
}

func TestShaMismatchRejected(t *testing.T) {
	clearEnv(t)
	f := newFakeHub(t, 1<<16)
	f.sha = strings.Repeat("0", 64)
	o := f.opts(t)
	_, _, err := Resolve(context.Background(), o)
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("err = %v", err)
	}
	blobs := filepath.Join(o.CacheDir, "models--vaktex--DOM-0.8B", "blobs")
	ents, _ := os.ReadDir(blobs)
	if len(ents) != 0 {
		t.Errorf("blobs left after mismatch: %v", ents)
	}
	if _, err := os.Stat(filepath.Join(o.CacheDir, "models--vaktex--DOM-0.8B", "snapshots")); !os.IsNotExist(err) {
		t.Error("snapshot created despite mismatch")
	}
}

func TestBadEtagRejected(t *testing.T) {
	clearEnv(t)
	f := newFakeHub(t, 1024)
	f.sha = "../../../../etc/passwd"
	_, _, err := Resolve(context.Background(), f.opts(t))
	if err == nil || !strings.Contains(err.Error(), "sha256 etag") {
		t.Fatalf("err = %v", err)
	}
}

func TestOfflineAndNetworkFallback(t *testing.T) {
	clearEnv(t)
	f := newFakeHub(t, 4096)
	o := f.opts(t)
	path, sha, err := Resolve(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	// Offline: no requests.
	n := len(f.hubAuth)
	oo := o
	oo.Offline = true
	p2, s2, err := Resolve(context.Background(), oo)
	if err != nil || p2 != path || s2 != sha || len(f.hubAuth) != n {
		t.Fatalf("offline: %v %v %v requests=%d", p2, s2, err, len(f.hubAuth)-n)
	}
	// HF_HUB_OFFLINE works too.
	t.Setenv("HF_HUB_OFFLINE", "1")
	if _, _, err := Resolve(context.Background(), o); err != nil || len(f.hubAuth) != n {
		t.Fatalf("HF_HUB_OFFLINE: %v", err)
	}
	t.Setenv("HF_HUB_OFFLINE", "")
	// Network down: falls back to the cache.
	f.hub.Close()
	p3, s3, err := Resolve(context.Background(), o)
	if err != nil || p3 != path || s3 != sha {
		t.Fatalf("fallback: %v %v %v", p3, s3, err)
	}
	// Cached() reads the same thing.
	if p4, s4, err := Cached(o); err != nil || p4 != path || s4 != sha {
		t.Fatalf("Cached: %v %v %v", p4, s4, err)
	}
	// Empty cache offline: a clear error.
	oo.CacheDir = t.TempDir()
	if _, _, err := Resolve(context.Background(), oo); !errors.Is(err, ErrOffline) {
		t.Fatalf("empty offline: %v", err)
	}
	// Empty cache, network down: the network error, not a cache hit.
	o.CacheDir = t.TempDir()
	if _, _, err := Resolve(context.Background(), o); err == nil || !strings.Contains(err.Error(), "cannot reach") {
		t.Fatalf("empty + down: %v", err)
	}
}

func TestCopyFallbackSha(t *testing.T) {
	clearEnv(t)
	f := newFakeHub(t, 2048)
	o := f.opts(t)
	path, sha, err := Resolve(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	// Replace the symlink with a copy, as on filesystems without symlinks.
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, f.blob, 0o600); err != nil {
		t.Fatal(err)
	}
	o.Offline = true
	if _, s, err := Resolve(context.Background(), o); err != nil || s != sha {
		t.Fatalf("copied snapshot: %v %v", s, err)
	}
}

func TestStatusMessages(t *testing.T) {
	clearEnv(t)
	cases := []struct {
		status int
		token  string
		want   string
	}{
		{401, "", "model repo is gated or private: request access on the model page, then set HF_TOKEN or run `hf auth login`"},
		{401, testToken, "rejected your token"},
		{403, testToken, "no access to"},
		{404, "", "set HF_TOKEN"},
		{404, testToken, "not found"},
		{500, testToken, "HTTP 500"},
	}
	for _, c := range cases {
		f := newFakeHub(t, 16)
		f.status = c.status
		o := f.opts(t)
		o.Token = c.token
		_, _, err := Resolve(context.Background(), o)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%d token=%v: err = %v, want %q", c.status, c.token != "", err, c.want)
		}
		if auth := c.status == 401 || c.status == 403 || (c.status == 404 && c.token == ""); auth != errors.Is(err, ErrAuth) {
			t.Errorf("%d token=%v: errors.Is(ErrAuth) = %v, want %v", c.status, c.token != "", !auth, auth)
		}
		if err != nil && strings.Contains(err.Error(), testToken) {
			t.Errorf("token leaked: %v", err)
		}
	}
	f := newFakeHub(t, 16)
	f.private = true
	o := f.opts(t)
	o.Token = ""
	if _, _, err := Resolve(context.Background(), o); !errors.Is(err, ErrPrivate) {
		t.Errorf("private without token: %v", err)
	}
}

func TestInsecureRedirectRefused(t *testing.T) {
	clearEnv(t)
	f := newFakeHub(t, 1024)
	f.getRedirect = "http://cdn.example.invalid/blob?token=" + testToken
	_, _, err := Resolve(context.Background(), f.opts(t))
	if err == nil || !strings.Contains(err.Error(), "insecure redirect") {
		t.Fatalf("err = %v", err)
	}
	if strings.Contains(err.Error(), testToken) || strings.Contains(err.Error(), "token=") {
		t.Errorf("error leaks query/token: %v", err)
	}
}

// A same-host HEAD redirect must not downgrade to http: the token travels on
// the next request. The endpoint is https, and the redirect points at the
// same host over plain http.
func TestHeadRedirectDowngradeRefused(t *testing.T) {
	clearEnv(t)
	var hits int
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Header().Set("Location", "http://"+r.Host+r.URL.Path)
		w.WriteHeader(http.StatusFound)
	}))
	t.Cleanup(srv.Close)
	ep, _ := url.Parse(srv.URL)
	r := &resolved{o: Options{Repo: testRepo, Revision: "main"}, token: testToken, endpoint: ep, client: srv.Client()}
	_, err := r.head(context.Background())
	if err == nil || !strings.Contains(err.Error(), "unexpected redirect") {
		t.Fatalf("err = %v, want unexpected redirect", err)
	}
	if hits != 1 {
		t.Fatalf("followed the downgrade redirect: %d requests", hits)
	}
}

func TestCheckRedirectDropsAuthOnHostChange(t *testing.T) {
	mk := func(u string) *http.Request {
		r, _ := http.NewRequest(http.MethodGet, u, nil)
		r.Header.Set("Authorization", "Bearer x")
		return r
	}
	via := []*http.Request{mk("https://huggingface.co/a")}
	same := mk("https://huggingface.co/b")
	if err := checkRedirect(same, via); err != nil || same.Header.Get("Authorization") == "" {
		t.Errorf("same host: %v auth=%q", err, same.Header.Get("Authorization"))
	}
	other := mk("https://cdn-lfs.hf.co/b")
	if err := checkRedirect(other, via); err != nil || other.Header.Get("Authorization") != "" {
		t.Errorf("cross host kept auth: %v", err)
	}
	if err := checkRedirect(mk("http://cdn-lfs.hf.co/b"), via); err == nil {
		t.Error("http redirect accepted")
	}
}

func TestEndpointMustBeHTTPS(t *testing.T) {
	clearEnv(t)
	for _, ep := range []string{"http://huggingface.co", "ftp://x", "https://user:" + testToken + "@hf.co", "not a url"} {
		_, _, err := Resolve(context.Background(), Options{Repo: testRepo, CacheDir: t.TempDir(), Token: testToken, Endpoint: ep})
		if err == nil {
			t.Errorf("%s accepted", ep)
		} else if strings.Contains(err.Error(), testToken) {
			t.Errorf("token leaked: %v", err)
		}
	}
	t.Setenv("HF_ENDPOINT", "http://evil.example")
	if _, _, err := Resolve(context.Background(), Options{Repo: testRepo, CacheDir: t.TempDir()}); err == nil {
		t.Error("insecure HF_ENDPOINT accepted")
	}
}

func TestValidation(t *testing.T) {
	badRepos := []string{"", "noslash", "a/b/c", "../x", "a/..", "a/../b", "/a/b", "a/b/", ".a/b", "a/-b", "a b/c", "a/b\x00", `a\b/c`, "a/b?x", strings.Repeat("a", 300) + "/b"}
	for _, r := range badRepos {
		if ValidateRepo(r) == nil {
			t.Errorf("repo %q accepted", r)
		}
	}
	for _, r := range []string{"vaktex/DOM-0.8B", "a/b", "A1/b_c.d-e"} {
		if err := ValidateRepo(r); err != nil {
			t.Errorf("repo %q rejected: %v", r, err)
		}
	}
	badRevs := []string{"", "..", "../main", "a/../b", `a\b`, "/main", "main/", "a//b", "a\nb", "refs:x"}
	for _, r := range badRevs {
		if ValidateRevision(r) == nil {
			t.Errorf("revision %q accepted", r)
		}
	}
	for _, r := range []string{"main", "v1.0", "refs/pr/1", testCommit} {
		if err := ValidateRevision(r); err != nil {
			t.Errorf("revision %q rejected: %v", r, err)
		}
	}
	clearEnv(t)
	if _, _, err := Resolve(context.Background(), Options{Repo: "../../etc", CacheDir: t.TempDir()}); err == nil {
		t.Error("Resolve accepted a bad repo")
	}
	if _, _, err := Resolve(context.Background(), Options{Repo: testRepo, Revision: "../../x", CacheDir: t.TempDir()}); err == nil {
		t.Error("Resolve accepted a bad revision")
	}
}

func TestTokenSources(t *testing.T) {
	clearEnv(t)
	if _, ok := ResolveToken(Options{}); ok {
		t.Fatal("found a token in a clean env")
	}
	home := t.TempDir()
	t.Setenv("HF_HOME", home)
	if err := os.WriteFile(filepath.Join(home, "token"), []byte("file-token\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if tok, _ := ResolveToken(Options{}); tok != "file-token" {
		t.Errorf("file token = %q", tok)
	}
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "legacy")
	if tok, _ := ResolveToken(Options{}); tok != "legacy" {
		t.Errorf("legacy env = %q", tok)
	}
	t.Setenv("HF_TOKEN", "env")
	if tok, _ := ResolveToken(Options{}); tok != "env" {
		t.Errorf("HF_TOKEN = %q", tok)
	}
	if tok, _ := ResolveToken(Options{Token: "explicit"}); tok != "explicit" {
		t.Errorf("explicit = %q", tok)
	}
}

func TestTokenNeverInErrors(t *testing.T) {
	clearEnv(t)
	// Every failure path we can provoke, with a token set.
	f := newFakeHub(t, 64)
	run := func(name string, o Options) {
		_, _, err := Resolve(context.Background(), o)
		if err != nil && strings.Contains(err.Error(), testToken) {
			t.Errorf("%s: token in error: %v", name, err)
		}
	}
	o := f.opts(t)
	f.status = 401
	run("401", o)
	f.status = 0
	f.sha = strings.Repeat("1", 64)
	run("mismatch", o)
	bad := o
	bad.Repo = testToken + "/x/y"
	run("bad repo", bad)
	bad = o
	bad.Endpoint = "http://" + testToken + ".example"
	run("insecure endpoint", bad)
	down := httptest.NewServer(http.NotFoundHandler())
	down.Close()
	bad = o
	bad.Endpoint = down.URL
	run("down", bad)
	if e := redactErr(fmt.Errorf("wrapped %w: %s", ErrPrivate, testToken), testToken); strings.Contains(e.Error(), testToken) || !errors.Is(e, ErrPrivate) {
		t.Errorf("redactErr: %v", e)
	}
}

func TestCacheDirEnv(t *testing.T) {
	t.Setenv("VAKT_CACHE", "/tmp/vakt-test-cache")
	if CacheDir() != "/tmp/vakt-test-cache/hub" || ScoresDir() != "/tmp/vakt-test-cache/scores" {
		t.Errorf("VAKT_CACHE: hub %s scores %s", CacheDir(), ScoresDir())
	}
	t.Setenv("VAKT_CACHE", "")
	if !strings.HasSuffix(CacheDir(), filepath.Join("vakt", "hub")) {
		t.Errorf("default cache dir = %s", CacheDir())
	}
}

// A hub that accepts the connection but never answers must not hang Resolve:
// the HEAD deadline fires and, with a cached snapshot, the cache is used.
func TestHungHubFallsBackToCache(t *testing.T) {
	clearEnv(t)
	f := newFakeHub(t, 2048)
	o := f.opts(t)
	if _, _, err := Resolve(context.Background(), o); err != nil {
		t.Fatal(err)
	}
	block := make(chan struct{})
	t.Cleanup(func() { close(block) })
	hung := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-block:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(hung.Close)
	old := headTimeout
	headTimeout = 300 * time.Millisecond
	t.Cleanup(func() { headTimeout = old })
	o.Endpoint = hung.URL
	start := time.Now()
	path, sha, err := Resolve(context.Background(), o)
	if err != nil || path == "" || sha != f.sha {
		t.Fatalf("Resolve = %q %q %v, want cached fallback", path, sha, err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Fatalf("took %s; HEAD deadline not applied", d)
	}
}

// A CDN that stalls mid-body trips the idle watchdog instead of hanging.
func TestStalledDownloadTimesOut(t *testing.T) {
	clearEnv(t)
	f := newFakeHub(t, 4096)
	block := make(chan struct{})
	t.Cleanup(func() { close(block) })
	stall := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(len(f.blob)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(f.blob[:100])
		w.(http.Flusher).Flush()
		select {
		case <-block:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(stall.Close)
	f.getRedirect = stall.URL + "/blob"
	old := idleTimeout
	idleTimeout = 300 * time.Millisecond
	t.Cleanup(func() { idleTimeout = old })
	start := time.Now()
	_, _, err := Resolve(context.Background(), f.opts(t))
	if err == nil || !strings.Contains(err.Error(), "stalled") {
		t.Fatalf("err = %v, want stalled", err)
	}
	if d := time.Since(start); d > 10*time.Second {
		t.Fatalf("took %s", d)
	}
}
