// Package hub downloads DOM-0.8B's weights from the Hugging Face Hub into a
// local cache laid out like huggingface_hub's:
//
//	<cache>/models--<owner>--<name>/blobs/<sha256>
//	<cache>/models--<owner>--<name>/snapshots/<commit>/model.safetensors -> ../../blobs/<sha256>
//	<cache>/models--<owner>--<name>/refs/<revision>   (contains <commit>)
//
// Only brand.ModelFile is ever fetched. The token is never logged, echoed in
// an error or written anywhere.
package hub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/vaktex/vakt/internal/brand"
)

// DefaultEndpoint is the Hugging Face Hub.
const DefaultEndpoint = "https://huggingface.co"

// MaxModelBytes refuses absurd downloads (DOM-0.8B fp32 is ~3.2 GB).
const MaxModelBytes int64 = 32 << 30

// Options configure Resolve.
type Options struct {
	Repo     string // default brand.ModelRepo
	Revision string // default "main"
	CacheDir string // default CacheDir()
	Token    string // default: HF_TOKEN, HUGGING_FACE_HUB_TOKEN, then the token file
	Endpoint string // default HF_ENDPOINT or DefaultEndpoint
	Offline  bool   // use the cache only (also HF_HUB_OFFLINE=1)
	Progress func(done, total int64)
	// HTTPClient overrides the transport (tests). Redirect policy is always
	// replaced by the package's own.
	HTTPClient *http.Client
}

var (
	repoRE   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)
	sha256RE = regexp.MustCompile(`^[0-9a-f]{64}$`)
	commitRE = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// Errors callers can test for.
var (
	ErrPrivate  = errors.New("model repo is private: set HF_TOKEN or run `hf auth login`")
	ErrNotFound = errors.New("model not found")
	ErrOffline  = errors.New("model not in the local cache")
)

// CacheDir is $VAKT_CACHE, or <user cache dir>/vakt/hub.
func CacheDir() string {
	if d := os.Getenv("VAKT_CACHE"); d != "" {
		return d
	}
	d, err := os.UserCacheDir()
	if err != nil {
		d = os.TempDir()
	}
	return filepath.Join(d, "vakt", "hub")
}

// ValidateRepo checks a repo id of the form owner/name.
func ValidateRepo(repo string) error {
	if len(repo) > 200 || !repoRE.MatchString(repo) || strings.Contains(repo, "..") {
		return fmt.Errorf("invalid model repo id %q (want owner/name)", printable(repo))
	}
	return nil
}

// ValidateRevision rejects revisions that could escape the cache directory.
func ValidateRevision(rev string) error {
	switch {
	case rev == "", len(rev) > 256:
		return errors.New("invalid revision: empty or too long")
	case strings.Contains(rev, ".."), strings.Contains(rev, `\`), strings.HasPrefix(rev, "/"),
		strings.HasSuffix(rev, "/"), strings.Contains(rev, "//"):
		return fmt.Errorf("invalid revision %q", printable(rev))
	}
	for _, r := range rev {
		if r < 0x20 || r == 0x7f || r == ':' {
			return fmt.Errorf("invalid revision %q", printable(rev))
		}
	}
	return nil
}

// ResolveToken returns the token from o.Token, the environment or the token
// file, and whether one was found. Callers must not print it.
func ResolveToken(o Options) (string, bool) {
	if t := strings.TrimSpace(o.Token); t != "" {
		return t, true
	}
	for _, k := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"} {
		if t := strings.TrimSpace(os.Getenv(k)); t != "" {
			return t, true
		}
	}
	var path string
	if h := os.Getenv("HF_HOME"); h != "" {
		path = filepath.Join(h, "token")
	} else if home, err := os.UserHomeDir(); err == nil {
		path = filepath.Join(home, ".cache", "huggingface", "token")
	}
	if path != "" {
		if b, err := os.ReadFile(path); err == nil { // #nosec G304 G703 -- the well-known HF token location ($HF_HOME/token)
			if t := strings.TrimSpace(string(b)); t != "" {
				return t, true
			}
		}
	}
	return "", false
}

type resolved struct {
	o        Options
	token    string
	endpoint *url.URL
	repoDir  string
	client   *http.Client
}

func prepare(o Options) (*resolved, error) {
	if o.Repo == "" {
		o.Repo = brand.ModelRepo
	}
	if o.Revision == "" {
		o.Revision = "main"
	}
	if o.CacheDir == "" {
		o.CacheDir = CacheDir()
	}
	if o.Endpoint == "" {
		o.Endpoint = os.Getenv("HF_ENDPOINT")
	}
	if o.Endpoint == "" {
		o.Endpoint = DefaultEndpoint
	}
	if v := os.Getenv("HF_HUB_OFFLINE"); v == "1" || strings.EqualFold(v, "true") {
		o.Offline = true
	}
	if err := ValidateRepo(o.Repo); err != nil {
		return nil, err
	}
	if err := ValidateRevision(o.Revision); err != nil {
		return nil, err
	}
	ep, err := url.Parse(strings.TrimRight(o.Endpoint, "/"))
	if err != nil || ep.Host == "" || ep.User != nil || ep.RawQuery != "" || ep.Fragment != "" {
		return nil, errors.New("invalid hub endpoint")
	}
	if !secureURL(ep) {
		return nil, fmt.Errorf("hub endpoint must use https (got %s)", redactURL(ep))
	}
	r := &resolved{o: o, endpoint: ep}
	r.token, _ = ResolveToken(o)
	r.repoDir = filepath.Join(o.CacheDir, "models--"+strings.ReplaceAll(o.Repo, "/", "--"))
	base := http.DefaultClient
	if o.HTTPClient != nil {
		base = o.HTTPClient
	}
	c := *base
	c.CheckRedirect = checkRedirect
	r.client = &c
	return r, nil
}

// isLocal reports whether host is a loopback name, where plain http is
// allowed for tests and local mirrors.
func isLocal(host string) bool {
	h := host
	if hh, _, err := net.SplitHostPort(host); err == nil {
		h = hh
	}
	return h == "localhost" || h == "127.0.0.1" || h == "::1"
}

func secureURL(u *url.URL) bool {
	return u.Scheme == "https" || (u.Scheme == "http" && isLocal(u.Host))
}

// checkRedirect refuses insecure redirects and never forwards the token to
// another host (HF redirects LFS downloads to a CDN with a signed URL).
func checkRedirect(req *http.Request, via []*http.Request) error {
	if len(via) >= 10 {
		return errors.New("too many redirects")
	}
	if !secureURL(req.URL) {
		return policyError{fmt.Errorf("refusing insecure redirect to %s", redactURL(req.URL))}
	}
	if req.URL.Host != via[0].URL.Host {
		req.Header.Del("Authorization")
	}
	return nil
}

// Resolve returns the local path and sha256 of brand.ModelFile for o.Repo at
// o.Revision, downloading it if needed. When offline, or when the hub cannot
// be reached and a snapshot is cached, it uses the cache.
func Resolve(ctx context.Context, o Options) (path, sha256hex string, err error) {
	token, _ := ResolveToken(o)
	defer func() { err = redactErr(err, token) }()
	r, err := prepare(o)
	if err != nil {
		return "", "", err
	}
	if r.o.Offline {
		return r.cached()
	}
	meta, err := r.head(ctx)
	if err != nil {
		var ne netError
		if errors.As(err, &ne) {
			if p, s, cerr := r.cached(); cerr == nil {
				return p, s, nil
			}
		}
		return "", "", err
	}
	if err := os.MkdirAll(filepath.Join(r.repoDir, "blobs"), 0o700); err != nil {
		return "", "", err
	}
	blob := filepath.Join(r.repoDir, "blobs", meta.sha)
	if st, err := os.Stat(blob); err != nil || st.Size() != meta.size {
		if err := r.download(ctx, meta, blob); err != nil {
			return "", "", err
		}
	} else if r.o.Progress != nil {
		r.o.Progress(meta.size, meta.size)
	}
	p, err := r.link(meta, blob)
	if err != nil {
		return "", "", err
	}
	return p, meta.sha, nil
}

// Cached returns the cached model for o without touching the network.
func Cached(o Options) (path, sha256hex string, err error) {
	token, _ := ResolveToken(o)
	defer func() { err = redactErr(err, token) }()
	r, err := prepare(o)
	if err != nil {
		return "", "", err
	}
	return r.cached()
}

func (r *resolved) cached() (string, string, error) {
	commit := r.o.Revision
	if !commitRE.MatchString(commit) {
		b, err := os.ReadFile(filepath.Join(r.repoDir, "refs", filepath.FromSlash(r.o.Revision))) // #nosec G304 -- revision validated
		if err != nil {
			return "", "", fmt.Errorf("%w: %s@%s (run `%s summon`)", ErrOffline, r.o.Repo, r.o.Revision, brand.Binary)
		}
		commit = strings.TrimSpace(string(b))
		if !commitRE.MatchString(commit) {
			return "", "", fmt.Errorf("corrupt cache ref for %s@%s", r.o.Repo, r.o.Revision)
		}
	}
	p := filepath.Join(r.repoDir, "snapshots", commit, brand.ModelFile)
	if _, err := os.Stat(p); err != nil { // #nosec G703 -- commit matched ^[0-9a-f]{40}$
		return "", "", fmt.Errorf("%w: %s@%s (run `%s summon`)", ErrOffline, r.o.Repo, r.o.Revision, brand.Binary)
	}
	if target, err := filepath.EvalSymlinks(p); err == nil {
		if base := filepath.Base(target); sha256RE.MatchString(base) &&
			filepath.Clean(filepath.Dir(target)) == filepath.Clean(mustEval(filepath.Join(r.repoDir, "blobs"))) {
			return p, base, nil
		}
	}
	// A copied snapshot: the sha sits next to it.
	if b, err := os.ReadFile(p + ".sha256"); err == nil { // #nosec G304 G703 -- validated repo/commit inside our cache
		if s := strings.TrimSpace(string(b)); sha256RE.MatchString(s) {
			return p, s, nil
		}
	}
	s, err := hashFile(p)
	if err != nil {
		return "", "", err
	}
	return p, s, nil
}

func mustEval(p string) string {
	if e, err := filepath.EvalSymlinks(p); err == nil {
		return e
	}
	return p
}

type fileMeta struct {
	commit, sha string
	size        int64
}

// policyError is a refusal by our own redirect policy; it is never retried
// and never treated as the hub being unreachable.
type policyError struct{ err error }

func (e policyError) Error() string { return e.err.Error() }
func (e policyError) Unwrap() error { return e.err }

// netError marks transport failures (as opposed to HTTP status errors).
type netError struct{ err error }

func (e netError) Error() string { return "cannot reach the model hub: " + e.err.Error() }
func (e netError) Unwrap() error { return e.err }

func (r *resolved) fileURL(rev string) string {
	u := *r.endpoint
	u.Path = strings.TrimRight(u.Path, "/") + "/" + r.o.Repo + "/resolve/" + url.PathEscape(rev) + "/" + brand.ModelFile
	u.RawPath = ""
	return u.String()
}

func (r *resolved) newRequest(ctx context.Context, method, u string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", brand.Binary+"/"+brand.Version)
	if r.token != "" {
		req.Header.Set("Authorization", "Bearer "+r.token)
	}
	return req, nil
}

// head reads the file metadata without following redirects (the LFS headers
// are on the redirect response itself). Relative redirects (renamed repos)
// are followed on the same host.
func (r *resolved) head(ctx context.Context) (fileMeta, error) {
	u := r.fileURL(r.o.Revision)
	noFollow := *r.client
	noFollow.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	for hop := 0; hop < 5; hop++ {
		req, err := r.newRequest(ctx, http.MethodHead, u)
		if err != nil {
			return fileMeta{}, err
		}
		resp, err := noFollow.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return fileMeta{}, ctx.Err()
			}
			return fileMeta{}, netError{stripURLError(err)}
		}
		_ = resp.Body.Close()
		if err := r.statusErr(resp.StatusCode); err != nil {
			return fileMeta{}, err
		}
		commit := resp.Header.Get("X-Repo-Commit")
		if resp.StatusCode >= 300 && resp.StatusCode < 400 && commit == "" {
			loc, err := resp.Location()
			// Same host only, and never a downgrade to plain http: the token
			// travels on the next HEAD.
			if err != nil || loc.Host != req.URL.Host || loc.Scheme != req.URL.Scheme || !secureURL(loc) {
				return fileMeta{}, errors.New("hub returned an unexpected redirect")
			}
			u = loc.String()
			continue
		}
		sha := normEtag(resp.Header.Get("X-Linked-Etag"))
		if sha == "" {
			sha = normEtag(resp.Header.Get("ETag"))
		}
		sizeStr := resp.Header.Get("X-Linked-Size")
		if sizeStr == "" {
			sizeStr = resp.Header.Get("Content-Length")
		}
		size, err := strconv.ParseInt(sizeStr, 10, 64)
		switch {
		case !commitRE.MatchString(commit):
			return fileMeta{}, errors.New("hub response has no valid X-Repo-Commit")
		case !sha256RE.MatchString(sha):
			return fileMeta{}, errors.New("hub response has no sha256 etag for the model file (is it stored with LFS?)")
		case err != nil || size <= 0 || size > MaxModelBytes:
			return fileMeta{}, errors.New("hub response has an invalid model size")
		}
		return fileMeta{commit: commit, sha: sha, size: size}, nil
	}
	return fileMeta{}, errors.New("too many redirects")
}

func normEtag(s string) string {
	s = strings.TrimPrefix(strings.TrimSpace(s), "W/")
	return strings.ToLower(strings.Trim(s, `"`))
}

func (r *resolved) statusErr(code int) error {
	switch {
	case code == http.StatusUnauthorized:
		if r.token == "" {
			return ErrPrivate
		}
		return fmt.Errorf("the hub rejected your token for %s (HTTP 401): check HF_TOKEN or run `hf auth login`", r.o.Repo)
	case code == http.StatusForbidden:
		return fmt.Errorf("access to %s is forbidden (HTTP 403): request access on the model page, or use a token with read access", r.o.Repo)
	case code == http.StatusNotFound:
		if r.token == "" {
			return fmt.Errorf("%w: %s@%s (if the repo is private, set HF_TOKEN or run `hf auth login`)", ErrNotFound, r.o.Repo, r.o.Revision)
		}
		return fmt.Errorf("%w: %s@%s (check the repo id, revision and token access)", ErrNotFound, r.o.Repo, r.o.Revision)
	case code == http.StatusTooManyRequests:
		return errors.New("the hub is rate limiting requests (HTTP 429): try again shortly")
	case code >= 400:
		return fmt.Errorf("the hub returned HTTP %d", code)
	}
	return nil
}

const chunk = 1 << 20

// download streams the blob to blobs/<sha>.incomplete, resuming with a Range
// request, verifies size and sha256, then renames it into place.
func (r *resolved) download(ctx context.Context, m fileMeta, blob string) error {
	tmp := blob + ".incomplete"
	f, err := os.OpenFile(tmp, os.O_RDWR|os.O_CREATE, 0o600) // #nosec G304 -- sha validated, inside our cache
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	have, err := io.Copy(h, f) // hash what a previous run left behind
	if err != nil {
		return err
	}
	if have > m.size {
		if err := restart(f, &have); err != nil {
			return err
		}
		h = sha256.New()
	}
	for attempt := 0; ; attempt++ {
		err := r.fetch(ctx, m, f, &have, &h)
		if err == nil {
			break
		}
		var ne netError
		if ctx.Err() != nil || !errors.As(err, &ne) || attempt >= 3 {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Duration(attempt+1) * time.Second):
		}
	}
	if have != m.size {
		return fmt.Errorf("download incomplete: got %d of %d bytes", have, m.size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != m.sha {
		_ = f.Close()
		_ = os.Remove(tmp)
		return fmt.Errorf("sha256 mismatch for %s: expected %s, got %s (partial download removed)", brand.ModelFile, m.sha, got)
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, blob)
}

func restart(f *os.File, have *int64) error {
	if err := f.Truncate(0); err != nil {
		return err
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	*have = 0
	return nil
}

func (r *resolved) fetch(ctx context.Context, m fileMeta, f *os.File, have *int64, h *hash.Hash) error {
	if *have == m.size {
		return nil
	}
	req, err := r.newRequest(ctx, http.MethodGet, r.fileURL(m.commit))
	if err != nil {
		return err
	}
	if *have > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", *have))
	}
	resp, err := r.client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if se := stripURLError(err); errors.As(se, new(policyError)) {
			return se
		}
		return netError{stripURLError(err)}
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusPartialContent && *have > 0:
		// resume
	case resp.StatusCode == http.StatusOK:
		if *have > 0 { // server ignored Range: start over
			if err := restart(f, have); err != nil {
				return err
			}
			*h = sha256.New()
		}
	case resp.StatusCode == http.StatusRequestedRangeNotSatisfiable:
		if err := restart(f, have); err != nil {
			return err
		}
		*h = sha256.New()
		return netError{errors.New("range not satisfiable; restarting")}
	default:
		if err := r.statusErr(resp.StatusCode); err != nil {
			return err
		}
		return fmt.Errorf("unexpected HTTP %d from the hub", resp.StatusCode)
	}
	if _, err := f.Seek(*have, io.SeekStart); err != nil {
		return err
	}
	body := io.LimitReader(resp.Body, m.size-*have+1)
	buf := make([]byte, chunk)
	for {
		n, rerr := body.Read(buf)
		if n > 0 {
			if *have+int64(n) > m.size {
				return errors.New("hub sent more data than the advertised size")
			}
			if _, err := f.Write(buf[:n]); err != nil {
				return err
			}
			_, _ = (*h).Write(buf[:n]) // hash.Hash.Write never returns an error
			*have += int64(n)
			if r.o.Progress != nil {
				r.o.Progress(*have, m.size)
			}
		}
		if rerr == io.EOF {
			return nil
		}
		if rerr != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return netError{rerr}
		}
	}
}

// link creates snapshots/<commit>/model.safetensors and refs/<revision>.
func (r *resolved) link(m fileMeta, blob string) (string, error) {
	snapDir := filepath.Join(r.repoDir, "snapshots", m.commit)
	if err := os.MkdirAll(snapDir, 0o700); err != nil {
		return "", err
	}
	p := filepath.Join(snapDir, brand.ModelFile)
	if target, err := filepath.EvalSymlinks(p); err == nil && filepath.Base(target) == m.sha {
		// already linked
	} else {
		_ = os.Remove(p)
		rel := filepath.Join("..", "..", "blobs", m.sha)
		if err := os.Symlink(rel, p); err != nil {
			if err := copyFile(blob, p); err != nil {
				return "", err
			}
			if err := os.WriteFile(p+".sha256", []byte(m.sha+"\n"), 0o600); err != nil {
				return "", err
			}
		}
	}
	if r.o.Revision != m.commit {
		ref := filepath.Join(r.repoDir, "refs", filepath.FromSlash(r.o.Revision))
		if err := os.MkdirAll(filepath.Dir(ref), 0o700); err != nil {
			return "", err
		}
		if err := writeAtomic(ref, []byte(m.commit)); err != nil {
			return "", err
		}
	}
	return p, nil
}

func writeAtomic(path string, data []byte) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src) // #nosec G304 -- inside our cache
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst+".tmp", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600) // #nosec G304 -- inside our cache
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Rename(dst+".tmp", dst)
}

func hashFile(p string) (string, error) {
	f, err := os.Open(p) // #nosec G304 G703 -- path built from validated repo/commit inside our cache
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// stripURLError drops the URL (which may carry signed CDN query strings)
// from transport errors and keeps only the cause.
func stripURLError(err error) error {
	var pe policyError
	if errors.As(err, &pe) {
		return pe
	}
	var ue *url.Error
	if errors.As(err, &ue) {
		return fmt.Errorf("%s %s: %w", ue.Op, redactURLString(ue.URL), ue.Err)
	}
	return err
}

func redactURL(u *url.URL) string {
	c := *u
	c.User = nil
	c.RawQuery = ""
	c.Fragment = ""
	return c.String()
}

func redactURLString(s string) string {
	u, err := url.Parse(s)
	if err != nil {
		return "<url>"
	}
	return redactURL(u)
}

// redactedError hides the original error (whose message contained the
// token) but still matches sentinel errors via errors.Is.
type redactedError struct {
	msg  string
	orig error
}

func (e *redactedError) Error() string        { return e.msg }
func (e *redactedError) Is(target error) bool { return errors.Is(e.orig, target) }

// redactErr guarantees the token never appears in an error string.
func redactErr(err error, token string) error {
	if err == nil || token == "" || !strings.Contains(err.Error(), token) {
		return err
	}
	return &redactedError{msg: strings.ReplaceAll(err.Error(), token, "***"), orig: err}
}

func printable(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r < 0x20 || r == 0x7f || (r >= 0x80 && r < 0xa0) {
			b.WriteRune('?')
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}
