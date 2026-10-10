// Package results keeps a project's past scan results in <root>/.vakt, keyed
// by each file's SHA-256, so they can be shown again without rescanning and
// dropped as soon as the file changes.
//
// The directory sits inside a tree that may be untrusted (a cloned repo can
// ship its own .vakt). Two rules follow:
//
//   - The results file is signed with HMAC-SHA256 under a per-user key kept
//     outside the tree (hub.RootDir()/results.key). A file the key did not
//     sign is ignored, so a repo cannot plant "clean" results to hide
//     findings.
//   - All access goes through an os.Root on the project, so a .vakt that is a
//     symlink cannot make vakt read or write outside the project.
//
// Only scores, spans and names are stored (no source), as in the score cache.
package results

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/vaktex/vakt/internal/report"
)

// Dir is the results directory under a project root.
const Dir = ".vakt"

const (
	fileName = "results.json"
	version  = 1
	// maxFileBytes caps the results file vakt will read.
	maxFileBytes = 64 << 20
)

// Entry is one file's results at the content hash they were computed for.
type Entry struct {
	SHA256 string        `json:"sha256"`
	Units  []report.Unit `json:"units"`
}

// Data is a project's stored results.
type Data struct {
	Version int              `json:"version"`
	Model   report.Model     `json:"model"`
	Files   map[string]Entry `json:"files"`
}

type signed struct {
	Body json.RawMessage `json:"body"`
	MAC  string          `json:"mac"`
}

// Hash is the content hash stored for a file.
func Hash(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// Key returns the per-user signing key in dir, creating it (0600) if needed.
func Key(dir string) ([]byte, error) {
	p := filepath.Join(dir, "results.key")
	if k, err := os.ReadFile(p); err == nil && len(k) == 32 { // #nosec G304 -- vakt's own cache dir
		return k, nil
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	k := make([]byte, 32)
	if _, err := rand.Read(k); err != nil {
		return nil, err
	}
	// O_EXCL: if another vakt created the key meanwhile, use theirs.
	f, err := os.OpenFile(p, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600) // #nosec G304 -- vakt's own cache dir
	if errors.Is(err, fs.ErrExist) {
		k, err := os.ReadFile(p) // #nosec G304 -- vakt's own cache dir
		if err != nil || len(k) != 32 {
			return nil, fmt.Errorf("results key %s is unreadable or malformed", p)
		}
		return k, nil
	}
	if err != nil {
		return nil, err
	}
	if _, err := f.Write(k); err != nil {
		_ = f.Close()
		return nil, err
	}
	return k, f.Close()
}

func mac(key, body []byte) string {
	m := hmac.New(sha256.New, key)
	m.Write(body)
	return hex.EncodeToString(m.Sum(nil))
}

// CleanRel reports whether rel is a clean, slash-separated path inside the root.
func CleanRel(rel string) bool {
	return rel != "" && !strings.HasPrefix(rel, "/") && !strings.Contains(rel, "\\") &&
		path.Clean(rel) == rel && rel != "." && !strings.HasPrefix(rel, "../") && rel != ".."
}

// Load returns the project's stored results, or empty Data when there are
// none or they were not signed with key. Only a failure to open root is an
// error: a missing, foreign or corrupt results file just means "no results".
func Load(root string, key []byte) (*Data, error) {
	r, err := os.OpenRoot(root)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	return load(r, key), nil
}

func empty() *Data { return &Data{Version: version, Files: map[string]Entry{}} }

func load(r *os.Root, key []byte) *Data {
	f, err := r.Open(path.Join(Dir, fileName))
	if err != nil {
		return empty()
	}
	defer f.Close()
	if st, err := f.Stat(); err != nil || !st.Mode().IsRegular() || st.Size() > maxFileBytes {
		return empty()
	}
	var s signed
	if err := json.NewDecoder(f).Decode(&s); err != nil || !hmac.Equal([]byte(mac(key, s.Body)), []byte(s.MAC)) {
		return empty()
	}
	var d Data
	if err := json.Unmarshal(s.Body, &d); err != nil || d.Version != version || d.Files == nil {
		return empty()
	}
	for rel := range d.Files {
		if !CleanRel(rel) {
			delete(d.Files, rel)
		}
	}
	return &d
}

// Update merges files into the stored results and writes them back. With
// replaceAll (a scan of the whole root), entries for other files are
// dropped. The write is atomic and stays inside root.
func Update(root string, key []byte, model report.Model, files map[string]Entry, replaceAll bool) error {
	r, err := os.OpenRoot(root)
	if err != nil {
		return err
	}
	defer r.Close()
	d := load(r, key)
	if replaceAll || d.Model != model {
		// Results from another model (or precision) are not comparable.
		d.Files = map[string]Entry{}
	}
	d.Model = model
	for rel, e := range files {
		if CleanRel(rel) {
			d.Files[rel] = e
		}
	}
	body, err := json.Marshal(d)
	if err != nil {
		return err
	}
	out, err := json.Marshal(signed{Body: body, MAC: mac(key, body)})
	if err != nil {
		return err
	}
	if err := r.Mkdir(Dir, 0o700); err != nil && !errors.Is(err, fs.ErrExist) {
		return fmt.Errorf("create %s: %w", Dir, err)
	}
	if st, err := r.Lstat(Dir); err != nil || !st.IsDir() {
		return fmt.Errorf("%s is not a directory", Dir)
	}
	// Never commit results: the directory ignores itself.
	if _, err := r.Lstat(path.Join(Dir, ".gitignore")); errors.Is(err, fs.ErrNotExist) {
		_ = r.WriteFile(path.Join(Dir, ".gitignore"), []byte("# Created by vakt: past scan results, never commit them.\n*\n"), 0o600)
	}
	tmp := path.Join(Dir, fileName+".tmp")
	if err := r.WriteFile(tmp, out, 0o600); err != nil {
		return err
	}
	if err := r.Rename(tmp, path.Join(Dir, fileName)); err != nil {
		_ = r.Remove(tmp)
		return err
	}
	return nil
}

// Fresh returns the stored units of files whose current contents still hash
// to what was scored, sorted like a report. Files that changed, vanished or
// grew past maxBytes are left out (they need a rescan).
func Fresh(root string, d *Data, maxBytes int64) ([]report.Unit, error) {
	r, err := os.OpenRoot(root)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	rels := make([]string, 0, len(d.Files))
	for rel := range d.Files {
		rels = append(rels, rel)
	}
	sort.Strings(rels)
	units := []report.Unit{}
	for _, rel := range rels {
		st, err := r.Stat(rel)
		if err != nil || !st.Mode().IsRegular() || st.Size() > maxBytes {
			continue
		}
		data, err := r.ReadFile(rel)
		if err != nil || Hash(data) != d.Files[rel].SHA256 {
			continue
		}
		units = append(units, d.Files[rel].Units...)
	}
	return units, nil
}

// ByFile groups a report's units into entries, hashing each file's contents
// from hashes (rel -> sha). Files with no units get an empty entry: "scanned,
// nothing to report" is a result too.
func ByFile(units []report.Unit, hashes map[string]string) map[string]Entry {
	out := make(map[string]Entry, len(hashes))
	for rel, sha := range hashes {
		out[rel] = Entry{SHA256: sha, Units: []report.Unit{}}
	}
	for _, u := range units {
		e, ok := out[u.File]
		if !ok {
			continue // no hash: the file was not read in this scan
		}
		e.Units = append(e.Units, u)
		out[u.File] = e
	}
	return out
}
