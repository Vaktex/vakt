// Package account signs the CLI in to a Vaktex account and keeps the
// resulting token on disk. Signing in sends only the device name; no code,
// paths or scan results ever leave the machine through this package.
package account

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"time"

	"github.com/vaktex/vakt/internal/brand"
)

// ErrNotSignedIn means there is no credentials file.
var ErrNotSignedIn = errors.New("not signed in")

// Credentials is what `vakt login` stores.
type Credentials struct {
	Token     string    `json:"token"`
	Email     string    `json:"email"`
	Plan      string    `json:"plan"`
	BaseURL   string    `json:"base_url"`
	CreatedAt time.Time `json:"created_at"`
}

// Dir is $VAKT_CONFIG_DIR, or the user config dir plus "vakt".
func Dir() (string, error) {
	if d := os.Getenv("VAKT_CONFIG_DIR"); d != "" {
		return d, nil
	}
	d, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("finding the config directory: %w", err)
	}
	return filepath.Join(d, brand.Binary), nil
}

// Path is the credentials file.
func Path() (string, error) {
	d, err := Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(d, "credentials.json"), nil
}

// Load reads the credentials file. It returns ErrNotSignedIn when there is none.
func Load() (*Credentials, error) {
	p, err := Path()
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(p) // #nosec G304 -- our own config file
	if errors.Is(err, fs.ErrNotExist) {
		return nil, ErrNotSignedIn
	}
	if err != nil {
		return nil, err
	}
	var c Credentials
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("reading %s: %w (run `%s logout`, then sign in again)", p, err, brand.Binary)
	}
	if c.Token == "" {
		return nil, ErrNotSignedIn
	}
	return &c, nil
}

// Save writes c atomically with mode 0600 in a 0700 directory.
func Save(c *Credentials) error {
	p, err := Path()
	if err != nil {
		return err
	}
	dir := filepath.Dir(p)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".credentials-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(append(b, '\n')); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), p)
}

// Delete removes the credentials file. It reports whether one existed.
func Delete() (bool, error) {
	p, err := Path()
	if err != nil {
		return false, err
	}
	err = os.Remove(p)
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	return err == nil, err
}
