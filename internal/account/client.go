package account

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/vaktex/vakt/internal/brand"
)

// DefaultBaseURL is used when VAKT_CLOUD_URL is unset.
const DefaultBaseURL = "https://vaktex.com"

var (
	// ErrDenied: the user refused the sign-in in the browser.
	ErrDenied = errors.New("Sign-in was denied in the browser.")
	// ErrExpired: the device code ran out before it was approved.
	ErrExpired = errors.New("The code expired. Run " + brand.Binary + " login again.")
	// ErrUnauthorized: the server no longer accepts the stored token.
	ErrUnauthorized = errors.New("Your sign-in is no longer valid. Run " + brand.Binary + " login.")
)

// BaseURL is $VAKT_CLOUD_URL or DefaultBaseURL, without trailing slashes.
func BaseURL() string {
	if u := strings.TrimSpace(os.Getenv("VAKT_CLOUD_URL")); u != "" {
		return strings.TrimRight(u, "/")
	}
	return DefaultBaseURL
}

// Client talks to the Vaktex account API.
type Client struct {
	BaseURL string
	HTTP    *http.Client
	// Now and Sleep drive the device-flow loop; tests replace them.
	Now   func() time.Time
	Sleep func(ctx context.Context, d time.Duration) error
}

// NewClient returns a client for base (trailing slashes trimmed). It refuses
// plain http except to loopback, so a token is never sent in the clear.
func NewClient(base string) (*Client, error) {
	base = strings.TrimRight(base, "/")
	u, err := url.Parse(base)
	if err != nil || u.Host == "" {
		return nil, fmt.Errorf("invalid Vaktex URL %q", base)
	}
	if !(u.Scheme == "https" || (u.Scheme == "http" && isLocal(u.Host))) {
		return nil, fmt.Errorf("refusing non-https Vaktex URL %q", base)
	}
	return &Client{
		BaseURL: base,
		HTTP: &http.Client{
			Timeout: 30 * time.Second,
			// Never follow redirects: the token must only go to BaseURL.
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
		Now:   time.Now,
		Sleep: sleepCtx,
	}, nil
}

func isLocal(host string) bool {
	h := host
	if hh, _, err := net.SplitHostPort(host); err == nil {
		h = hh
	}
	return h == "localhost" || h == "127.0.0.1" || h == "::1"
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// DeviceCode is the server's answer to a device-code request.
type DeviceCode struct {
	DeviceCode              string `json:"device_code"`
	UserCode                string `json:"user_code"`
	VerificationURI         string `json:"verification_uri"`
	VerificationURIComplete string `json:"verification_uri_complete"`
	ExpiresIn               int    `json:"expires_in"`
	Interval                int    `json:"interval"`
}

// Token is a successful device-flow result.
type Token struct {
	Token string `json:"token"`
	Email string `json:"email"`
	Plan  string `json:"plan"`
}

// Account is GET /api/vakt/account.
type Account struct {
	Plan          string `json:"plan"`
	EmailVerified bool   `json:"emailVerified"`
	Month         *struct {
		Month    string  `json:"month"`
		SpentUSD float64 `json:"spentUsd"`
		ResetsAt string  `json:"resetsAt"`
	} `json:"month"`
	Today *struct {
		Requests int    `json:"requests"`
		Limit    int    `json:"limit"`
		ResetsAt string `json:"resetsAt"`
	} `json:"today"`
}

// apiError is a non-200 response.
type apiError struct {
	Status int
	Code   string // the body's "error", if any
}

func (e *apiError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("Vaktex server error: %s (HTTP %d)", e.Code, e.Status)
	}
	return fmt.Sprintf("Vaktex server error (HTTP %d)", e.Status)
}

// do sends a request and decodes a 200 body into out. raw, if non-nil,
// receives the body bytes.
func (c *Client) do(ctx context.Context, method, path, token string, body any, out any, raw *[]byte) error {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, rd)
	if err != nil {
		return err
	}
	req.Header.Set("x-vakt-client", "cli/"+brand.Version)
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("cannot reach %s: %w", c.BaseURL, unwrapURL(err))
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("reading the response from %s: %w", c.BaseURL, err)
	}
	if resp.StatusCode != http.StatusOK {
		var e struct {
			Error string `json:"error"`
		}
		_ = json.Unmarshal(b, &e)
		return &apiError{Status: resp.StatusCode, Code: e.Error}
	}
	if raw != nil {
		*raw = b
	}
	if out != nil {
		if err := json.Unmarshal(b, out); err != nil {
			return fmt.Errorf("unexpected response from %s: %w", c.BaseURL, err)
		}
	}
	return nil
}

func unwrapURL(err error) error {
	var ue *url.Error
	if errors.As(err, &ue) {
		return ue.Err
	}
	return err
}

// RequestCode starts the device flow. Only the device name is sent.
func (c *Client) RequestCode(ctx context.Context, deviceName string) (*DeviceCode, error) {
	var dc DeviceCode
	if err := c.do(ctx, http.MethodPost, "/api/vakt/device/code", "", map[string]string{"device_name": deviceName}, &dc, nil); err != nil {
		return nil, err
	}
	if dc.DeviceCode == "" || dc.UserCode == "" || dc.VerificationURI == "" {
		return nil, fmt.Errorf("unexpected response from %s: missing device code", c.BaseURL)
	}
	return &dc, nil
}

// maxTransient is how many consecutive unexpected failures the poll loop
// tolerates before giving up.
const maxTransient = 3

// WaitForToken polls until the user approves or denies the code, the code
// expires, or ctx is cancelled. It honours interval, slow_down and expires_in.
func (c *Client) WaitForToken(ctx context.Context, dc *DeviceCode) (*Token, error) {
	interval := time.Duration(dc.Interval) * time.Second
	if interval <= 0 {
		interval = 5 * time.Second
	}
	expiresIn := time.Duration(dc.ExpiresIn) * time.Second
	if expiresIn <= 0 {
		expiresIn = 10 * time.Minute
	}
	deadline := c.Now().Add(expiresIn)
	transient := 0
	wait := interval
	for {
		if err := c.Sleep(ctx, wait); err != nil {
			return nil, err
		}
		if !c.Now().Before(deadline) {
			return nil, ErrExpired
		}
		var tok Token
		err := c.do(ctx, http.MethodPost, "/api/vakt/device/token", "", map[string]string{"device_code": dc.DeviceCode}, &tok, nil)
		if err == nil {
			if tok.Token == "" {
				return nil, fmt.Errorf("unexpected response from %s: missing token", c.BaseURL)
			}
			return &tok, nil
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		wait = interval
		var ae *apiError
		if errors.As(err, &ae) && ae.Status == http.StatusBadRequest {
			switch ae.Code {
			case "authorization_pending":
				transient = 0
				continue
			case "slow_down":
				transient = 0
				interval += 5 * time.Second
				wait = interval
				continue
			case "access_denied":
				return nil, ErrDenied
			case "expired_token":
				return nil, ErrExpired
			}
		}
		transient++
		if transient > maxTransient {
			return nil, err
		}
		wait = interval * time.Duration(1<<transient) // back off: 2x, 4x, 8x
	}
}

// Account fetches the account summary. raw is the response body as sent.
func (c *Client) Account(ctx context.Context, token string) (acct *Account, raw []byte, err error) {
	var a Account
	err = c.do(ctx, http.MethodGet, "/api/vakt/account", token, nil, &a, &raw)
	var ae *apiError
	if errors.As(err, &ae) && ae.Status == http.StatusUnauthorized {
		return nil, nil, ErrUnauthorized
	}
	if err != nil {
		return nil, nil, err
	}
	return &a, raw, nil
}

// Revoke invalidates token on the server.
func (c *Client) Revoke(ctx context.Context, token string) error {
	return c.do(ctx, http.MethodPost, "/api/vakt/auth/revoke", token, struct{}{}, nil, nil)
}
