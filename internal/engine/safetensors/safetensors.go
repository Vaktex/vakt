// Package safetensors validates the header of an untrusted .safetensors file
// before any tensor data is handed to a numeric backend, and computes the
// file's SHA-256.
//
// File layout: an 8-byte little-endian uint64 header length N, then N bytes
// of UTF-8 JSON, then the data buffer. Tensor data_offsets are relative to
// the start of the data buffer.
//
// The validator is deliberately stricter than most loaders:
//   - the header length is bounded and must fit in the file;
//   - the header must be valid UTF-8 and exactly one JSON object (trailing
//     whitespace padding is allowed, anything else is not);
//   - duplicate keys are rejected at every level;
//   - unknown per-tensor fields are rejected;
//   - "__metadata__" must be a string->string map;
//   - dtypes must be in an allowlist, shapes are bounded and non-negative,
//     byte sizes are computed with overflow checks and must match the
//     offsets exactly;
//   - tensors must tile the data buffer exactly: no overlap, no gaps, no
//     trailing bytes (the same rule the reference Rust implementation uses).
//
// This package is pure Go and has no build tags.
package safetensors

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"unicode/utf8"
)

// DType is a safetensors element type, e.g. "F32".
type DType string

// Common dtypes.
const (
	F64  DType = "F64"
	F32  DType = "F32"
	F16  DType = "F16"
	BF16 DType = "BF16"
	I64  DType = "I64"
	I32  DType = "I32"
	I16  DType = "I16"
	I8   DType = "I8"
	U64  DType = "U64"
	U32  DType = "U32"
	U16  DType = "U16"
	U8   DType = "U8"
	Bool DType = "BOOL"
	F8E4 DType = "F8_E4M3"
	F8E5 DType = "F8_E5M2"
)

// dtypeSizes maps every dtype this package knows how to size to its element
// size in bytes. Sub-byte dtypes are intentionally absent.
var dtypeSizes = map[DType]uint64{
	F64: 8, F32: 4, F16: 2, BF16: 2,
	I64: 8, I32: 4, I16: 2, I8: 1,
	U64: 8, U32: 4, U16: 2, U8: 1,
	Bool: 1, F8E4: 1, F8E5: 1,
}

// Size returns the element size in bytes, or 0 if the dtype is unknown.
func (d DType) Size() int64 { return int64(dtypeSizes[d]) } // #nosec G115 -- table values are <= 8

// Tensor describes one tensor entry. Begin and End are byte offsets relative
// to the start of the data buffer (Header.DataStart in the file).
type Tensor struct {
	Name  string
	DType DType
	Shape []int64
	Begin int64
	End   int64
}

// Header is a validated safetensors header.
type Header struct {
	Tensors  map[string]Tensor
	Metadata map[string]string
	// DataStart is the absolute file offset of the data buffer (8 + N).
	DataStart int64
	// FileSize is the size of the file in bytes at validation time.
	FileSize int64
}

// Limits bounds what ReadHeader accepts. Zero values select the defaults.
type Limits struct {
	MaxHeaderBytes int64   // default 100 MiB
	MaxTensors     int     // default 4096
	MaxDims        int     // default 8
	AllowedDTypes  []DType // default F32, BF16, F16
}

// Defaults.
const (
	DefaultMaxHeaderBytes = 100 << 20
	DefaultMaxTensors     = 4096
	DefaultMaxDims        = 8
	// MaxNameBytes is the maximum tensor name length.
	MaxNameBytes = 512
	metadataKey  = "__metadata__"
)

// DefaultAllowedDTypes is the default dtype allowlist.
func DefaultAllowedDTypes() []DType { return []DType{F32, BF16, F16} }

func (l Limits) withDefaults() Limits {
	if l.MaxHeaderBytes <= 0 {
		l.MaxHeaderBytes = DefaultMaxHeaderBytes
	}
	if l.MaxTensors <= 0 {
		l.MaxTensors = DefaultMaxTensors
	}
	if l.MaxDims <= 0 {
		l.MaxDims = DefaultMaxDims
	}
	if len(l.AllowedDTypes) == 0 {
		l.AllowedDTypes = DefaultAllowedDTypes()
	}
	return l
}

// Sentinel errors; every error returned by ReadHeader wraps exactly one.
var (
	ErrNotRegular      = errors.New("safetensors: not a regular file")
	ErrTruncated       = errors.New("safetensors: file truncated")
	ErrHeaderTooLarge  = errors.New("safetensors: header too large")
	ErrInvalidHeader   = errors.New("safetensors: invalid header JSON")
	ErrDuplicateKey    = errors.New("safetensors: duplicate key")
	ErrBadMetadata     = errors.New("safetensors: invalid __metadata__")
	ErrBadName         = errors.New("safetensors: invalid tensor name")
	ErrTooManyTensors  = errors.New("safetensors: too many tensors")
	ErrBadDType        = errors.New("safetensors: dtype not allowed")
	ErrBadShape        = errors.New("safetensors: invalid shape")
	ErrOverflow        = errors.New("safetensors: tensor size overflows")
	ErrSizeMismatch    = errors.New("safetensors: tensor size does not match offsets")
	ErrBadOffsets      = errors.New("safetensors: invalid data_offsets")
	ErrOverlap         = errors.New("safetensors: overlapping tensors")
	ErrNotContiguous   = errors.New("safetensors: data buffer not contiguous")
	ErrMissingTensor   = errors.New("safetensors: missing tensor")
	ErrShapeMismatch   = errors.New("safetensors: shape mismatch")
	ErrDTypeNotAllowed = errors.New("safetensors: dtype not accepted for tensor")
)

// ReadHeader opens path, validates its safetensors header against lim and
// returns it. It reads only the 8-byte prefix and the header; tensor data is
// never read.
func ReadHeader(path string, lim Limits) (*Header, error) {
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("safetensors: open: %w", err)
	}
	defer f.Close() //nolint:errcheck // read-only

	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("safetensors: stat: %w", err)
	}
	if !st.Mode().IsRegular() {
		return nil, fmt.Errorf("%w: %s", ErrNotRegular, st.Mode())
	}
	return readHeader(f, st.Size(), lim)
}

func readHeader(r io.Reader, fileSize int64, lim Limits) (*Header, error) {
	lim = lim.withDefaults()

	if fileSize < 8 {
		return nil, fmt.Errorf("%w: %d bytes, need at least 8", ErrTruncated, fileSize)
	}
	var prefix [8]byte
	if _, err := io.ReadFull(r, prefix[:]); err != nil {
		return nil, fmt.Errorf("%w: reading header length: %v", ErrTruncated, err)
	}
	n := binary.LittleEndian.Uint64(prefix[:])
	if n > uint64(lim.MaxHeaderBytes) { // #nosec G115 -- MaxHeaderBytes > 0 after defaults
		return nil, fmt.Errorf("%w: %d bytes > limit %d", ErrHeaderTooLarge, n, lim.MaxHeaderBytes)
	}
	if n > uint64(fileSize-8) { // #nosec G115 -- fileSize >= 8 checked above
		return nil, fmt.Errorf("%w: header length %d exceeds file size %d - 8", ErrTruncated, n, fileSize)
	}
	if n < 2 {
		return nil, fmt.Errorf("%w: header length %d", ErrInvalidHeader, n)
	}
	hlen := int64(n) // #nosec G115 -- n <= MaxHeaderBytes (int64)
	buf := make([]byte, hlen)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, fmt.Errorf("%w: reading header: %v", ErrTruncated, err)
	}
	if !utf8.Valid(buf) {
		return nil, fmt.Errorf("%w: not valid UTF-8", ErrInvalidHeader)
	}

	h := &Header{
		Tensors:   make(map[string]Tensor),
		DataStart: 8 + hlen,
		FileSize:  fileSize,
	}
	p := &parser{dec: json.NewDecoder(bytes.NewReader(buf)), lim: lim}
	p.dec.UseNumber()
	if err := p.parseHeader(h); err != nil {
		return nil, err
	}
	if err := checkLayout(h); err != nil {
		return nil, err
	}
	return h, nil
}

type parser struct {
	dec *json.Decoder
	lim Limits
}

func (p *parser) token() (json.Token, error) {
	t, err := p.dec.Token()
	if err != nil {
		if errors.Is(err, io.EOF) {
			err = io.ErrUnexpectedEOF
		}
		return nil, fmt.Errorf("%w: %v", ErrInvalidHeader, err)
	}
	return t, nil
}

func (p *parser) delim(want json.Delim, ctx string) error {
	t, err := p.token()
	if err != nil {
		return err
	}
	if d, ok := t.(json.Delim); !ok || d != want {
		return fmt.Errorf("%w: %s: expected %q, got %s", ErrInvalidHeader, ctx, want, describe(t))
	}
	return nil
}

func (p *parser) str(ctx string) (string, error) {
	t, err := p.token()
	if err != nil {
		return "", err
	}
	s, ok := t.(string)
	if !ok {
		return "", fmt.Errorf("%w: %s: expected string, got %s", ErrInvalidHeader, ctx, describe(t))
	}
	return s, nil
}

func describe(t json.Token) string {
	switch v := t.(type) {
	case json.Delim:
		return strconv.Quote(v.String())
	case string:
		return "string"
	case json.Number:
		return "number"
	case bool:
		return "bool"
	case nil:
		return "null"
	default:
		return fmt.Sprintf("%T", t)
	}
}

func (p *parser) parseHeader(h *Header) error {
	if err := p.delim('{', "header"); err != nil {
		return err
	}
	seen := make(map[string]struct{})
	for p.dec.More() {
		key, err := p.str("header key")
		if err != nil {
			return err
		}
		if _, dup := seen[key]; dup {
			return fmt.Errorf("%w: %q", ErrDuplicateKey, key)
		}
		seen[key] = struct{}{}

		if key == metadataKey {
			md, err := p.parseMetadata()
			if err != nil {
				return err
			}
			h.Metadata = md
			continue
		}
		if err := checkName(key); err != nil {
			return err
		}
		if len(h.Tensors) >= p.lim.MaxTensors {
			return fmt.Errorf("%w: more than %d", ErrTooManyTensors, p.lim.MaxTensors)
		}
		t, err := p.parseTensor(key)
		if err != nil {
			return err
		}
		h.Tensors[key] = t
	}
	if err := p.delim('}', "header"); err != nil {
		return err
	}
	// Exactly one value: only whitespace may follow.
	if t, err := p.dec.Token(); !errors.Is(err, io.EOF) {
		if err != nil {
			return fmt.Errorf("%w: trailing data: %v", ErrInvalidHeader, err)
		}
		return fmt.Errorf("%w: trailing data after header object: %s", ErrInvalidHeader, describe(t))
	}
	return nil
}

func (p *parser) parseMetadata() (map[string]string, error) {
	t, err := p.token()
	if err != nil {
		return nil, err
	}
	if d, ok := t.(json.Delim); !ok || d != '{' {
		return nil, fmt.Errorf("%w: expected object, got %s", ErrBadMetadata, describe(t))
	}
	md := make(map[string]string)
	for p.dec.More() {
		k, err := p.str("__metadata__ key")
		if err != nil {
			return nil, err
		}
		if _, dup := md[k]; dup {
			return nil, fmt.Errorf("%w: __metadata__.%q", ErrDuplicateKey, k)
		}
		t, err := p.token()
		if err != nil {
			return nil, err
		}
		v, ok := t.(string)
		if !ok {
			return nil, fmt.Errorf("%w: value of %q is %s, want string", ErrBadMetadata, k, describe(t))
		}
		md[k] = v
	}
	if err := p.delim('}', "__metadata__"); err != nil {
		return nil, err
	}
	return md, nil
}

func checkName(name string) error {
	if name == "" {
		return fmt.Errorf("%w: empty", ErrBadName)
	}
	if len(name) > MaxNameBytes {
		return fmt.Errorf("%w: %d bytes > %d", ErrBadName, len(name), MaxNameBytes)
	}
	for i := 0; i < len(name); i++ {
		if c := name[i]; c < 0x20 || c > 0x7e {
			return fmt.Errorf("%w: %q: non-printable-ASCII byte 0x%02x at %d", ErrBadName, name, c, i)
		}
	}
	return nil
}

func (p *parser) parseTensor(name string) (Tensor, error) {
	t := Tensor{Name: name}
	if err := p.delim('{', "tensor "+strconv.Quote(name)); err != nil {
		return t, err
	}
	var haveDType, haveShape, haveOffsets bool
	var offsets []int64
	for p.dec.More() {
		field, err := p.str("tensor field")
		if err != nil {
			return t, err
		}
		var dup bool
		switch field {
		case "dtype":
			dup, haveDType = haveDType, true
			if !dup {
				s, err := p.str("dtype")
				if err != nil {
					return t, fmt.Errorf("tensor %q: %w", name, err)
				}
				t.DType = DType(s)
			}
		case "shape":
			dup, haveShape = haveShape, true
			if !dup {
				t.Shape, err = p.intArray(name, "shape", p.lim.MaxDims, ErrBadShape)
				if err != nil {
					return t, err
				}
			}
		case "data_offsets":
			dup, haveOffsets = haveOffsets, true
			if !dup {
				offsets, err = p.intArray(name, "data_offsets", 2, ErrBadOffsets)
				if err != nil {
					return t, err
				}
			}
		default:
			return t, fmt.Errorf("%w: tensor %q: unknown field %q", ErrInvalidHeader, name, field)
		}
		if dup {
			return t, fmt.Errorf("%w: tensor %q field %q", ErrDuplicateKey, name, field)
		}
	}
	if err := p.delim('}', "tensor "+strconv.Quote(name)); err != nil {
		return t, err
	}
	if !haveDType || !haveShape || !haveOffsets {
		return t, fmt.Errorf("%w: tensor %q: missing dtype, shape or data_offsets", ErrInvalidHeader, name)
	}
	if len(offsets) != 2 {
		return t, fmt.Errorf("%w: tensor %q: need 2 offsets, got %d", ErrBadOffsets, name, len(offsets))
	}
	t.Begin, t.End = offsets[0], offsets[1]
	return t, p.checkTensor(t)
}

// intArray parses a JSON array of at most max non-negative int64 values.
func (p *parser) intArray(name, field string, maxLen int, sentinel error) ([]int64, error) {
	if err := p.delim('[', "tensor "+strconv.Quote(name)+" "+field); err != nil {
		return nil, err
	}
	out := []int64{}
	for p.dec.More() {
		if len(out) >= maxLen {
			return nil, fmt.Errorf("%w: tensor %q: %s has more than %d entries", sentinel, name, field, maxLen)
		}
		t, err := p.token()
		if err != nil {
			return nil, err
		}
		num, ok := t.(json.Number)
		if !ok {
			return nil, fmt.Errorf("%w: tensor %q: %s entry is %s, want integer", sentinel, name, field, describe(t))
		}
		v, err := strconv.ParseInt(string(num), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("%w: tensor %q: %s entry %s is not an int64", sentinel, name, field, num)
		}
		if v < 0 {
			return nil, fmt.Errorf("%w: tensor %q: %s entry %d is negative", sentinel, name, field, v)
		}
		out = append(out, v)
	}
	if err := p.delim(']', "tensor "+strconv.Quote(name)+" "+field); err != nil {
		return nil, err
	}
	return out, nil
}

func (p *parser) checkTensor(t Tensor) error {
	if !slices.Contains(p.lim.AllowedDTypes, t.DType) {
		return fmt.Errorf("%w: tensor %q: %q (allowed %v)", ErrBadDType, t.Name, t.DType, p.lim.AllowedDTypes)
	}
	size, ok := dtypeSizes[t.DType]
	if !ok {
		return fmt.Errorf("%w: tensor %q: unknown element size for %q", ErrBadDType, t.Name, t.DType)
	}
	nbytes, err := byteSize(t.Shape, size)
	if err != nil {
		return fmt.Errorf("tensor %q shape %v: %w", t.Name, t.Shape, err)
	}
	if t.End < t.Begin {
		return fmt.Errorf("%w: tensor %q: end %d < begin %d", ErrBadOffsets, t.Name, t.End, t.Begin)
	}
	if got := uint64(t.End - t.Begin); got != nbytes { // #nosec G115 -- End >= Begin >= 0
		return fmt.Errorf("%w: tensor %q: %s %v needs %d bytes, offsets span %d",
			ErrSizeMismatch, t.Name, t.DType, t.Shape, nbytes, got)
	}
	return nil
}

// byteSize returns prod(shape)*elemSize, failing on int64 overflow. A zero
// dimension makes the tensor empty regardless of the other dimensions.
func byteSize(shape []int64, elemSize uint64) (uint64, error) {
	if slices.Contains(shape, 0) {
		return 0, nil
	}
	n := elemSize
	for _, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("%w: negative dimension %d", ErrBadShape, d)
		}
		hi, lo := bits.Mul64(n, uint64(d))
		if hi != 0 || lo > math.MaxInt64 {
			return 0, ErrOverflow
		}
		n = lo
	}
	return n, nil
}

// checkLayout verifies every tensor lies inside the data buffer and that the
// tensors tile it exactly, with no overlap, gaps or trailing bytes.
func checkLayout(h *Header) error {
	dataSize := h.FileSize - h.DataStart
	ts := make([]Tensor, 0, len(h.Tensors))
	for _, t := range h.Tensors {
		if t.End > dataSize {
			return fmt.Errorf("%w: tensor %q: end %d > data buffer size %d", ErrBadOffsets, t.Name, t.End, dataSize)
		}
		ts = append(ts, t)
	}
	slices.SortFunc(ts, func(a, b Tensor) int {
		if a.Begin != b.Begin {
			return cmp.Compare(a.Begin, b.Begin)
		}
		if a.End != b.End {
			return cmp.Compare(a.End, b.End)
		}
		return cmp.Compare(a.Name, b.Name)
	})
	var cursor int64
	prev := ""
	for _, t := range ts {
		switch {
		case t.Begin < cursor:
			return fmt.Errorf("%w: %q [%d,%d) overlaps %q ending at %d", ErrOverlap, t.Name, t.Begin, t.End, prev, cursor)
		case t.Begin > cursor:
			return fmt.Errorf("%w: gap [%d,%d) before %q", ErrNotContiguous, cursor, t.Begin, t.Name)
		}
		if t.End > cursor {
			cursor, prev = t.End, t.Name
		}
	}
	if cursor != dataSize {
		return fmt.Errorf("%w: tensors cover %d bytes of a %d byte data buffer", ErrNotContiguous, cursor, dataSize)
	}
	return nil
}
