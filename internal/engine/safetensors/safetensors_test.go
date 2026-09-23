package safetensors

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// writeRaw writes an 8-byte length prefix n, then header, then dataLen zero
// bytes, and returns the path.
func writeRaw(tb testing.TB, n uint64, header []byte, dataLen int) string {
	tb.Helper()
	var buf bytes.Buffer
	var p [8]byte
	binary.LittleEndian.PutUint64(p[:], n)
	buf.Write(p[:])
	buf.Write(header)
	buf.Write(make([]byte, dataLen))
	path := filepath.Join(tb.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		tb.Fatal(err)
	}
	return path
}

// writeFile writes a well-formed prefix for header.
func writeFile(tb testing.TB, header string, dataLen int) string {
	tb.Helper()
	return writeRaw(tb, uint64(len(header)), []byte(header), dataLen)
}

func TestReadHeaderMalformed(t *testing.T) {
	longName := strings.Repeat("a", MaxNameBytes+1)
	manyTensors := func(n int) string {
		var sb strings.Builder
		sb.WriteString("{")
		for i := range n {
			if i > 0 {
				sb.WriteString(",")
			}
			fmt.Fprintf(&sb, `"t%d":{"dtype":"F32","shape":[1],"data_offsets":[%d,%d]}`, i, 4*i, 4*i+4)
		}
		sb.WriteString("}")
		return sb.String()
	}

	cases := []struct {
		name    string
		path    func(t *testing.T) string
		lim     Limits
		wantErr error
	}{
		{"file shorter than 8 bytes", func(t *testing.T) string {
			p := filepath.Join(t.TempDir(), "x")
			if err := os.WriteFile(p, []byte{1, 2, 3}, 0o600); err != nil {
				t.Fatal(err)
			}
			return p
		}, Limits{}, ErrTruncated},
		{"header length 2^63", func(t *testing.T) string {
			return writeRaw(t, 1<<63, []byte("{}"), 0)
		}, Limits{}, ErrHeaderTooLarge},
		{"header length 2^64-1", func(t *testing.T) string {
			return writeRaw(t, math.MaxUint64, []byte("{}"), 0)
		}, Limits{}, ErrHeaderTooLarge},
		{"header length over custom limit", func(t *testing.T) string {
			return writeFile(t, `{"__metadata__":{"a":"b"}}`, 0)
		}, Limits{MaxHeaderBytes: 8}, ErrHeaderTooLarge},
		{"header length > file", func(t *testing.T) string {
			return writeRaw(t, 1000, []byte("{}"), 0)
		}, Limits{}, ErrTruncated},
		{"header length too short", func(t *testing.T) string {
			return writeRaw(t, 1, []byte("{"), 0)
		}, Limits{}, ErrInvalidHeader},
		{"truncated JSON", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[1]`, 0)
		}, Limits{}, ErrInvalidHeader},
		{"non-object array", func(t *testing.T) string { return writeFile(t, `[]`, 0) }, Limits{}, ErrInvalidHeader},
		{"non-object string", func(t *testing.T) string { return writeFile(t, `"x"`, 0) }, Limits{}, ErrInvalidHeader},
		{"invalid UTF-8", func(t *testing.T) string {
			return writeRaw(t, 6, []byte("{\"\xff\":1}"[:6]), 0)
		}, Limits{}, ErrInvalidHeader},
		{"trailing value", func(t *testing.T) string { return writeFile(t, `{} {}`, 0) }, Limits{}, ErrInvalidHeader},
		{"trailing garbage", func(t *testing.T) string { return writeFile(t, `{}x`, 0) }, Limits{}, ErrInvalidHeader},
		{"tensor not an object", func(t *testing.T) string { return writeFile(t, `{"a":1}`, 0) }, Limits{}, ErrInvalidHeader},
		{"duplicate tensor key", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`, 8)
		}, Limits{}, ErrDuplicateKey},
		{"duplicate metadata key", func(t *testing.T) string {
			return writeFile(t, `{"__metadata__":{"k":"1","k":"2"}}`, 0)
		}, Limits{}, ErrDuplicateKey},
		{"duplicate __metadata__", func(t *testing.T) string {
			return writeFile(t, `{"__metadata__":{},"__metadata__":{}}`, 0)
		}, Limits{}, ErrDuplicateKey},
		{"duplicate tensor field", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","dtype":"F16","shape":[1],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrDuplicateKey},
		{"unknown tensor field", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"x":1}}`, 4)
		}, Limits{}, ErrInvalidHeader},
		{"missing data_offsets", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[1]}}`, 4)
		}, Limits{}, ErrInvalidHeader},
		{"bad dtype unknown", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"Q4","shape":[1],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrBadDType},
		{"bad dtype not allowlisted", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrBadDType},
		{"dtype not a string", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":4,"shape":[1],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrInvalidHeader},
		{"negative dim", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrBadShape},
		{"fractional dim", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrBadShape},
		{"dim exceeds int64", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[9223372036854775808],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrBadShape},
		{"dims product overflows int64", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrOverflow},
		{"dims times dtype size overflows int64", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[4611686018427387904],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrOverflow},
		{"deep dims", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[1,1,1,1,1,1,1,1,1],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrBadShape},
		{"size mismatch", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrSizeMismatch},
		{"end < begin", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[0],"data_offsets":[8,4]}}`, 8)
		}, Limits{}, ErrBadOffsets},
		{"negative offset", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[1],"data_offsets":[-4,0]}}`, 4)
		}, Limits{}, ErrBadOffsets},
		{"three offsets", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4,8]}}`, 8)
		}, Limits{}, ErrBadOffsets},
		{"one offset", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[1],"data_offsets":[4]}}`, 8)
		}, Limits{}, ErrBadOffsets},
		{"end > buffer", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, 4)
		}, Limits{}, ErrBadOffsets},
		{"end > buffer huge", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"U8","shape":[9223372036854775807],"data_offsets":[0,9223372036854775807]}}`, 4)
		}, Limits{AllowedDTypes: []DType{U8}}, ErrBadOffsets},
		{"overlapping tensors", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}`, 12)
		}, Limits{}, ErrOverlap},
		{"identical offsets", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 8)
		}, Limits{}, ErrOverlap},
		{"gap between tensors", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}`, 12)
		}, Limits{}, ErrNotContiguous},
		{"trailing data bytes", func(t *testing.T) string {
			return writeFile(t, `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 5)
		}, Limits{}, ErrNotContiguous},
		{"metadata non-string value", func(t *testing.T) string {
			return writeFile(t, `{"__metadata__":{"format":1}}`, 0)
		}, Limits{}, ErrBadMetadata},
		{"metadata nested object", func(t *testing.T) string {
			return writeFile(t, `{"__metadata__":{"format":{}}}`, 0)
		}, Limits{}, ErrBadMetadata},
		{"metadata not an object", func(t *testing.T) string {
			return writeFile(t, `{"__metadata__":"pt"}`, 0)
		}, Limits{}, ErrBadMetadata},
		{"over-long name", func(t *testing.T) string {
			return writeFile(t, `{"`+longName+`":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrBadName},
		{"non-printable name", func(t *testing.T) string {
			return writeFile(t, `{"a\u0001":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrBadName},
		{"non-ASCII name", func(t *testing.T) string {
			return writeFile(t, `{"é":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrBadName},
		{"empty name", func(t *testing.T) string {
			return writeFile(t, `{"":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 4)
		}, Limits{}, ErrBadName},
		{"too many tensors", func(t *testing.T) string {
			return writeFile(t, manyTensors(5), 20)
		}, Limits{MaxTensors: 4}, ErrTooManyTensors},
		{"too many tensors default", func(t *testing.T) string {
			return writeFile(t, manyTensors(DefaultMaxTensors+1), 4*(DefaultMaxTensors+1))
		}, Limits{}, ErrTooManyTensors},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h, err := ReadHeader(tc.path(t), tc.lim)
			if err == nil {
				t.Fatalf("ReadHeader succeeded (%d tensors), want %v", len(h.Tensors), tc.wantErr)
			}
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("got error %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestReadHeaderNotRegular(t *testing.T) {
	if _, err := ReadHeader(t.TempDir(), Limits{}); !errors.Is(err, ErrNotRegular) {
		t.Fatalf("got %v, want ErrNotRegular", err)
	}
	if _, err := ReadHeader(filepath.Join(t.TempDir(), "missing"), Limits{}); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("got %v, want ErrNotExist", err)
	}
}

func TestReadHeaderEdgeCasesAccepted(t *testing.T) {
	cases := map[string]struct {
		header  string
		dataLen int
		tensors int
	}{
		"empty object":         {`{}`, 0, 0},
		"space padding":        {`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}    `, 4, 1},
		"scalar tensor":        {`{"a":{"dtype":"F16","shape":[],"data_offsets":[0,2]}}`, 2, 1},
		"zero-size tensors":    {`{"z":{"dtype":"F32","shape":[0,4294967296],"data_offsets":[0,0]},"y":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 4, 3},
		"zero-size at the end": {`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"z":{"dtype":"BF16","shape":[0],"data_offsets":[4,4]}}`, 4, 2},
		"field order":          {`{"a":{"data_offsets":[0,4],"shape":[2],"dtype":"BF16"}}`, 4, 1},
		"metadata only":        {`{"__metadata__":{"format":"pt"}}`, 0, 0},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			h, err := ReadHeader(writeFile(t, tc.header, tc.dataLen), Limits{})
			if err != nil {
				t.Fatal(err)
			}
			if len(h.Tensors) != tc.tensors {
				t.Fatalf("got %d tensors, want %d", len(h.Tensors), tc.tensors)
			}
		})
	}
}

// buildFile serialises tensors (with data filled by a counter) the way a
// real writer would, returning the file path and the expected header.
func buildFile(tb testing.TB, md map[string]string, ts []Tensor) (string, []byte) {
	tb.Helper()
	type entry struct {
		DType       DType    `json:"dtype"`
		Shape       []int64  `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}
	obj := map[string]any{}
	if md != nil {
		obj[metadataKey] = md
	}
	var data []byte
	for i := range ts {
		n, err := byteSize(ts[i].Shape, uint64(ts[i].DType.Size()))
		if err != nil {
			tb.Fatal(err)
		}
		ts[i].Begin = int64(len(data))
		for j := range n {
			data = append(data, byte(j))
		}
		ts[i].End = int64(len(data))
		obj[ts[i].Name] = entry{ts[i].DType, ts[i].Shape, [2]int64{ts[i].Begin, ts[i].End}}
	}
	hdr, err := json.Marshal(obj)
	if err != nil {
		tb.Fatal(err)
	}
	for len(hdr)%8 != 0 {
		hdr = append(hdr, ' ')
	}
	var buf bytes.Buffer
	var p [8]byte
	binary.LittleEndian.PutUint64(p[:], uint64(len(hdr)))
	buf.Write(p[:])
	buf.Write(hdr)
	buf.Write(data)
	path := filepath.Join(tb.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		tb.Fatal(err)
	}
	return path, buf.Bytes()
}

func TestRoundTrip(t *testing.T) {
	md := map[string]string{"format": "pt", "note": "ünïcode ok in metadata"}
	want := []Tensor{
		{Name: "w", DType: F32, Shape: []int64{3, 4}},
		{Name: "b", DType: BF16, Shape: []int64{4}},
		{Name: "h", DType: F16, Shape: []int64{2, 1, 3}},
		{Name: "s", DType: F32, Shape: []int64{}},
	}
	path, raw := buildFile(t, md, want)

	h, err := ReadHeader(path, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if h.FileSize != int64(len(raw)) {
		t.Errorf("FileSize = %d, want %d", h.FileSize, len(raw))
	}
	if n := binary.LittleEndian.Uint64(raw); h.DataStart != 8+int64(n) {
		t.Errorf("DataStart = %d, want %d", h.DataStart, 8+n)
	}
	if len(h.Metadata) != len(md) || h.Metadata["format"] != "pt" || h.Metadata["note"] != md["note"] {
		t.Errorf("Metadata = %v, want %v", h.Metadata, md)
	}
	if len(h.Tensors) != len(want) {
		t.Fatalf("got %d tensors, want %d", len(h.Tensors), len(want))
	}
	for _, w := range want {
		g, ok := h.Tensors[w.Name]
		if !ok {
			t.Fatalf("missing %q", w.Name)
		}
		if g.Name != w.Name || g.DType != w.DType || !slices.Equal(g.Shape, w.Shape) || g.Begin != w.Begin || g.End != w.End {
			t.Errorf("tensor %q = %+v, want %+v", w.Name, g, w)
		}
		// The offsets address the bytes the writer put there.
		chunk := raw[h.DataStart+g.Begin : h.DataStart+g.End]
		for j, c := range chunk {
			if c != byte(j) {
				t.Fatalf("tensor %q byte %d = %d, want %d", w.Name, j, c, byte(j))
			}
		}
	}

	exp := []Expect{
		{Name: "w", Shape: []int64{3, 4}, DTypes: []DType{F32}},
		{Name: "b", Shape: []int64{4}, DTypes: []DType{BF16, F16}},
		{Name: "s", Shape: []int64{}},
	}
	if err := h.Require(exp); err != nil {
		t.Fatalf("Require: %v", err)
	}

	sum := sha256.Sum256(raw)
	got, err := SHA256File(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if got != hex.EncodeToString(sum[:]) {
		t.Fatalf("SHA256File = %s, want %x", got, sum)
	}
}

func TestRequireErrors(t *testing.T) {
	path, _ := buildFile(t, nil, []Tensor{
		{Name: "w", DType: F32, Shape: []int64{3, 4}},
		{Name: "b", DType: BF16, Shape: []int64{4}},
	})
	h, err := ReadHeader(path, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name    string
		exp     Expect
		wantErr error
		wantMsg []string
	}{
		{"missing", Expect{Name: "nope", Shape: []int64{1}}, ErrMissingTensor, []string{`"nope"`}},
		{"shape", Expect{Name: "w", Shape: []int64{4, 3}}, ErrShapeMismatch, []string{`"w"`, "got [3 4]", "want [4 3]"}},
		{"rank", Expect{Name: "b", Shape: []int64{4, 1}}, ErrShapeMismatch, []string{`"b"`, "got [4]", "want [4 1]"}},
		{"dtype", Expect{Name: "b", Shape: []int64{4}, DTypes: []DType{F32, F16}}, ErrDTypeNotAllowed, []string{`"b"`, "got BF16", "[F32 F16]"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := h.Require([]Expect{tc.exp})
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("got %v, want %v", err, tc.wantErr)
			}
			for _, m := range tc.wantMsg {
				if !strings.Contains(err.Error(), m) {
					t.Errorf("error %q does not contain %q", err, m)
				}
			}
		})
	}
	// All violations are reported together.
	err = h.Require([]Expect{cases[0].exp, cases[1].exp, cases[3].exp})
	for _, c := range []int{0, 1, 3} {
		if !errors.Is(err, cases[c].wantErr) {
			t.Errorf("joined error %v missing %v", err, cases[c].wantErr)
		}
	}
}

func TestDOMExpectations(t *testing.T) {
	bb := DOMExpectations(BackbonePrefix)
	base := DOMExpectations(BaseModelPrefix)
	// 2 + 24*5 + 18 linear*9 + 6 full*6 (+ 4 heads)
	if len(bb) != 324 || len(base) != 320 {
		t.Fatalf("len = %d/%d, want 324/320", len(bb), len(base))
	}
	names := map[string]Expect{}
	for _, e := range bb {
		if _, dup := names[e.Name]; dup {
			t.Fatalf("duplicate expectation %q", e.Name)
		}
		names[e.Name] = e
		if !slices.Equal(e.DTypes, []DType{F32, BF16, F16}) {
			t.Errorf("%q dtypes %v", e.Name, e.DTypes)
		}
		if strings.Contains(e.Name, "_head.") == strings.HasPrefix(e.Name, BackbonePrefix) {
			t.Errorf("%q: heads must be unprefixed, everything else prefixed", e.Name)
		}
	}
	for i := range 24 {
		full := i == 3 || i == 7 || i == 11 || i == 15 || i == 19 || i == 23
		_, hasSelf := names[fmt.Sprintf("backbone.layers.%d.self_attn.q_proj.weight", i)]
		_, hasLin := names[fmt.Sprintf("backbone.layers.%d.linear_attn.A_log", i)]
		if hasSelf != full || hasLin == full {
			t.Errorf("layer %d: self_attn=%v linear_attn=%v, want full=%v", i, hasSelf, hasLin, full)
		}
	}
	for name, shape := range map[string][]int64{
		"backbone.embed_tokens.weight":                      {248320, 1024},
		"backbone.layers.0.linear_attn.conv1d.weight":       {6144, 1, 4},
		"backbone.layers.23.self_attn.o_proj.weight":        {1024, 2048},
		"backbone.layers.5.mlp.down_proj.weight":            {1024, 3584},
		"auxiliary_head.weight":                             {18, 1024},
		"binary_head.bias":                                  {1},
		"backbone.layers.22.linear_attn.in_proj_qkv.weight": {6144, 1024},
	} {
		if e, ok := names[name]; !ok || !slices.Equal(e.Shape, shape) {
			t.Errorf("%q = %v (present %v), want %v", name, e.Shape, ok, shape)
		}
	}
	for _, e := range base {
		if !strings.HasPrefix(e.Name, BaseModelPrefix) {
			t.Errorf("base expectation %q lacks prefix", e.Name)
		}
	}
}

// A full DOM-shaped header with real shapes and a sparse (never written)
// data region, so the test needs no model download.
func TestDOMSyntheticSparse(t *testing.T) {
	exp := DOMExpectations(BackbonePrefix)
	type entry struct {
		DType       DType    `json:"dtype"`
		Shape       []int64  `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}
	obj := map[string]entry{}
	var off int64
	for i, e := range exp {
		dt := BF16
		if i%7 == 0 {
			dt = F32
		}
		n, err := byteSize(e.Shape, uint64(dt.Size()))
		if err != nil {
			t.Fatal(err)
		}
		obj[e.Name] = entry{dt, e.Shape, [2]int64{off, off + int64(n)}}
		off += int64(n)
	}
	obj["mtp.fc.weight"] = entry{F16, []int64{2, 2}, [2]int64{off, off + 8}}
	off += 8
	hdr, err := json.Marshal(obj)
	if err != nil {
		t.Fatal(err)
	}
	path := writeFile(t, string(hdr), 0)
	if err := os.Truncate(path, 8+int64(len(hdr))+off); err != nil { // sparse
		t.Fatal(err)
	}
	h, err := ReadHeader(path, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if err := h.Require(exp); err != nil {
		t.Fatal(err)
	}
	if err := h.Require(DOMExpectations(BaseModelPrefix)); !errors.Is(err, ErrMissingTensor) {
		t.Fatalf("base-prefix Require = %v, want ErrMissingTensor", err)
	}
	// A structurally valid but wrong shape (same byte size) passes ReadHeader
	// and is caught by Require.
	e := obj["backbone.norm.weight"]
	e.Shape = []int64{512, 2}
	obj["backbone.norm.weight"] = e
	hdr, _ = json.Marshal(obj)
	path = writeFile(t, string(hdr), 0)
	if err := os.Truncate(path, 8+int64(len(hdr))+off); err != nil {
		t.Fatal(err)
	}
	if h, err = ReadHeader(path, Limits{}); err != nil {
		t.Fatal(err)
	}
	if err := h.Require(exp); !errors.Is(err, ErrShapeMismatch) || !strings.Contains(err.Error(), "backbone.norm.weight") {
		t.Fatalf("Require = %v, want shape mismatch naming backbone.norm.weight", err)
	}
}

const (
	realModelsDir = "/Users/shearer/vaktex/classification_models"
	mockF32Path   = realModelsDir + "/Harness/testdata/models/mock-dom-0.8b/model.safetensors"
	mockBF16Path  = realModelsDir + "/Harness/testdata/models/mock-dom-0.8b-bf16/model.safetensors"
	basePath      = realModelsDir + "/Qwen3.5-0.8B-Base/model.safetensors-00001-of-00001.safetensors"
	mockF32SHA256 = "50ea15ec0fb1e1fc44e06ecc02f880d09b4ad98c302e9a0b8d155f921773b719"
)

func requireFile(tb testing.TB, path string) {
	tb.Helper()
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("real model not present: %v", err)
	}
}

func TestRealFiles(t *testing.T) {
	cases := []struct {
		name, path, prefix string
		check              func(t *testing.T, h *Header)
	}{
		{"mock-dom-0.8b fp32", mockF32Path, BackbonePrefix, func(t *testing.T, h *Header) {
			for n, tn := range h.Tensors {
				if tn.DType != F32 {
					t.Errorf("%q dtype %s, want F32", n, tn.DType)
				}
			}
		}},
		{"mock-dom-0.8b bf16", mockBF16Path, BackbonePrefix, func(t *testing.T, h *Header) {
			for n, tn := range h.Tensors {
				want := BF16
				if strings.Contains(n, "_head.") {
					want = F32
				}
				if tn.DType != want {
					t.Errorf("%q dtype %s, want %s", n, tn.DType, want)
				}
			}
		}},
		{"Qwen3.5-0.8B-Base", basePath, BaseModelPrefix, func(t *testing.T, h *Header) {
			if len(h.Tensors) <= 320 {
				t.Errorf("expected extra (visual/mtp) tensors, got %d", len(h.Tensors))
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			requireFile(t, tc.path)
			h, err := ReadHeader(tc.path, Limits{})
			if err != nil {
				t.Fatal(err)
			}
			if err := h.Require(DOMExpectations(tc.prefix)); err != nil {
				t.Fatal(err)
			}
			tc.check(t, h)
		})
	}
	t.Run("mock-dom-0.8b fp32 sha256", func(t *testing.T) {
		requireFile(t, mockF32Path)
		if testing.Short() {
			t.Skip("hashes 3 GB")
		}
		got, err := SHA256File(context.Background(), mockF32Path)
		if err != nil {
			t.Fatal(err)
		}
		if got != mockF32SHA256 {
			t.Fatalf("sha256 = %s, want %s", got, mockF32SHA256)
		}
	})
}

// cancelReader returns zeros forever and cancels ctx during its second Read,
// so hashing can only terminate via cancellation.
type cancelReader struct {
	cancel context.CancelFunc
	reads  int
}

func (r *cancelReader) Read(p []byte) (int, error) {
	r.reads++
	if r.reads == 2 {
		r.cancel()
	}
	clear(p)
	return len(p), nil
}

func TestSHA256FileCancellation(t *testing.T) {
	path := writeFile(t, `{}`, 1<<20)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := SHA256File(ctx, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled: got %v, want context.Canceled", err)
	}

	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	r := &cancelReader{cancel: cancel}
	if _, err := sha256Reader(ctx, r); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-stream: got %v, want context.Canceled", err)
	}
	if r.reads != 2 {
		t.Fatalf("reads after cancel: %d, want stop right after cancellation (2)", r.reads)
	}

	if _, err := SHA256File(context.Background(), filepath.Join(t.TempDir(), "missing")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing file: got %v", err)
	}
}

func TestSHA256FileSmall(t *testing.T) {
	for _, n := range []int{0, 1, hashChunk - 1, hashChunk, hashChunk + 1, 3*hashChunk + 17} {
		data := bytes.Repeat([]byte{0xab}, n)
		p := filepath.Join(t.TempDir(), "f")
		if err := os.WriteFile(p, data, 0o600); err != nil {
			t.Fatal(err)
		}
		got, err := SHA256File(context.Background(), p)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		if got != hex.EncodeToString(sum[:]) {
			t.Fatalf("n=%d: %s != %x", n, got, sum)
		}
	}
}

// ReadHeader must read only the prefix and header, never the data buffer.
func TestReadHeaderReadsOnlyHeader(t *testing.T) {
	hdr := `{"a":{"dtype":"F32","shape":[262144],"data_offsets":[0,1048576]}}`
	var buf bytes.Buffer
	var p [8]byte
	binary.LittleEndian.PutUint64(p[:], uint64(len(hdr)))
	buf.Write(p[:])
	buf.WriteString(hdr)
	cr := &countingReader{r: io.MultiReader(&buf, zeroReader{})}
	if _, err := readHeader(cr, int64(8+len(hdr)+1<<20), Limits{}); err != nil {
		t.Fatal(err)
	}
	if cr.n != int64(8+len(hdr)) {
		t.Fatalf("read %d bytes, want %d", cr.n, 8+len(hdr))
	}
}

type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

type zeroReader struct{}

func (zeroReader) Read(p []byte) (int, error) { clear(p); return len(p), nil }

func FuzzReadHeader(f *testing.F) {
	pre := func(s string) []byte {
		var p [8]byte
		binary.LittleEndian.PutUint64(p[:], uint64(len(s)))
		return append(p[:], s...)
	}
	f.Add([]byte{})
	f.Add(pre(`{}`))
	f.Add(append(pre(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`), 0, 0, 0, 0))
	f.Add(append(pre(`{"__metadata__":{"k":"v"},"a":{"dtype":"BF16","shape":[2,1],"data_offsets":[0,4]}}`), 0, 0, 0, 0))
	f.Add(append(pre(`{"a":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,4]}}`), 0, 0, 0, 0))
	f.Add(append(pre(`{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"F16","shape":[1],"data_offsets":[1,3]}}`), 0, 0, 0))
	f.Add(pre(`{"a":{"dtype":"F32","shape":[-1],"data_offsets":[4,0]},"a":1}`))
	f.Add(pre(`[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]`))
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0x80, '{', '}'})

	dir := f.TempDir()
	lim := Limits{MaxHeaderBytes: 1 << 20, MaxTensors: 64}
	f.Fuzz(func(t *testing.T, data []byte) {
		tmp, err := os.CreateTemp(dir, "f-*.safetensors")
		if err != nil {
			t.Fatal(err)
		}
		path := tmp.Name()
		defer os.Remove(path) //nolint:errcheck
		if _, err := tmp.Write(data); err != nil {
			t.Fatal(err)
		}
		if err := tmp.Close(); err != nil {
			t.Fatal(err)
		}
		h, err := ReadHeader(path, lim)
		// The in-memory path must agree with the file path.
		if _, merr := readHeader(bytes.NewReader(data), int64(len(data)), lim); (merr == nil) != (err == nil) {
			t.Fatalf("file err %v, memory err %v", err, merr)
		}
		if err != nil {
			if h != nil {
				t.Fatal("non-nil header with error")
			}
			return
		}
		// Invariants of an accepted header.
		if h.FileSize != int64(len(data)) || h.DataStart < 10 || h.DataStart > h.FileSize {
			t.Fatalf("bad sizes: %+v", h)
		}
		var total int64
		for n, tn := range h.Tensors {
			if n != tn.Name || tn.Begin < 0 || tn.End < tn.Begin || h.DataStart+tn.End > h.FileSize {
				t.Fatalf("bad tensor %+v", tn)
			}
			if !slices.Contains(DefaultAllowedDTypes(), tn.DType) || len(tn.Shape) > DefaultMaxDims {
				t.Fatalf("bad tensor %+v", tn)
			}
			total += tn.End - tn.Begin
		}
		if total != h.FileSize-h.DataStart {
			t.Fatalf("tensors cover %d of %d data bytes", total, h.FileSize-h.DataStart)
		}
		_ = h.Require(DOMExpectations(BackbonePrefix))
	})
}

func BenchmarkReadHeaderReal(b *testing.B) {
	for _, bc := range []struct{ name, path string }{
		{"mock-f32-3GB", mockF32Path},
		{"mock-bf16-1.5GB", mockBF16Path},
		{"qwen-base-1.7GB", basePath},
	} {
		b.Run(bc.name, func(b *testing.B) {
			requireFile(b, bc.path)
			exp := DOMExpectations(BackbonePrefix)
			if bc.path == basePath {
				exp = DOMExpectations(BaseModelPrefix)
			}
			b.ReportAllocs()
			for b.Loop() {
				h, err := ReadHeader(bc.path, Limits{})
				if err != nil {
					b.Fatal(err)
				}
				if err := h.Require(exp); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkSHA256FileReal(b *testing.B) {
	requireFile(b, mockF32Path)
	st, err := os.Stat(mockF32Path)
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(st.Size())
	for b.Loop() {
		if _, err := SHA256File(context.Background(), mockF32Path); err != nil {
			b.Fatal(err)
		}
	}
}
