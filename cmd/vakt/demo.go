package main

import (
	"context"
	"time"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/engine"
	"github.com/vaktex/vakt/internal/labels"
	"github.com/vaktex/vakt/internal/report"
)

// demoUnits are fabricated units for --demo. The code is only hashed by the
// fake engine, so it just needs to differ between units.
var demoUnits = []core.Unit{
	{File: "src/net/packet.c", Language: "C", Kind: core.KindFunction, Name: "read_packet", StartLine: 41, EndLine: 88,
		Code: "int read_packet(int fd, char *buf) { char tmp[256]; size_t n = recv(fd, tmp, 1024, 0); memcpy(buf, tmp, n); return n; }"},
	{File: "src/net/packet.c", Language: "C", Kind: core.KindFunction, Name: "checksum", StartLine: 90, EndLine: 104,
		Code: "uint16_t checksum(const uint8_t *p, size_t n) { uint32_t s = 0; while (n--) s += *p++; return (uint16_t)s; }"},
	{File: "app/views/search.py", Language: "Python", Kind: core.KindFunction, Name: "search", StartLine: 12, EndLine: 31,
		Code: "def search(request):\n    q = request.GET['q']\n    return db.execute(f\"SELECT * FROM items WHERE name LIKE '%{q}%'\")"},
	{File: "app/views/admin.py", Language: "Python", Kind: core.KindMethod, Name: "AdminView.delete_user", StartLine: 55, EndLine: 73,
		Code: "def delete_user(self, request, uid):\n    User.objects.get(id=uid).delete()\n    return redirect('/admin')"},
	{File: "web/upload.ts", Language: "TypeScript", Kind: core.KindFunction, Name: "saveUpload", StartLine: 8, EndLine: 40,
		Code: "export async function saveUpload(req) { const p = path.join(UPLOADS, req.body.name); await fs.writeFile(p, req.body.data); }"},
	{File: "web/render.ts", Language: "TypeScript", Kind: core.KindFunction, Name: "renderComment", StartLine: 3, EndLine: 11,
		Code: "export function renderComment(c) { el.innerHTML = `<p>${c.body}</p>`; }"},
	{File: "internal/auth/session.go", Language: "Go", Kind: core.KindFunction, Name: "NewSessionID", StartLine: 20, EndLine: 29,
		Code: "func NewSessionID() string { return strconv.Itoa(rand.Int()) }"},
	{File: "internal/auth/session.go", Language: "Go", Kind: core.KindMethod, Name: "Store.Get", StartLine: 31, EndLine: 52,
		Code: "func (s *Store) Get(id string) *Session { s.mu.RLock(); defer s.mu.RUnlock(); return s.m[id] }"},
	{File: "lib/parser.rs", Language: "Rust", Kind: core.KindFunction, Name: "parse_header", StartLine: 100, EndLine: 180,
		SplitPart: 1, SplitOf: 2, ParentStartLine: 100, ParentEndLine: 260,
		Code: "fn parse_header(b: &[u8]) -> Header { let len = u32::from_be_bytes(b[0..4].try_into().unwrap()); unsafe { ... } }"},
	{File: "lib/parser.rs", Language: "Rust", Kind: core.KindFunction, Name: "parse_header", StartLine: 181, EndLine: 260,
		SplitPart: 2, SplitOf: 2, ParentStartLine: 100, ParentEndLine: 260,
		Code: "    let body = &b[4..4 + len as usize]; Header { len, body: body.to_vec() } }"},
	{File: "scripts/deploy.sh", Language: "Shell", Kind: core.KindFile, StartLine: 1, EndLine: 37,
		Code: "#!/bin/sh\ncurl -k https://example.com/install.sh | sh\neval \"$1\""},
}

// demoFamily is the family each demoUnits entry illustrates.
var demoFamily = []string{
	"memory_safety", "memory_safety", "data_neutralization", "authorization",
	"file_and_path", "web_security", "cryptographic_issues",
	"synchronization_and_concurrency", "memory_safety", "memory_safety", "communication_security",
}

// demoReport scores demoUnits with engine.Fake. It is the only place the CLI
// uses the fake engine, and the report says backend "fake".
func demoReport(ctx context.Context, o ScanOptions) (*report.Report, error) {
	start := time.Now()
	var eng core.Engine = engine.Fake{}
	defer eng.Close()
	batch := make([][]int32, len(demoUnits))
	for i, u := range demoUnits {
		ids := make([]int32, 0, len(u.Code))
		for _, b := range []byte(u.Code) {
			ids = append(ids, int32(b))
		}
		batch[i] = ids
	}
	scores, err := eng.Score(ctx, batch)
	if err != nil {
		return nil, err
	}
	results := make([]report.Result, len(demoUnits))
	files := map[string]bool{}
	for i, u := range demoUnits {
		// The fake engine's family scores are hash noise; give each unit
		// the family its code actually shows, so the demo reads sensibly.
		sc := scores[i]
		for f := range sc.Families {
			sc.Families[f] = 0.005
		}
		if fi := labels.Index(demoFamily[i]); fi >= 0 {
			sc.Families[fi] = 0.55 + 0.4*float32(len(u.Code)%7)/7
		}
		results[i] = report.Result{Unit: u, Tokens: len(batch[i]) * 37, Scores: sc, Cached: i%5 == 4}
		files[u.File] = true
	}
	info := eng.Info()
	meta := report.Meta{
		Root:          "/demo/acme-shop",
		StartedAt:     start,
		Duration:      time.Since(start) + 1840*time.Millisecond, // pretend it took a moment
		Files:         len(files) + 3,
		CacheHits:     len(demoUnits) / 5,
		Skipped:       []report.Skip{{File: "web/vendor.min.js", Reason: "minified"}},
		ModelRepo:     brand.ModelRepo,
		ModelRevision: "demo",
		ModelSHA:      info.ModelSHA,
		Backend:       info.Backend, // "fake"
		Device:        "synthetic",
		Precision:     info.Precision,
	}
	return report.Build(meta, results, o.Threshold), nil
}
