package pipeline

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"time"

	bolt "go.etcd.io/bbolt"

	"github.com/vaktex/vakt/internal/core"
)

// Cache stores scores by core.CacheKey (model sha + precision + prompt
// hash). Only scores and token counts are stored, never source code.
type Cache struct {
	db *bolt.DB
}

var bucket = []byte("scores")

// record layout: tokens u32 | truncated u8 | severity f32 | 18 x f32
const recordSize = 4 + 1 + 4 + 4*core.NumFamilies

// OpenCache opens (or creates) the cache in dir. The directory is 0700 and
// the file 0600. A lock held by another vakt process makes OpenCache fail
// after one second, and the caller runs without a cache.
func OpenCache(dir string) (*Cache, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	db, err := bolt.Open(filepath.Join(dir, "scores.db"), 0o600, &bolt.Options{Timeout: time.Second, NoFreelistSync: true})
	if err != nil {
		return nil, err
	}
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(bucket)
		return err
	}); err != nil {
		db.Close()
		return nil, err
	}
	return &Cache{db: db}, nil
}

// Close closes the database.
func (c *Cache) Close() error {
	if c == nil {
		return nil
	}
	return c.db.Close()
}

type cached struct {
	Tokens    int
	Truncated bool
	Scores    core.Scores
}

// Get looks up keys; missing or malformed entries are absent from the map.
func (c *Cache) Get(keys []string) map[string]cached {
	out := make(map[string]cached, len(keys))
	if c == nil {
		return out
	}
	_ = c.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucket)
		for _, k := range keys {
			if v := b.Get([]byte(k)); v != nil {
				if r, ok := decode(v); ok {
					out[k] = r
				}
			}
		}
		return nil
	})
	return out
}

// Put stores a batch of results in one transaction.
func (c *Cache) Put(entries map[string]cached) error {
	if c == nil || len(entries) == 0 {
		return nil
	}
	return c.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucket)
		for k, r := range entries {
			if err := b.Put([]byte(k), encode(r)); err != nil {
				return err
			}
		}
		return nil
	})
}

func encode(r cached) []byte {
	buf := make([]byte, recordSize)
	binary.LittleEndian.PutUint32(buf[0:], uint32(min(max(r.Tokens, 0), math.MaxInt32))) // #nosec G115 -- clamped
	if r.Truncated {
		buf[4] = 1
	}
	binary.LittleEndian.PutUint32(buf[5:], math.Float32bits(r.Scores.Severity))
	for i, f := range r.Scores.Families {
		binary.LittleEndian.PutUint32(buf[9+4*i:], math.Float32bits(f))
	}
	return buf
}

func decode(v []byte) (cached, bool) {
	if len(v) != recordSize {
		return cached{}, false
	}
	var r cached
	r.Tokens = int(binary.LittleEndian.Uint32(v[0:]))
	r.Truncated = v[4] == 1
	r.Scores.Severity = math.Float32frombits(binary.LittleEndian.Uint32(v[5:]))
	if !valid(r.Scores.Severity) {
		return cached{}, false
	}
	for i := range r.Scores.Families {
		f := math.Float32frombits(binary.LittleEndian.Uint32(v[9+4*i:]))
		if !valid(f) {
			return cached{}, false
		}
		r.Scores.Families[i] = f
	}
	return r, true
}

func valid(f float32) bool { return f >= 0 && f <= 1 } // also rejects NaN
