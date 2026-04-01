// index.go -- OPTIMIZE THIS FILE (and store.go / wal.go).
//
// This secondary index maps tag -> sorted []key.
//
// Baseline bottlenecks:
//   - A single mutex protects all tags.
//   - Add/Remove use linear scans over the entire per-tag slice.
//   - Add re-sorts the slice after every successful insert.
//   - CountPrefix scans every key in the tag bucket.

package main

import (
	"sort"
	"strings"
	"sync"
)

type TagIndex struct {
	mu   sync.RWMutex
	tags map[string][]string
}

func NewTagIndex() *TagIndex {
	return &TagIndex{tags: make(map[string][]string, 64)}
}

func (t *TagIndex) Add(tag, key string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	keys := t.tags[tag]
	for _, existing := range keys {
		if existing == key {
			return
		}
	}
	keys = append(keys, key)
	sort.Strings(keys)
	t.tags[tag] = keys
}

func (t *TagIndex) Remove(tag, key string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	keys := t.tags[tag]
	for i, existing := range keys {
		if existing == key {
			copy(keys[i:], keys[i+1:])
			keys = keys[:len(keys)-1]
			break
		}
	}
	if len(keys) == 0 {
		delete(t.tags, tag)
		return
	}
	t.tags[tag] = keys
}

func (t *TagIndex) CountPrefix(tag, prefix string) int {
	t.mu.RLock()
	defer t.mu.RUnlock()

	count := 0
	for _, key := range t.tags[tag] {
		if strings.HasPrefix(key, prefix) {
			count++
		}
	}
	return count
}

func (t *TagIndex) TotalKeys() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var total int64
	for _, keys := range t.tags {
		total += int64(len(keys))
	}
	return total
}
