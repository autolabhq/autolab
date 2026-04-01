// store.go -- OPTIMIZE THIS FILE (and index.go / wal.go).
//
// The baseline store is intentionally simple and correct, but pessimizes
// concurrency in three ways:
//
//  1. A single global RWMutex protects the entire keyspace.
//  2. Writes serialize WAL append + primary map update + secondary index update
//     under the same lock.
//  3. Every Get copies the full value before returning it.

package main

import "sync"

type entry struct {
	tag   string
	value []byte
}

type Store struct {
	mu    sync.RWMutex
	items map[string]*entry
	index *TagIndex
	wal   *WAL
	seq   uint64
}

func NewStore() *Store {
	return &Store{
		items: make(map[string]*entry, 1<<16),
		index: NewTagIndex(),
		wal:   NewWAL(),
	}
}

func (s *Store) nextSeq() uint64 {
	s.seq++
	return s.seq
}

func (s *Store) Put(key, tag string, value []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()

	seq := s.nextSeq()
	s.wal.AppendPut(seq, key, tag, value)

	if old, ok := s.items[key]; ok && old.tag != tag {
		s.index.Remove(old.tag, key)
	}

	copied := make([]byte, len(value))
	copy(copied, value)
	s.items[key] = &entry{tag: tag, value: copied}
	s.index.Add(tag, key)
}

func (s *Store) Get(key string) ([]byte, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	e, ok := s.items[key]
	if !ok {
		return nil, false
	}
	out := make([]byte, len(e.value))
	copy(out, e.value)
	return out, true
}

func (s *Store) Delete(key string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	seq := s.nextSeq()
	s.wal.AppendDelete(seq, key)

	e, ok := s.items[key]
	if !ok {
		return false
	}
	delete(s.items, key)
	s.index.Remove(e.tag, key)
	return true
}

func (s *Store) CountPrefix(tag, prefix string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.index.CountPrefix(tag, prefix)
}

func (s *Store) Stats() StoreStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var liveKeys, liveBytes int64
	for _, e := range s.items {
		liveKeys++
		liveBytes += int64(len(e.value))
	}
	walStats := s.wal.Stats()
	return StoreStats{
		LiveKeys:    liveKeys,
		LiveBytes:   liveBytes,
		IndexKeys:   s.index.TotalKeys(),
		WALRecords:  walStats.records,
		WALBytes:    walStats.logicalBytes,
		WALChecksum: walStats.checksum,
	}
}
