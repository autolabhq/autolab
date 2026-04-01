package main

import (
	"hash/fnv"
	"sync"
	"sync/atomic"
)

type entry struct {
	tag   string
	value []byte
}

type storeShard struct {
	mu    sync.RWMutex
	items map[string]entry
}

type Store struct {
	shards [64]storeShard
	index  *TagIndex
	wal    *WAL
	seq    atomic.Uint64
}

func NewStore() *Store {
	s := &Store{
		index: NewTagIndex(),
		wal:   NewWAL(),
	}
	for i := range s.shards {
		s.shards[i].items = make(map[string]entry, 2048)
	}
	return s
}

func (s *Store) Put(key, tag string, value []byte) {
	seq := s.seq.Add(1)
	s.wal.AppendPut(seq, key, tag, value)

	shard := s.shardFor(key)
	copied := append([]byte(nil), value...)

	var oldTag string
	var hadOld bool
	shard.mu.Lock()
	if old, ok := shard.items[key]; ok {
		oldTag = old.tag
		hadOld = true
	}
	shard.items[key] = entry{tag: tag, value: copied}
	shard.mu.Unlock()

	if hadOld && oldTag != tag {
		s.index.Remove(oldTag, key)
	}
	s.index.Add(tag, key)
}

func (s *Store) Get(key string) ([]byte, bool) {
	shard := s.shardFor(key)
	shard.mu.RLock()
	e, ok := shard.items[key]
	shard.mu.RUnlock()
	if !ok {
		return nil, false
	}
	return e.value, true
}

func (s *Store) Delete(key string) bool {
	seq := s.seq.Add(1)
	s.wal.AppendDelete(seq, key)

	shard := s.shardFor(key)
	var oldTag string
	var ok bool
	shard.mu.Lock()
	if old, exists := shard.items[key]; exists {
		oldTag = old.tag
		ok = true
		delete(shard.items, key)
	}
	shard.mu.Unlock()
	if ok {
		s.index.Remove(oldTag, key)
	}
	return ok
}

func (s *Store) CountPrefix(tag, prefix string) int {
	return s.index.CountPrefix(tag, prefix)
}

func (s *Store) Stats() StoreStats {
	var liveKeys, liveBytes int64
	for i := range s.shards {
		shard := &s.shards[i]
		shard.mu.RLock()
		liveKeys += int64(len(shard.items))
		for _, e := range shard.items {
			liveBytes += int64(len(e.value))
		}
		shard.mu.RUnlock()
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

func (s *Store) shardFor(key string) *storeShard {
	h := fnv.New32a()
	_, _ = h.Write([]byte(key))
	return &s.shards[h.Sum32()&63]
}
