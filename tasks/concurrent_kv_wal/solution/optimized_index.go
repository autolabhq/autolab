package main

import (
	"hash/fnv"
	"strings"
	"sync"
)

type tagShard struct {
	mu   sync.RWMutex
	tags map[string]map[string]struct{}
}

type TagIndex struct {
	shards [32]tagShard
}

func NewTagIndex() *TagIndex {
	t := &TagIndex{}
	for i := range t.shards {
		t.shards[i].tags = make(map[string]map[string]struct{}, 16)
	}
	return t
}

func (t *TagIndex) Add(tag, key string) {
	shard := t.shardFor(tag)
	shard.mu.Lock()
	set := shard.tags[tag]
	if set == nil {
		set = make(map[string]struct{}, 64)
		shard.tags[tag] = set
	}
	set[key] = struct{}{}
	shard.mu.Unlock()
}

func (t *TagIndex) Remove(tag, key string) {
	shard := t.shardFor(tag)
	shard.mu.Lock()
	if set := shard.tags[tag]; set != nil {
		delete(set, key)
		if len(set) == 0 {
			delete(shard.tags, tag)
		}
	}
	shard.mu.Unlock()
}

func (t *TagIndex) CountPrefix(tag, prefix string) int {
	shard := t.shardFor(tag)
	shard.mu.RLock()
	set := shard.tags[tag]
	count := 0
	for key := range set {
		if strings.HasPrefix(key, prefix) {
			count++
		}
	}
	shard.mu.RUnlock()
	return count
}

func (t *TagIndex) TotalKeys() int64 {
	var total int64
	for i := range t.shards {
		shard := &t.shards[i]
		shard.mu.RLock()
		for _, set := range shard.tags {
			total += int64(len(set))
		}
		shard.mu.RUnlock()
	}
	return total
}

func (t *TagIndex) shardFor(tag string) *tagShard {
	h := fnv.New32a()
	_, _ = h.Write([]byte(tag))
	return &t.shards[h.Sum32()&31]
}
