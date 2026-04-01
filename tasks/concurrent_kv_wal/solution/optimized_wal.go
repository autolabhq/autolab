package main

import (
	"bytes"
	"hash/fnv"
	"strconv"
	"sync"
)

type walStats struct {
	records      int64
	logicalBytes int64
	checksum     uint64
}

type WAL struct {
	mu      sync.Mutex
	sink    bytes.Buffer
	stats   walStats
	pending int
}

func NewWAL() *WAL {
	return &WAL{}
}

func (w *WAL) AppendPut(seq uint64, key, tag string, value []byte) {
	w.mu.Lock()
	w.appendPutRecord(seq, key, tag, value)
	w.stats.records++
	w.stats.logicalBytes += logicalPutSize(key, tag, value)
	w.stats.checksum ^= walChecksumPut(key, tag, value)
	w.pending++
	if w.pending >= 256 || w.sink.Len() >= 1<<20 {
		w.flushLocked()
	}
	w.mu.Unlock()
}

func (w *WAL) AppendDelete(seq uint64, key string) {
	w.mu.Lock()
	w.appendDeleteRecord(seq, key)
	w.stats.records++
	w.stats.logicalBytes += logicalDeleteSize(key)
	w.stats.checksum ^= walChecksumDelete(key)
	w.pending++
	if w.pending >= 256 || w.sink.Len() >= 1<<20 {
		w.flushLocked()
	}
	w.mu.Unlock()
}

func (w *WAL) Stats() walStats {
	w.mu.Lock()
	w.flushLocked()
	stats := w.stats
	w.mu.Unlock()
	return stats
}

func (w *WAL) flushLocked() {
	w.pending = 0
}

func (w *WAL) appendPutRecord(seq uint64, key, tag string, value []byte) {
	var scratch [32]byte
	w.sink.WriteByte('P')
	w.sink.WriteByte('|')
	b := strconv.AppendUint(scratch[:0], seq, 10)
	w.sink.Write(b)
	w.sink.WriteByte('|')
	w.sink.WriteString(key)
	w.sink.WriteByte('|')
	w.sink.WriteString(tag)
	w.sink.WriteByte('|')
	w.sink.Write(value)
	w.sink.WriteByte('\n')
}

func (w *WAL) appendDeleteRecord(seq uint64, key string) {
	var scratch [32]byte
	w.sink.WriteByte('D')
	w.sink.WriteByte('|')
	b := strconv.AppendUint(scratch[:0], seq, 10)
	w.sink.Write(b)
	w.sink.WriteByte('|')
	w.sink.WriteString(key)
	w.sink.WriteByte('\n')
}

func logicalPutSize(key, tag string, value []byte) int64 {
	return int64(1 + len(key) + len(tag) + len(value))
}

func logicalDeleteSize(key string) int64 {
	return int64(1 + len(key))
}

func walChecksumPut(key, tag string, value []byte) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte{1})
	_, _ = h.Write([]byte(key))
	_, _ = h.Write([]byte(tag))
	_, _ = h.Write(value)
	return h.Sum64()
}

func walChecksumDelete(key string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte{2})
	_, _ = h.Write([]byte(key))
	return h.Sum64()
}
