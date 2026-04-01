// wal.go -- OPTIMIZE THIS FILE (and store.go / index.go).
//
// The baseline WAL intentionally behaves like a bad commit path:
//   - Uses fmt.Sprintf for every record.
//   - Flushes the bufio.Writer on every append.
//   - Serializes all writers under one mutex.
//
// The verifier does NOT inspect the physical WAL bytes. Correctness is defined
// by the semantic counters and checksum exposed through Stats().

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"hash/fnv"
	"sync"
)

type walStats struct {
	records      int64
	logicalBytes int64
	checksum     uint64
}

type WAL struct {
	mu     sync.Mutex
	sink   bytes.Buffer
	writer *bufio.Writer
	stats  walStats
}

func NewWAL() *WAL {
	w := &WAL{}
	w.writer = bufio.NewWriterSize(&w.sink, 4096)
	return w
}

func (w *WAL) AppendPut(seq uint64, key, tag string, value []byte) {
	w.mu.Lock()
	defer w.mu.Unlock()

	record := fmt.Sprintf("P|%d|%s|%s|%s\n", seq, key, tag, string(value))
	_, _ = w.writer.WriteString(record)
	_ = w.writer.Flush()

	w.stats.records++
	w.stats.logicalBytes += logicalPutSize(key, tag, value)
	w.stats.checksum ^= walChecksumPut(key, tag, value)
}

func (w *WAL) AppendDelete(seq uint64, key string) {
	w.mu.Lock()
	defer w.mu.Unlock()

	record := fmt.Sprintf("D|%d|%s\n", seq, key)
	_, _ = w.writer.WriteString(record)
	_ = w.writer.Flush()

	w.stats.records++
	w.stats.logicalBytes += logicalDeleteSize(key)
	w.stats.checksum ^= walChecksumDelete(key)
}

func (w *WAL) Stats() walStats {
	w.mu.Lock()
	defer w.mu.Unlock()
	_ = w.writer.Flush()
	return w.stats
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
