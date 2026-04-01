// main.go -- benchmark driver and workload parser for concurrent_kv_wal.
//
// DO NOT MODIFY THIS FILE.

package main

import (
	"bufio"
	"flag"
	"fmt"
	"hash/fnv"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

func main() {
	mode := flag.String("mode", "benchmark", "benchmark or verify")
	input := flag.String("input", "", "input workload file")
	runs := flag.Int("runs", 5, "number of timed runs")
	flag.Parse()

	if *input == "" {
		fmt.Fprintln(os.Stderr, "usage: solve -input FILE [-mode benchmark|verify] [-runs N]")
		os.Exit(1)
	}

	workload, err := loadWorkload(*input)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load workload:", err)
		os.Exit(1)
	}

	if *mode == "verify" {
		res := execute(workload)
		fmt.Print(formatResult(workload, res))
		return
	}

	_ = execute(workload) // warm-up

	times := make([]float64, *runs)
	var last Result
	for i := range times {
		t0 := time.Now()
		last = execute(workload)
		times[i] = time.Since(t0).Seconds()
	}
	sort.Float64s(times)
	median := times[*runs/2]
	fmt.Printf("ops=%d runs=%d time=%.6f checksum=%d final_keys=%d\n",
		workload.TotalOps, *runs, median, last.Checksum, last.FinalKeys)
}

func loadWorkload(path string) (Workload, error) {
	f, err := os.Open(path)
	if err != nil {
		return Workload{}, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 256*1024), 8<<20)

	var wl Workload
	var current *Phase
	lineNo := 0

	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "workers ") {
			n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "workers ")))
			if err != nil || n <= 0 {
				return Workload{}, fmt.Errorf("line %d: invalid workers declaration", lineNo)
			}
			wl.WorkerCount = n
			continue
		}
		if strings.HasPrefix(line, "phase ") {
			if wl.WorkerCount == 0 {
				return Workload{}, fmt.Errorf("line %d: phase before workers", lineNo)
			}
			wl.Phases = append(wl.Phases, Phase{
				Name:    strings.TrimSpace(strings.TrimPrefix(line, "phase ")),
				Workers: make([][]Operation, wl.WorkerCount),
			})
			current = &wl.Phases[len(wl.Phases)-1]
			continue
		}
		if current == nil {
			return Workload{}, fmt.Errorf("line %d: operation before phase", lineNo)
		}
		fields := strings.Split(line, "|")
		if len(fields) != 6 {
			return Workload{}, fmt.Errorf("line %d: expected 6 fields, got %d", lineNo, len(fields))
		}
		worker, err := strconv.Atoi(fields[0])
		if err != nil || worker < 0 || worker >= wl.WorkerCount {
			return Workload{}, fmt.Errorf("line %d: invalid worker id", lineNo)
		}

		var op Operation
		switch fields[1] {
		case "put":
			op = Operation{Kind: opPut, Key: fields[2], Tag: fields[3], Value: []byte(fields[4])}
		case "get":
			op = Operation{Kind: opGet, Key: fields[2]}
		case "del":
			op = Operation{Kind: opDelete, Key: fields[2]}
		case "count":
			op = Operation{Kind: opCount, Tag: fields[3], Prefix: fields[5]}
		default:
			return Workload{}, fmt.Errorf("line %d: unknown op kind %q", lineNo, fields[1])
		}
		current.Workers[worker] = append(current.Workers[worker], op)
		wl.TotalOps++
	}

	if err := scanner.Err(); err != nil {
		return Workload{}, err
	}
	if wl.WorkerCount == 0 || len(wl.Phases) == 0 {
		return Workload{}, fmt.Errorf("incomplete workload")
	}
	return wl, nil
}

func execute(wl Workload) Result {
	store := NewStore()
	var out Result

	for _, phase := range wl.Phases {
		results := make([]workerResult, wl.WorkerCount)
		var wg sync.WaitGroup
		wg.Add(wl.WorkerCount)
		for workerID := 0; workerID < wl.WorkerCount; workerID++ {
			ops := phase.Workers[workerID]
			go func(dst *workerResult, ops []Operation) {
				defer wg.Done()
				var wr workerResult
				for _, op := range ops {
					switch op.Kind {
					case opPut:
						store.Put(op.Key, op.Tag, op.Value)
						wr.putOps++
						wr.checksum ^= payloadChecksum(op.Value) + uint64(len(op.Key))*17 + uint64(len(op.Tag))*31
					case opGet:
						value, ok := store.Get(op.Key)
						if ok {
							wr.getHits++
							wr.checksum ^= payloadChecksum(value) + uint64(len(op.Key))*131
						} else {
							wr.getMisses++
							wr.checksum ^= uint64(len(op.Key))*977 + 0x9e3779b97f4a7c15
						}
					case opDelete:
						if store.Delete(op.Key) {
							wr.deleteHits++
							wr.checksum ^= 0xd6e8feb86659fd93 ^ uint64(len(op.Key))*53
						} else {
							wr.deleteMisses++
							wr.checksum ^= 0xa4093822299f31d0 ^ uint64(len(op.Key))*71
						}
					case opCount:
						n := store.CountPrefix(op.Tag, op.Prefix)
						wr.scanOps++
						wr.scanTotal += int64(n)
						wr.checksum ^= uint64(n)*0x517cc1b727220a95 ^ stringChecksum(op.Tag) ^ (stringChecksum(op.Prefix) << 1)
					}
				}
				*dst = wr
			}(&results[workerID], ops)
		}
		wg.Wait()

		for _, wr := range results {
			out.PutOps += wr.putOps
			out.GetHits += wr.getHits
			out.GetMisses += wr.getMisses
			out.DeleteHits += wr.deleteHits
			out.DeleteMisses += wr.deleteMisses
			out.ScanOps += wr.scanOps
			out.ScanTotal += wr.scanTotal
			out.Checksum ^= wr.checksum
		}
	}

	stats := store.Stats()
	out.FinalKeys = stats.LiveKeys
	out.FinalBytes = stats.LiveBytes
	out.IndexKeys = stats.IndexKeys
	out.WALRecords = stats.WALRecords
	out.WALBytes = stats.WALBytes
	out.WALChecksum = stats.WALChecksum
	out.Checksum ^= stats.WALChecksum ^ uint64(stats.LiveKeys)*0x94d049bb133111eb ^ uint64(stats.LiveBytes)
	return out
}

func formatResult(wl Workload, res Result) string {
	var b strings.Builder
	fmt.Fprintf(&b, "workers=%d\n", wl.WorkerCount)
	fmt.Fprintf(&b, "phases=%d\n", len(wl.Phases))
	fmt.Fprintf(&b, "total_ops=%d\n", wl.TotalOps)
	fmt.Fprintf(&b, "put_ops=%d\n", res.PutOps)
	fmt.Fprintf(&b, "get_hits=%d\n", res.GetHits)
	fmt.Fprintf(&b, "get_misses=%d\n", res.GetMisses)
	fmt.Fprintf(&b, "delete_hits=%d\n", res.DeleteHits)
	fmt.Fprintf(&b, "delete_misses=%d\n", res.DeleteMisses)
	fmt.Fprintf(&b, "scan_ops=%d\n", res.ScanOps)
	fmt.Fprintf(&b, "scan_total=%d\n", res.ScanTotal)
	fmt.Fprintf(&b, "final_keys=%d\n", res.FinalKeys)
	fmt.Fprintf(&b, "final_bytes=%d\n", res.FinalBytes)
	fmt.Fprintf(&b, "index_keys=%d\n", res.IndexKeys)
	fmt.Fprintf(&b, "wal_records=%d\n", res.WALRecords)
	fmt.Fprintf(&b, "wal_bytes=%d\n", res.WALBytes)
	fmt.Fprintf(&b, "wal_checksum=%d\n", res.WALChecksum)
	fmt.Fprintf(&b, "checksum=%d\n", res.Checksum)
	return b.String()
}

func payloadChecksum(b []byte) uint64 {
	h := fnv.New64a()
	_, _ = h.Write(b)
	return h.Sum64()
}

func stringChecksum(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return h.Sum64()
}
