//go:build ignore

package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
)

func main() {
	workers := flag.Int("workers", 4, "worker count")
	load := flag.Int("load", 3000, "load ops per worker")
	mixed := flag.Int("mixed", 9000, "mixed ops per worker")
	query := flag.Int("query", 4000, "query ops per worker")
	seed := flag.Int("seed", 42, "deterministic seed")
	output := flag.String("output", "", "output path")
	flag.Parse()

	if *output == "" {
		fmt.Fprintln(os.Stderr, "-output is required")
		os.Exit(1)
	}

	f, err := os.Create(*output)
	if err != nil {
		fmt.Fprintln(os.Stderr, "create:", err)
		os.Exit(1)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 8<<20)
	defer w.Flush()

	fmt.Fprintf(w, "workers %d\n", *workers)

	fmt.Fprintln(w, "phase load")
	for worker := 0; worker < *workers; worker++ {
		for i := 0; i < *load; i++ {
			key := keyFor(worker, i)
			tag := tagFor(worker, i)
			value := valueFor(*seed, worker, i, 0)
			fmt.Fprintf(w, "%d|put|%s|%s|%s|\n", worker, key, tag, value)
		}
	}

	fmt.Fprintln(w, "phase mixed")
	for worker := 0; worker < *workers; worker++ {
		for i := 0; i < *mixed; i++ {
			mode := i % 20
			switch {
			case mode < 7:
				idx := (i*17 + worker*13) % (*load + i/40 + 1)
				fmt.Fprintf(w, "%d|get|%s|||\n", worker, keyFor(worker, idx))
			case mode < 11:
				idx := (i*11 + worker*5) % (*load + i/32 + 1)
				tag := tagFor(worker, idx+i/19)
				value := valueFor(*seed, worker, i, 1)
				fmt.Fprintf(w, "%d|put|%s|%s|%s|\n", worker, keyFor(worker, idx), tag, value)
			case mode < 14:
				idx := *load + i/3
				tag := tagFor(worker, idx+i)
				value := valueFor(*seed, worker, i, 2)
				fmt.Fprintf(w, "%d|put|%s|%s|%s|\n", worker, keyFor(worker, idx), tag, value)
			case mode < 17:
				idx := (i*7 + worker*19) % (*load + i/5 + 1)
				fmt.Fprintf(w, "%d|del|%s|||\n", worker, keyFor(worker, idx))
			default:
				tag := sharedTag((i + worker) % 24)
				prefix := fmt.Sprintf("tenant/%02d/", worker)
				fmt.Fprintf(w, "%d|count||%s||%s\n", worker, tag, prefix)
			}
		}
	}

	fmt.Fprintln(w, "phase query")
	for worker := 0; worker < *workers; worker++ {
		for i := 0; i < *query; i++ {
			if i%3 == 0 {
				tag := sharedTag((worker*7 + i) % 24)
				prefix := fmt.Sprintf("tenant/%02d/", (worker+i)%*workers)
				fmt.Fprintf(w, "%d|count||%s||%s\n", worker, tag, prefix)
				continue
			}
			idx := (i*23 + worker*29) % (*load + *mixed/6 + 1)
			fmt.Fprintf(w, "%d|get|%s|||\n", worker, keyFor(worker, idx))
		}
	}
}

func keyFor(worker, idx int) string {
	return fmt.Sprintf("tenant/%02d/key/%06d", worker, idx)
}

func tagFor(worker, idx int) string {
	return sharedTag((worker*5 + idx*3) % 24)
}

func sharedTag(idx int) string {
	return fmt.Sprintf("tag/%02d", idx%24)
}

func valueFor(seed, worker, idx, salt int) string {
	base := seed*131 + worker*977 + idx*37 + salt*541
	return fmt.Sprintf(
		"val-%08x-%08x-%08x-%08x",
		base*17+11,
		base*19+23,
		base*29+31,
		base*43+47,
	)
}
