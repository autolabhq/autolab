# Regex Engine — Reference

## Background
Regular expression matching is a core primitive in text editors, compilers, log analyzers, network intrusion detection, and database query engines. Real-world engines (RE2, Rust's `regex`, PCRE2) invest heavily in compilation strategies because the same engine may match millions of lines per second across many patterns.

## Baseline Approach
The unoptimized implementation parses each pattern into an AST and matches using recursive backtracking: it tries the pattern from every start position in the haystack, recursively descending the AST and allocating temporary `Vec<usize>` state sets during matching. This is O(2^n) in the worst case for ambiguous patterns and produces significant allocation churn across 100,000 haystacks.

## Possible Optimization Directions
1. **Thompson NFA simulation** — compile AST to an NFA bytecode tape; simulate lockstep over the full state set per character; degrades worst-case from exponential to O(text_len × nfa_states) (~10–20x speedup)
2. **Bytecode instruction tape** — replace AST-walking interpreter with a flat instruction array (`Byte`, `Split`, `Jump`, `Match`, etc.) for better cache locality and branch predictability during simulation
3. **Bitset active-state tracking** — represent the active NFA state set as a fixed-size bitset rather than a `Vec<usize>` to eliminate allocation and reduce per-step work
4. **Flat character classes** — store character class membership as a `Box<[bool; 256]>` lookup table instead of iterating over ranges at match time
5. **Anchored fast path** — detect `^`-anchored patterns at compile time and skip re-seeding the start state at every position (~2x for anchored patterns)
6. **Lazy DFA caching** — cache NFA state-set → DFA state mappings on demand; many real inputs hit a small number of NFA state-set transitions, giving near-DFA speed without full determinization
7. **Literal prefilters** — extract required literal prefixes and use `memchr` to skip positions that cannot start a match

## Reference Solution
Compiles each pattern to a flat bytecode NFA (Thompson construction) with `Op::Byte`, `Op::Dot`, `Op::Class(Box<[bool;256]>)`, `Op::Split`, `Op::Jump`, `Op::Start`, `Op::End`, and `Op::Match` instructions. At search time, runs a two-`Vec` active-state simulation: `add_state` recursively expands epsilon transitions (`Split`, `Jump`, `Start`, `End`) into a seen-deduped state list, then each input byte advances all active states in lockstep. Anchored patterns are detected at compile time and skip re-inserting the start state on each character step.

## Source
- Ken Thompson, *Regular Expression Search Algorithm* (1968)
- Russ Cox, *Regular Expression Matching Can Be Simple And Fast* — https://swtch.com/~rsc/regexp/regexp1.html
- RE2 hybrid NFA/DFA design — Russ Cox
- Andrew Gallant, Rust `regex` crate internals
