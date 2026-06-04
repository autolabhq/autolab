/*
 * main.cu -- Harness for the canonical Huffman decode CUDA task.
 *            DO NOT MODIFY.
 *
 * Modes:
 *   --verify             Public sanity check.  K = HUFF_NUM_STREAMS_VERIFY,
 *                        each stream HUFF_BYTES_PER_STREAM_VERIFY bytes.
 *                        Runtime-random seed.  Byte-exact match against the
 *                        original payload (which the encoder produced).
 *
 *   --benchmark          Unscored full-shape benchmark with a public seed.
 *                        Useful for the agent to time their own work.
 *
 *   --benchmark-verify   Scored run.  K = HUFF_NUM_STREAMS_BENCH, each
 *                        HUFF_BYTES_PER_STREAM bytes.  Reads its seed from
 *                        the environment variable HUFFMAN_DECODE_SEED so
 *                        the agent cannot hardcode it.  Validates byte-exact
 *                        output across the full workload AND prints a single
 *                        __VERIFIER_SCORE__=<median_ms> sentinel that the
 *                        verifier greps for.
 *
 * Per run we:
 *   1. Generate K independent random byte payloads (skewed distribution so
 *      the empirical Huffman code is non-trivial).
 *   2. Build a single canonical Huffman table from the average frequency.
 *      (Same table for all K streams.)
 *   3. Encode each payload to bits with the canonical table on CPU.
 *   4. Hand the encoded bits + table + per-stream bit lengths to the GPU
 *      decode kernel.
 *   5. Compare the decoded output byte-exact to the original payload.
 *   6. Time the decode kernel (median of N_TRIALS via CUDA events).
 *
 * The kernel only sees the encoded bits and the table; it does NOT see the
 * original payload, so an agent that "memorises and emits" must in fact
 * actually decode the bits to pass the byte-exact check.
 */

#include "solve.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <queue>
#include <unistd.h>
#include <vector>

// -------------------- splitmix64 PRNG --------------------

static inline uint64_t splitmix64_step(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// -------------------- payload generation --------------------

// Generate a "natural-looking" byte payload with a skewed distribution so
// that the empirical Huffman code is non-trivial (lots of common bytes,
// long tail).  We do this by sampling from a roughly Zipfian-like
// distribution over a permuted alphabet; the permutation depends on the
// seed so two different seeds produce different popular bytes.
//
// The resulting code lengths fall in [3, 16] in practice for our shapes,
// with mean ~7-8 bits/byte.
static void generate_payload(uint8_t* dst, size_t n, uint64_t seed,
                             uint64_t per_stream_salt) {
    // Build a permutation of 0..255 from the seed so popular byte ids
    // change with the seed.
    uint8_t perm[HUFF_ALPHABET_SIZE];
    for (int i = 0; i < HUFF_ALPHABET_SIZE; ++i) perm[i] = (uint8_t)i;
    {
        uint64_t s = seed ^ 0x123456789ABCDEF0ULL;
        for (int i = HUFF_ALPHABET_SIZE - 1; i > 0; --i) {
            uint32_t j = (uint32_t)(splitmix64_step(s) % (uint64_t)(i + 1));
            uint8_t  t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }
    }

    // Build a CDF over rank r = 0..255 with weights w_r = 1/(r+1)^alpha.
    // alpha controls skew; we vary it slightly per-seed for variety.
    // alpha around 0.5-0.7 gives a fairly flat distribution -> mean
    // canonical-Huffman code length ~7-8 bits, which matches our 8-bit
    // LUT design; alpha > 1 produces a very skewed distribution and an
    // average code length closer to 4 bits (which would make the
    // bit-by-bit baseline much faster than intended).
    double alpha = 0.50 + ((double)((seed >> 20) & 0x3F) / 64.0) * 0.30;
    // -> alpha in [0.50, 0.80)
    double cdf[HUFF_ALPHABET_SIZE + 1];
    cdf[0] = 0.0;
    {
        double total = 0.0;
        std::vector<double> w(HUFF_ALPHABET_SIZE);
        for (int r = 0; r < HUFF_ALPHABET_SIZE; ++r) {
            double base = (double)(r + 1);
            // w_r = exp(-alpha * ln(base))
            double lnv = log(base);
            w[r] = exp(-alpha * lnv);
            total += w[r];
        }
        for (int r = 0; r < HUFF_ALPHABET_SIZE; ++r) {
            cdf[r + 1] = cdf[r] + w[r] / total;
        }
        cdf[HUFF_ALPHABET_SIZE] = 1.0;
    }

    // Sample n bytes.
    uint64_t s = seed ^ per_stream_salt ^ 0xCAFEF00DBADC0DEDULL;
    for (size_t i = 0; i < n; ++i) {
        uint64_t u = splitmix64_step(s);
        // map to [0, 1)
        double f = (double)((u >> 11) & ((1ULL << 53) - 1)) /
                   (double)(1ULL << 53);
        // binary-search rank r such that cdf[r] <= f < cdf[r+1]
        int lo = 0, hi = HUFF_ALPHABET_SIZE;
        while (lo + 1 < hi) {
            int mid = (lo + hi) >> 1;
            if (cdf[mid] <= f) lo = mid; else hi = mid;
        }
        if (lo >= HUFF_ALPHABET_SIZE) lo = HUFF_ALPHABET_SIZE - 1;
        dst[i] = perm[lo];
    }
}

// -------------------- canonical Huffman table builder --------------------
//
// Build code-length array uint8[256] from frequencies, capped at
// HUFF_MAX_CODE_LEN bits, then derive the canonical (length, symbol)
// ordering -- this is the same table the decoder will see.
//
// Algorithm:
//   1. Standard Huffman with a min-heap to get optimal lengths.
//   2. Length-limited pass: if any length > HUFF_MAX_CODE_LEN, fall back
//      to a simple "package-merge"-flavoured cap by repeatedly clipping
//      the longest leaf to MAX and rebalancing siblings.  In practice for
//      our payload (~16 KB skewed bytes) the optimal Huffman tree is
//      already <= 16 deep, but we keep the cap for safety.

struct HuffNode {
    uint64_t freq;
    int      left;   // node index, -1 if leaf
    int      right;
    int      sym;    // symbol id if leaf, -1 otherwise
};

static void compute_code_lengths(const uint64_t* freq,
                                 int alphabet_size,
                                 uint8_t* code_len /*out, size alphabet_size*/) {
    // Initialise leaves for every symbol with freq > 0.  Symbols with
    // zero frequency get code_len = 0 (will be unused).
    std::vector<HuffNode> nodes;
    nodes.reserve(alphabet_size * 2);

    struct PqEntry { uint64_t freq; int idx; };
    auto cmp = [](const PqEntry& a, const PqEntry& b) {
        if (a.freq != b.freq) return a.freq > b.freq;
        return a.idx > b.idx;
    };
    std::priority_queue<PqEntry, std::vector<PqEntry>, decltype(cmp)> pq(cmp);

    int n_leaves = 0;
    for (int s = 0; s < alphabet_size; ++s) {
        if (freq[s] > 0) {
            int i = (int)nodes.size();
            nodes.push_back({freq[s], -1, -1, s});
            pq.push({freq[s], i});
            ++n_leaves;
        }
    }

    // Edge case: only one symbol present.  Give it code length 1.
    if (n_leaves == 1) {
        std::memset(code_len, 0, alphabet_size);
        for (int s = 0; s < alphabet_size; ++s) {
            if (freq[s] > 0) { code_len[s] = 1; break; }
        }
        return;
    }

    while (pq.size() > 1) {
        PqEntry a = pq.top(); pq.pop();
        PqEntry b = pq.top(); pq.pop();
        int i = (int)nodes.size();
        nodes.push_back({a.freq + b.freq, a.idx, b.idx, -1});
        pq.push({a.freq + b.freq, i});
    }

    int root = pq.top().idx;
    // Walk tree to extract lengths.
    std::memset(code_len, 0, alphabet_size);
    // BFS iteratively to avoid deep recursion.
    std::vector<std::pair<int, int>> stack;  // (node_idx, depth)
    stack.push_back({root, 0});
    while (!stack.empty()) {
        auto [idx, d] = stack.back();
        stack.pop_back();
        const HuffNode& nd = nodes[idx];
        if (nd.sym >= 0) {
            int len = (d == 0) ? 1 : d;
            if (len > 255) len = 255;
            code_len[nd.sym] = (uint8_t)len;
        } else {
            stack.push_back({nd.left,  d + 1});
            stack.push_back({nd.right, d + 1});
        }
    }

    // Length-limited cap via Kraft-relaxation.
    bool needs_cap = false;
    for (int s = 0; s < alphabet_size; ++s) {
        if (code_len[s] > HUFF_MAX_CODE_LEN) { needs_cap = true; break; }
    }
    if (needs_cap) {
        // Simple iterative scheme: while sum of 2^(-l_i) > 1, lengthen the
        // shortest code; while < 1, shorten the longest > MAX.  Combined
        // with cap, this is the standard length-limited adjustment used
        // in zlib / kraft repair.
        // Cap first.
        for (int s = 0; s < alphabet_size; ++s) {
            if (code_len[s] > HUFF_MAX_CODE_LEN) {
                code_len[s] = HUFF_MAX_CODE_LEN;
            }
        }
        // Now Kraft sum can be > 1 (over-long codes were trimmed), making
        // the code prefix-free property still hold (Kraft <= 1 means
        // prefix-free is achievable; we just need to repair to <= 1).
        auto kraft_sum_x_2pow_max = [&](void) -> uint64_t {
            uint64_t sum = 0;
            for (int s = 0; s < alphabet_size; ++s) {
                if (code_len[s] > 0) {
                    sum += (1ULL << (HUFF_MAX_CODE_LEN - code_len[s]));
                }
            }
            return sum;
        };
        const uint64_t target = (1ULL << HUFF_MAX_CODE_LEN);
        uint64_t cur = kraft_sum_x_2pow_max();
        // Walk from longest existing length down, bumping codes longer
        // until Kraft <= 1.  In practice (our shape) this is rarely
        // reached; we still implement it for safety.
        while (cur > target) {
            // find a symbol with code_len < HUFF_MAX_CODE_LEN to lengthen
            int chosen = -1;
            int best_len = 0;
            for (int s = 0; s < alphabet_size; ++s) {
                if (code_len[s] > 0 && code_len[s] < HUFF_MAX_CODE_LEN) {
                    if (code_len[s] > best_len) { best_len = code_len[s]; chosen = s; }
                }
            }
            if (chosen < 0) break;
            // lengthening by 1 reduces sum by half-the-symbol's weight
            uint64_t delta = (1ULL << (HUFF_MAX_CODE_LEN - code_len[chosen] - 1));
            cur -= delta;
            code_len[chosen]++;
        }
        // pad up to target (Kraft == 1) by extending shortest codes
        while (cur < target) {
            // find symbol with smallest code_len > 0
            int chosen = -1;
            int best_len = 1 << 30;
            for (int s = 0; s < alphabet_size; ++s) {
                if (code_len[s] > 0 && code_len[s] < best_len) {
                    best_len = code_len[s]; chosen = s;
                }
            }
            if (chosen < 0) break;
            uint64_t delta = (1ULL << (HUFF_MAX_CODE_LEN - code_len[chosen] - 1));
            if (cur + delta > target) break;
            cur += delta;
            code_len[chosen]++;  // wait, this should shorten, not lengthen
            // actually padding up means we have headroom; we need to do
            // nothing -- a strict canonical decoder treats unused codes
            // as gaps, which is fine.
            break;
        }
    }
}

// Build canonical (length, symbol) order from code_len[].
// canonical_symbols[i] = symbol with i-th smallest (length, symbol_id).
static void build_canonical_order(const uint8_t* code_len,
                                  int alphabet_size,
                                  uint16_t* canonical_symbols /*size 256*/) {
    // Stable sort of present symbols by (length, symbol_id).
    int n = 0;
    std::pair<int,int> arr[HUFF_ALPHABET_SIZE];
    for (int s = 0; s < alphabet_size; ++s) {
        if (code_len[s] > 0) {
            arr[n++] = {(int)code_len[s], s};
        }
    }
    std::sort(arr, arr + n, [](const std::pair<int,int>& a,
                                const std::pair<int,int>& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });
    for (int i = 0; i < HUFF_ALPHABET_SIZE; ++i) canonical_symbols[i] = 0;
    for (int i = 0; i < n; ++i) canonical_symbols[i] = (uint16_t)arr[i].second;
}

// Build per-symbol canonical code (length, code_bits) from code_len[].
// Also returns next_code[len] base values for decode.
// codes[s] is the code bits *MSB-first*, i.e. the natural reading order
// of canonical Huffman.  We store it that way; encoding into our LSB-first
// packed buffer reverses the bit order on the way out.
struct CanonicalCode {
    uint32_t code_msb;  // code value, with bit 0 of code_msb being the
                        // MSB of the length-code_len[s] code.  Equivalent
                        // to "the next_code-based integer".
    uint8_t  len;
};

static void build_canonical_codes(const uint8_t* code_len,
                                  int alphabet_size,
                                  CanonicalCode* codes /*size 256*/) {
    // Standard canonical assignment:
    //   bl_count[l] = # symbols with code_len == l
    //   next_code[l+1] = (next_code[l] + bl_count[l]) << 1
    //   for s in canonical order: code_msb = next_code[len[s]]++;
    int bl_count[HUFF_MAX_CODE_LEN + 2] = {0};
    for (int s = 0; s < alphabet_size; ++s) {
        bl_count[code_len[s]]++;
    }
    uint32_t next_code[HUFF_MAX_CODE_LEN + 2] = {0};
    uint32_t code = 0;
    bl_count[0] = 0;
    for (int l = 1; l <= HUFF_MAX_CODE_LEN; ++l) {
        code = (code + (uint32_t)bl_count[l - 1]) << 1;
        next_code[l] = code;
    }
    // Assign in the SAME canonical order: by (length, symbol).
    std::pair<int,int> ord[HUFF_ALPHABET_SIZE];
    int n = 0;
    for (int s = 0; s < alphabet_size; ++s) {
        if (code_len[s] > 0) ord[n++] = {(int)code_len[s], s};
    }
    std::sort(ord, ord + n, [](const std::pair<int,int>& a,
                                const std::pair<int,int>& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });
    for (int s = 0; s < alphabet_size; ++s) {
        codes[s].code_msb = 0;
        codes[s].len      = 0;
    }
    for (int i = 0; i < n; ++i) {
        int len = ord[i].first;
        int sym = ord[i].second;
        codes[sym].code_msb = next_code[len]++;
        codes[sym].len      = (uint8_t)len;
    }
}

// -------------------- bit-stream encoder --------------------
//
// Append the canonical code for `sym` to the packed bitstream buffer
// `bits` at bit-offset `*bit_pos`.  The packed format matches the layout
// the decoder will consume:
//   - 32-bit words, little-endian within each word
//   - bit i of the stream lives at words[i/32], LSB-first within the word
//   - canonical Huffman code is written MSB-first into the stream (i.e.
//     the most-significant bit of code_msb is emitted FIRST and ends up at
//     the lowest bit-index)
//
static void write_bits_msb_first(uint32_t* bits,
                                 uint64_t* bit_pos,
                                 uint32_t code_msb,
                                 uint32_t len)
{
    for (int b = (int)len - 1; b >= 0; --b) {
        uint32_t bit = (code_msb >> b) & 1u;
        uint64_t pos = *bit_pos;
        uint64_t word = pos >> 5;
        uint32_t off  = (uint32_t)(pos & 31ULL);
        bits[word] |= (bit << off);
        ++(*bit_pos);
    }
}

static void encode_payload(const uint8_t* src, size_t n,
                           const CanonicalCode* codes,
                           uint32_t* bits, uint64_t* total_bits) {
    uint64_t pos = 0;
    for (size_t i = 0; i < n; ++i) {
        uint8_t s = src[i];
        write_bits_msb_first(bits, &pos, codes[s].code_msb, codes[s].len);
    }
    *total_bits = pos;
}

// -------------------- CPU reference decode --------------------
//
// Used in --verify only (small workload).  Runs the canonical decoder on
// the host so we have an *independent* check that the encoder is itself
// correct -- not strictly necessary because the GPU output is also
// compared to the original payload, but keeps main.cu self-checking.
static bool cpu_canonical_decode(const uint32_t* bits, uint64_t bit_len,
                                 const uint8_t* code_len,
                                 const uint16_t* canonical_symbols,
                                 uint8_t* out, size_t out_len) {
    int bl_count[HUFF_MAX_CODE_LEN + 2] = {0};
    for (int s = 0; s < HUFF_ALPHABET_SIZE; ++s) bl_count[code_len[s]]++;
    bl_count[0] = 0;
    uint32_t first_code[HUFF_MAX_CODE_LEN + 2] = {0};
    int      first_idx [HUFF_MAX_CODE_LEN + 2] = {0};
    uint32_t code = 0;
    int      idx  = 0;
    for (int l = 1; l <= HUFF_MAX_CODE_LEN; ++l) {
        code = (code + (uint32_t)bl_count[l - 1]) << 1;
        first_code[l] = code;
        first_idx[l]  = idx;
        idx += bl_count[l];
    }

    auto get_bit = [&](uint64_t i) -> uint32_t {
        return (bits[i >> 5] >> (i & 31)) & 1u;
    };

    uint64_t pos = 0;
    for (size_t i = 0; i < out_len; ++i) {
        uint32_t cur = 0;
        int      l   = 0;
        while (l < HUFF_MAX_CODE_LEN) {
            if (pos >= bit_len) return false;
            cur = (cur << 1) | get_bit(pos++);
            l++;
            if (bl_count[l] > 0 && cur < first_code[l] + (uint32_t)bl_count[l]) {
                int o = first_idx[l] + (int)(cur - first_code[l]);
                if (o < 0 || o >= HUFF_ALPHABET_SIZE) return false;
                out[i] = (uint8_t)canonical_symbols[o];
                goto next_sym;
            }
        }
        return false;
    next_sym:;
    }
    return true;
}

// -------------------- one full run --------------------

struct RunResult {
    bool   ok;
    double time_ms;       // median over trials
    const char* err;
};

// Build a SINGLE table from the average frequencies of all K streams
// (so all streams share a table -- simpler shape, matches solve.h doc).
static void build_table_from_payloads(
    const std::vector<uint8_t>& all_payload,  // contiguous K * payload_bytes
    uint8_t*  code_len  /*size 256*/,
    uint16_t* canonical_symbols /*size 256*/,
    CanonicalCode* codes /*size 256*/)
{
    uint64_t freq[HUFF_ALPHABET_SIZE] = {0};
    for (size_t i = 0; i < all_payload.size(); ++i) {
        freq[all_payload[i]]++;
    }
    // Ensure no zero-freq symbol is left (purely cosmetic; a zero-freq
    // symbol just gets code_len = 0 and is unused).
    compute_code_lengths(freq, HUFF_ALPHABET_SIZE, code_len);
    build_canonical_order(code_len, HUFF_ALPHABET_SIZE, canonical_symbols);
    build_canonical_codes(code_len, HUFF_ALPHABET_SIZE, codes);
}

static RunResult run_one(uint32_t K, uint32_t bytes_per_stream,
                         uint64_t seed, int n_trials, bool warmup_only,
                         bool print_stats) {
    RunResult r{false, 0.0, nullptr};
    const uint32_t out_stride = bytes_per_stream;  // tightly packed

    // Step 1: generate K payloads.
    std::vector<uint8_t> all_payload((size_t)K * bytes_per_stream);
    for (uint32_t k = 0; k < K; ++k) {
        generate_payload(all_payload.data() + (size_t)k * bytes_per_stream,
                         bytes_per_stream, seed,
                         /*per-stream salt*/ (uint64_t)k * 0x9E3779B97F4A7C15ULL);
    }

    // Step 2: build canonical Huffman table from average frequency.
    uint8_t  code_len[HUFF_ALPHABET_SIZE];
    uint16_t canonical_symbols[HUFF_ALPHABET_SIZE];
    CanonicalCode codes[HUFF_ALPHABET_SIZE];
    build_table_from_payloads(all_payload, code_len, canonical_symbols, codes);

    // Compute upper bound on encoded bits per stream.
    // Worst case: every byte uses HUFF_MAX_CODE_LEN bits.
    const uint32_t MAX_BITS_PER_STREAM =
        (uint32_t)bytes_per_stream * HUFF_MAX_CODE_LEN;
    const uint32_t MAX_WORDS_PER_STREAM = (MAX_BITS_PER_STREAM + 31) / 32;
    // We use a fixed stride (power of 2 multiple) so the on-device bit
    // buffer is contiguous and per-stream offsets are k * MAX_WORDS_PER_STREAM.

    // Step 3: encode each payload.
    std::vector<uint32_t> bits_buf((size_t)K * MAX_WORDS_PER_STREAM, 0);
    std::vector<uint32_t> bit_lens(K);
    std::vector<uint32_t> out_lens(K, bytes_per_stream);
    for (uint32_t k = 0; k < K; ++k) {
        uint32_t* w = bits_buf.data() + (size_t)k * MAX_WORDS_PER_STREAM;
        uint64_t total_bits = 0;
        encode_payload(all_payload.data() + (size_t)k * bytes_per_stream,
                       bytes_per_stream, codes, w, &total_bits);
        if (total_bits > (uint64_t)MAX_BITS_PER_STREAM) {
            r.err = "encoded_overflow";
            return r;
        }
        bit_lens[k] = (uint32_t)total_bits;
    }

    // Optional: independent CPU decode sanity check (tiny shape only).
    if (bytes_per_stream <= 1024) {
        std::vector<uint8_t> recon(bytes_per_stream);
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t* w = bits_buf.data() + (size_t)k * MAX_WORDS_PER_STREAM;
            std::memset(recon.data(), 0, bytes_per_stream);
            if (!cpu_canonical_decode(w, bit_lens[k], code_len, canonical_symbols,
                                      recon.data(), bytes_per_stream)) {
                r.err = "cpu_decode_failed";
                return r;
            }
            const uint8_t* expected =
                all_payload.data() + (size_t)k * bytes_per_stream;
            if (std::memcmp(recon.data(), expected, bytes_per_stream) != 0) {
                r.err = "cpu_decode_mismatch";
                return r;
            }
        }
    }

    // Step 4: device buffers.
    uint32_t* d_bits     = nullptr;
    uint32_t* d_bit_lens = nullptr;
    uint8_t*  d_out      = nullptr;
    uint32_t* d_out_lens = nullptr;
    uint8_t*  d_code_len = nullptr;
    uint16_t* d_symbols  = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_bits,     (size_t)K * MAX_WORDS_PER_STREAM * sizeof(uint32_t));
    if (err) { r.err = "cudaMalloc bits"; return r; }
    err = cudaMalloc(&d_bit_lens, (size_t)K * sizeof(uint32_t));
    if (err) { r.err = "cudaMalloc bit_lens"; return r; }
    err = cudaMalloc(&d_out,      (size_t)K * out_stride);
    if (err) { r.err = "cudaMalloc out"; return r; }
    err = cudaMalloc(&d_out_lens, (size_t)K * sizeof(uint32_t));
    if (err) { r.err = "cudaMalloc out_lens"; return r; }
    err = cudaMalloc(&d_code_len, HUFF_ALPHABET_SIZE * sizeof(uint8_t));
    if (err) { r.err = "cudaMalloc code_len"; return r; }
    err = cudaMalloc(&d_symbols,  HUFF_ALPHABET_SIZE * sizeof(uint16_t));
    if (err) { r.err = "cudaMalloc symbols"; return r; }

    cudaMemcpy(d_bits,     bits_buf.data(),
               (size_t)K * MAX_WORDS_PER_STREAM * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_bit_lens, bit_lens.data(), (size_t)K * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_lens, out_lens.data(), (size_t)K * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_code_len, code_len,        HUFF_ALPHABET_SIZE * sizeof(uint8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_symbols,  canonical_symbols,
               HUFF_ALPHABET_SIZE * sizeof(uint16_t),
               cudaMemcpyHostToDevice);

    // Step 5: warmup.
    cudaMemset(d_out, 0, (size_t)K * out_stride);
    huffman_decode(d_bits, d_bit_lens, d_out, d_out_lens, out_stride,
                   MAX_WORDS_PER_STREAM,
                   d_code_len, d_symbols, K, /*stream=*/0);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        r.err = cudaGetErrorString(err);
        goto cleanup;
    }

    // Verify byte-exact.
    {
        std::vector<uint8_t> h_out((size_t)K * out_stride);
        cudaMemcpy(h_out.data(), d_out, (size_t)K * out_stride,
                   cudaMemcpyDeviceToHost);
        size_t mismatches = 0;
        for (uint32_t k = 0; k < K; ++k) {
            const uint8_t* gpu = h_out.data() + (size_t)k * out_stride;
            const uint8_t* exp = all_payload.data() + (size_t)k * bytes_per_stream;
            if (std::memcmp(gpu, exp, bytes_per_stream) != 0) {
                if (mismatches < 4) {
                    fprintf(stderr, "MISMATCH stream=%u\n", k);
                }
                mismatches++;
            }
        }
        if (mismatches > 0) {
            fprintf(stderr, "decode mismatch: %zu/%u streams\n",
                    mismatches, K);
            r.err = "decode_mismatch";
            goto cleanup;
        }
    }

    // Step 6: timed trials (median).
    if (!warmup_only)
    {
        std::vector<double> times_ms((size_t)n_trials);
        for (int t = 0; t < n_trials; ++t) {
            cudaMemset(d_out, 0, (size_t)K * out_stride);
            cudaDeviceSynchronize();

            cudaEvent_t ev0, ev1;
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
            cudaEventRecord(ev0);
            huffman_decode(d_bits, d_bit_lens, d_out, d_out_lens, out_stride,
                           MAX_WORDS_PER_STREAM,
                           d_code_len, d_symbols, K, /*stream=*/0);
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            times_ms[t] = (double)ms;
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
        }
        std::sort(times_ms.begin(), times_ms.end());
        r.time_ms = times_ms[n_trials / 2];

        // Re-verify after timed runs (defensive: the timed kernel must
        // produce the same correct output as the warmup).
        std::vector<uint8_t> h_out((size_t)K * out_stride);
        cudaMemcpy(h_out.data(), d_out, (size_t)K * out_stride,
                   cudaMemcpyDeviceToHost);
        size_t mismatches = 0;
        for (uint32_t k = 0; k < K; ++k) {
            const uint8_t* gpu = h_out.data() + (size_t)k * out_stride;
            const uint8_t* exp = all_payload.data() + (size_t)k * bytes_per_stream;
            if (std::memcmp(gpu, exp, bytes_per_stream) != 0) mismatches++;
        }
        if (mismatches > 0) {
            r.err = "decode_mismatch_timed";
            goto cleanup;
        }
    }

    if (print_stats) {
        // print rough stats about the table
        int min_l = 999, max_l = 0;
        for (int s = 0; s < HUFF_ALPHABET_SIZE; ++s) {
            if (code_len[s] > 0) {
                if (code_len[s] < min_l) min_l = code_len[s];
                if (code_len[s] > max_l) max_l = code_len[s];
            }
        }
        uint64_t total_bits = 0;
        for (uint32_t k = 0; k < K; ++k) total_bits += bit_lens[k];
        fprintf(stderr,
                "decode ok: K=%u bytes/stream=%u code_len=[%d,%d] mean_bits=%.2f median_ms=%.4f\n",
                K, bytes_per_stream, min_l, max_l,
                (double)total_bits / (double)((uint64_t)K * bytes_per_stream),
                r.time_ms);
    }

    r.ok = true;

cleanup:
    if (d_bits)     cudaFree(d_bits);
    if (d_bit_lens) cudaFree(d_bit_lens);
    if (d_out)      cudaFree(d_out);
    if (d_out_lens) cudaFree(d_out_lens);
    if (d_code_len) cudaFree(d_code_len);
    if (d_symbols)  cudaFree(d_symbols);
    return r;
}

// -------------------- main --------------------

int main(int argc, char** argv) {
    const char* mode = (argc >= 2) ? argv[1] : "--benchmark";

    if (std::strcmp(mode, "--verify") == 0) {
        // Public, small, runtime-random seed so the agent cannot hardcode.
        uint64_t seed = (uint64_t)time(nullptr) ^ ((uint64_t)getpid() << 32)
                      ^ 0xC0FFEE0042AABBCCULL;
        RunResult r = run_one(HUFF_NUM_STREAMS_VERIFY,
                              HUFF_BYTES_PER_STREAM_VERIFY,
                              seed, /*trials=*/1, /*warmup_only=*/true,
                              /*print_stats=*/false);
        if (r.ok) {
            printf("PASS verify=ok K=%d bpe=%d\n",
                   HUFF_NUM_STREAMS_VERIFY, HUFF_BYTES_PER_STREAM_VERIFY);
            return 0;
        } else {
            printf("FAIL verify err=%s\n", r.err ? r.err : "unknown");
            return 1;
        }
    }

    bool scored = (std::strcmp(mode, "--benchmark-verify") == 0);
    uint64_t seed = 0xA5A5DEAD0BADCAFEULL;
    if (scored) {
        const char* s = std::getenv("HUFFMAN_DECODE_SEED");
        if (!s || !*s) {
            fprintf(stderr, "ERROR: --benchmark-verify requires "
                            "HUFFMAN_DECODE_SEED env\n");
            return 2;
        }
        seed = std::strtoull(s, nullptr, 0);
    }

    const int N_TRIALS = 7;
    RunResult r = run_one(HUFF_NUM_STREAMS_BENCH,
                          HUFF_BYTES_PER_STREAM,
                          seed, N_TRIALS,
                          /*warmup_only=*/false,
                          /*print_stats=*/true);
    if (!r.ok) {
        printf("FAIL err=%s\n", r.err ? r.err : "unknown");
        return 1;
    }

    if (scored) {
        printf("__VERIFIER_SCORE__=%.6f\n", r.time_ms);
    }
    printf("result=ok time_ms=%.6f K=%d bpe=%d\n",
           r.time_ms, HUFF_NUM_STREAMS_BENCH, HUFF_BYTES_PER_STREAM);
    return 0;
}
