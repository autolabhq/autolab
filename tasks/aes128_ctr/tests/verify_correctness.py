#!/usr/bin/env python3
"""
Correctness verifier for the aes128_ctr benchmark.

Compiles a small test harness against solve.c and verifies AES-128-CTR output
against a built-in pure-Python reference implementation.

Exit 0 -> all tests passed
Exit 1 -> at least one test failed
"""

import subprocess
import sys
import os
import struct

APPDIR = "/app"

# ── Pure-Python AES-128-CTR reference ────────────────────────────────────────

SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]
RCON = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]


def _xtime(a):
    return ((a << 1) ^ 0x1b) & 0xff if (a & 0x80) else (a << 1) & 0xff


def _mix_col(col):
    a, b, c, d = col
    return [
        _xtime(a) ^ _xtime(b) ^ b ^ c ^ d,
        a ^ _xtime(b) ^ _xtime(c) ^ c ^ d,
        a ^ b ^ _xtime(c) ^ _xtime(d) ^ d,
        _xtime(a) ^ a ^ b ^ c ^ _xtime(d),
    ]


def _key_expand(key):
    rk = [list(key[:16])]
    for r in range(1, 11):
        prev = rk[r - 1]
        w = list(prev)
        w[0] = SBOX[prev[13]] ^ RCON[r] ^ prev[0]
        w[1] = SBOX[prev[14]] ^ prev[1]
        w[2] = SBOX[prev[15]] ^ prev[2]
        w[3] = SBOX[prev[12]] ^ prev[3]
        for j in range(4, 16):
            w[j] = w[j - 4] ^ prev[j]
        rk.append(w)
    return rk


def _aes_block(rk, block):
    """Encrypt one 16-byte block (column-major state layout)."""
    s = list(block[:16])
    # AddRoundKey 0
    for i in range(16):
        s[i] ^= rk[0][i]
    for rnd in range(1, 10):
        # SubBytes
        for i in range(16):
            s[i] = SBOX[s[i]]
        # ShiftRows
        t = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t
        t = s[2]; s[2] = s[10]; s[10] = t
        t = s[6]; s[6] = s[14]; s[14] = t
        t = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t
        # MixColumns
        for c in range(4):
            mc = _mix_col([s[4*c], s[4*c+1], s[4*c+2], s[4*c+3]])
            for i in range(4):
                s[4*c + i] = mc[i]
        # AddRoundKey
        for i in range(16):
            s[i] ^= rk[rnd][i]
    # Final round
    for i in range(16):
        s[i] = SBOX[s[i]]
    t = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t
    t = s[2]; s[2] = s[10]; s[10] = t
    t = s[6]; s[6] = s[14]; s[14] = t
    t = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t
    for i in range(16):
        s[i] ^= rk[10][i]
    return bytes(s)


def ref_aes128_ctr(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    """Pure-Python AES-128-CTR reference."""
    rk = _key_expand(list(key))
    counter = bytearray(iv)
    out = bytearray()
    for off in range(0, len(plaintext), 16):
        ks = _aes_block(rk, bytes(counter))
        chunk = plaintext[off:off + 16]
        out.extend(b ^ k for b, k in zip(chunk, ks))
        # Big-endian 128-bit increment
        for i in range(15, -1, -1):
            counter[i] = (counter[i] + 1) & 0xff
            if counter[i] != 0:
                break
    return bytes(out)


# ── C test harness ────────────────────────────────────────────────────────────
# Reads from stdin: key[16] || iv[16] || plaintext[N]
# Writes to stdout: ciphertext[N] as raw bytes (hex-encoded, one line)

HARNESS_SRC = r"""
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "solve.h"

int main(void) {
    /* Read key (16) + iv (16) + plaintext (arbitrary) from stdin. */
    size_t cap = 4096, len = 0;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return 1;
    int c;
    while ((c = fgetc(stdin)) != EOF) {
        if (len == cap) {
            cap *= 2;
            buf = (uint8_t *)realloc(buf, cap);
            if (!buf) return 1;
        }
        buf[len++] = (uint8_t)c;
    }
    if (len < 32) { free(buf); return 2; }
    const uint8_t *key = buf;
    const uint8_t *iv  = buf + 16;
    const uint8_t *pt  = buf + 32;
    size_t pt_len = len - 32;

    uint8_t *ct = (uint8_t *)malloc(pt_len ? pt_len : 1);
    if (!ct) { free(buf); return 1; }

    aes128_ctr_encrypt(key, iv, pt, ct, pt_len);

    for (size_t i = 0; i < pt_len; i++)
        printf("%02x", ct[i]);
    printf("\n");

    free(ct);
    free(buf);
    return 0;
}
"""

HARNESS_BIN = "/tmp/aes_verify_harness"


def build_harness():
    src = "/tmp/aes_verify_harness.c"
    with open(src, "w") as f:
        f.write(HARNESS_SRC)
    r = subprocess.run(
        ["gcc", "-O2", "-march=native", "-std=c99",
         "-I", APPDIR,
         "-o", HARNESS_BIN,
         src,
         os.path.join(APPDIR, "solve.c")],
        capture_output=True, text=True
    )
    try:
        os.unlink(src)
    except OSError:
        pass
    if r.returncode != 0:
        print("Harness build failed:", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)


def run_case(key: bytes, iv: bytes, pt: bytes) -> str:
    """Return hex-encoded ciphertext from the harness, or None on error."""
    inp = key + iv + pt
    r = subprocess.run([HARNESS_BIN], input=inp, capture_output=True, timeout=30)
    if r.returncode != 0:
        return None
    return r.stdout.decode().strip()


def make_tests():
    """Return list of (description, key, iv, plaintext) tuples."""
    tests = []

    # NIST SP 800-38A Appendix F.5.1 — AES-128-CTR
    nist_key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    nist_iv  = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
    nist_pt  = bytes.fromhex(
        "6bc1bee22e409f96e93d7e117393172a"
        "ae2d8a571e03ac9c9eb76fac45af8e51"
        "30c81c46a35ce411e5fbc1191a0a52ef"
        "f69f2445df4f9b17ad2b417be66c3710"
    )
    tests.append(("NIST F.5.1 full (4 blocks)", nist_key, nist_iv, nist_pt))

    # Partial block tests using NIST key/IV
    tests.append(("1-byte plaintext",   nist_key, nist_iv, nist_pt[:1]))
    tests.append(("15-byte plaintext",  nist_key, nist_iv, nist_pt[:15]))
    tests.append(("16-byte plaintext",  nist_key, nist_iv, nist_pt[:16]))
    tests.append(("17-byte plaintext",  nist_key, nist_iv, nist_pt[:17]))
    tests.append(("31-byte plaintext",  nist_key, nist_iv, nist_pt[:31]))
    tests.append(("32-byte plaintext",  nist_key, nist_iv, nist_pt[:32]))
    tests.append(("63-byte plaintext",  nist_key, nist_iv, nist_pt[:48] + b"\x00" * 15))
    tests.append(("64-byte all-zero",   nist_key, nist_iv, b"\x00" * 64))

    # CTR self-inverse: encrypt twice → back to plaintext
    double_enc = ref_aes128_ctr(nist_key, nist_iv,
                                 ref_aes128_ctr(nist_key, nist_iv, nist_pt))
    tests.append(("CTR self-inverse (double encrypt)", nist_key, nist_iv, double_enc))

    # Different key: all-zero key, all-zero IV
    zero_key = b"\x00" * 16
    zero_iv  = b"\x00" * 16
    tests.append(("All-zero key/IV, 16-byte zero PT",  zero_key, zero_iv, b"\x00" * 16))
    tests.append(("All-zero key/IV, 32-byte zero PT",  zero_key, zero_iv, b"\x00" * 32))
    tests.append(("All-zero key/IV, 48-byte zero PT",  zero_key, zero_iv, b"\x00" * 48))

    # Counter wrap: IV with last bytes near 0xff to exercise carry propagation
    wrap_iv = bytes.fromhex("000000000000000000000000ffffffff")
    tests.append(("Counter carry propagation (3 blocks)", zero_key, wrap_iv, b"\xab" * 48))

    # All-0xff plaintext, all-0xff key
    ff_key = b"\xff" * 16
    ff_iv  = bytes.fromhex("ffffffffffffffffffffffffffffffff")
    tests.append(("All-0xff key/IV, 16-byte all-0xff PT", ff_key, ff_iv, b"\xff" * 16))

    # Larger input (256 bytes)
    large_pt = bytes(range(256))
    tests.append(("256-byte sequential input, NIST key/IV", nist_key, nist_iv, large_pt))

    return tests


def main():
    build_harness()

    tests = make_tests()
    failures = 0

    for desc, key, iv, pt in tests:
        ref_hex = ref_aes128_ctr(key, iv, pt).hex()
        got_hex = run_case(key, iv, pt)

        if got_hex is None:
            print(f"FAIL: {desc}  [harness error]", file=sys.stderr)
            failures += 1
        elif got_hex == ref_hex:
            print(f"OK:   {desc}")
        else:
            print(f"FAIL: {desc}", file=sys.stderr)
            print(f"      got:      {got_hex[:64]}{'...' if len(got_hex) > 64 else ''}",
                  file=sys.stderr)
            print(f"      expected: {ref_hex[:64]}{'...' if len(ref_hex) > 64 else ''}",
                  file=sys.stderr)
            failures += 1

    # Verify NIST test vector explicitly (belt-and-suspenders)
    nist_key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    nist_iv  = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
    nist_pt  = bytes.fromhex(
        "6bc1bee22e409f96e93d7e117393172a"
        "ae2d8a571e03ac9c9eb76fac45af8e51"
        "30c81c46a35ce411e5fbc1191a0a52ef"
        "f69f2445df4f9b17ad2b417be66c3710"
    )
    nist_expected = (
        "874d6191b620e3261bef6864990db6ce"
        "9806f66b7970fdff8617187bb9fffdff"
        "5ae4df3edbd5d35e5b4f09020db03eab"
        "1e031dda2fbe03d1792170a0f3009cee"
    )
    got = run_case(nist_key, nist_iv, nist_pt)
    if got != nist_expected:
        print("FAIL: NIST SP 800-38A F.5.1 explicit check", file=sys.stderr)
        print(f"      got:      {got}", file=sys.stderr)
        print(f"      expected: {nist_expected}", file=sys.stderr)
        failures += 1
    else:
        print("OK:   NIST SP 800-38A F.5.1 explicit check")

    if failures:
        print(f"\n{failures}/{len(tests)+1} test(s) FAILED", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)+1} correctness tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
