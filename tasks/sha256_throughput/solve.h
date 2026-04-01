#ifndef SOLVE_H
#define SOLVE_H

#include <stddef.h>
#include <stdint.h>

/*
 * Compute the SHA-256 hash of `len` bytes at `data`.
 * Write the 32-byte binary digest into `digest[0..31]`.
 *
 * Must match the FIPS 180-4 SHA-256 specification exactly.
 */
void sha256(const uint8_t *data, size_t len, uint8_t *digest);

#endif /* SOLVE_H */
