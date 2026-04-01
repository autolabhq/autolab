#ifndef SOLVE_H
#define SOLVE_H

#include <stddef.h>
#include <stdint.h>

/*
 * AES-128 Counter Mode (CTR) encryption / decryption.
 *
 * Parameters
 * ----------
 * key       : 16-byte AES-128 key (FIPS 197 format)
 * iv        : 16-byte initial counter block (big-endian 128-bit integer,
 *             incremented by 1 for each subsequent 16-byte block)
 * plaintext : input buffer of `len` bytes
 * ciphertext: output buffer of `len` bytes (may alias plaintext for in-place)
 * len       : number of bytes to process (any length ≥ 0)
 *
 * CTR mode is self-inverse: encrypt(encrypt(pt)) == pt.
 * The function must match NIST SP 800-38A Appendix F.5.1 exactly.
 */
void aes128_ctr_encrypt(const uint8_t key[16], const uint8_t iv[16],
                         const uint8_t *plaintext, uint8_t *ciphertext,
                         size_t len);

#endif /* SOLVE_H */
