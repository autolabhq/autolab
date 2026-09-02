#ifndef LOW_TERM_TOTAL2_SELECTED13_SALT_CONDITIONED_DIRECT_REPLAY_PROBE_H
#define LOW_TERM_TOTAL2_SELECTED13_SALT_CONDITIONED_DIRECT_REPLAY_PROBE_H

#include <stdint.h>

#define SELECTED13_SALT_DIRECT_REPLAY_RECORD_COUNT 1
#define SELECTED13_SALT_DIRECT_REPLAY_ACCEPTED_COUNT 0
#define SELECTED13_SALT_DIRECT_REPLAY_BELOW_RHO_ACCEPTED_COUNT 0
#define SELECTED13_SALT_DIRECT_REPLAY_HELDOUT_BELOW_RHO_ACCEPTED_COUNT 0

typedef struct {
  uint64_t candidate_index;
  uint64_t candidate_id_u64;
  uint64_t transfer_index;
  uint64_t top_leaf_index;
  uint64_t known_positive_transfer;
  uint64_t below_rho;
  uint64_t public_key_verified;
  uint64_t relation_derived_ecdlp;
  uint64_t rank;
  uint64_t relation_count;
  uint64_t derived_secret;
  uint64_t ops_over_rho_scaled_1e6;
  uint64_t status_code;
} selected13_salt_direct_replay_record_t;

static const selected13_salt_direct_replay_record_t SELECTED13_SALT_DIRECT_REPLAY_RECORDS[] = {
  {0ULL, 12296531619443554774ULL, 9839ULL, 94ULL, 0ULL, 1ULL, 0ULL, 0ULL, 2ULL, 2ULL, 0ULL, 700730ULL, 4ULL},
};

#endif
