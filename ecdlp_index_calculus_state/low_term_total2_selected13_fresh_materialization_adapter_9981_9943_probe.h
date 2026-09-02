#ifndef LOW_TERM_TOTAL2_SELECTED13_FRESH_MATERIALIZATION_ADAPTER_H
#define LOW_TERM_TOTAL2_SELECTED13_FRESH_MATERIALIZATION_ADAPTER_H

#include <stdint.h>

#define SELECTED13_FRESH_MATERIALIZATION_TARGET_COUNT 2
#define SELECTED13_FRESH_MATERIALIZATION_FAMILY_LANE_COUNT 6
#define SELECTED13_FRESH_MATERIALIZATION_DIRECT_VERIFIED_COUNT 0
#define SELECTED13_FRESH_MATERIALIZATION_RELATION_EXPORT_COUNT 0
#define SELECTED13_FRESH_MATERIALIZATION_RELATION_DERIVED_ECDLP 0

typedef struct {
  uint64_t transfer_index;
  uint64_t backfill_row_id_u64;
  uint64_t backfill_row_check_hash_u64;
  uint64_t direct_ops_over_rho_scaled;
  uint64_t direct_public_key_verified;
  uint64_t shared_product_public_key_verified;
  uint64_t full_gate_replay_result_code;
  uint64_t materialized;
  uint64_t needs_fresh_direct_verification;
  uint64_t below_rho_label;
} selected13_fresh_materialization_target_t;

typedef struct {
  uint64_t transfer_index;
  uint64_t family_mask;
  uint64_t source_tier_code;
  uint64_t source_secret_count;
  uint64_t candidate_form_count;
  uint64_t replay_result_code;
  uint64_t fresh_source_solve_required;
  uint64_t lane_hash_u64;
} selected13_fresh_materialization_lane_t;

static const selected13_fresh_materialization_target_t SELECTED13_FRESH_MATERIALIZATION_TARGETS[] = {
  {9981ULL, 85214203672131ULL, 9466797536760925517ULL, 72262774ULL, 0ULL, 0ULL, 2ULL, 1ULL, 1ULL, 1ULL},
  {9943ULL, 84151286056248ULL, 430448500522037891ULL, 64963504ULL, 0ULL, 0ULL, 2ULL, 1ULL, 1ULL, 1ULL},
};

static const selected13_fresh_materialization_lane_t SELECTED13_FRESH_MATERIALIZATION_LANES[] = {
  {9981ULL, 33ULL, 1ULL, 1ULL, 2ULL, 1ULL, 0ULL, 15625142965677174875ULL},
  {9981ULL, 17408ULL, 1ULL, 1ULL, 2ULL, 1ULL, 0ULL, 1868107030282002951ULL},
  {9981ULL, 34816ULL, 3ULL, 1ULL, 2ULL, 1ULL, 1ULL, 3634773673468780329ULL},
  {9943ULL, 33ULL, 3ULL, 1ULL, 2ULL, 1ULL, 1ULL, 9375066655858740447ULL},
  {9943ULL, 17408ULL, 2ULL, 1ULL, 2ULL, 1ULL, 0ULL, 12968638412138502884ULL},
  {9943ULL, 34816ULL, 2ULL, 1ULL, 2ULL, 1ULL, 0ULL, 16027564389323554833ULL},
};

#endif
