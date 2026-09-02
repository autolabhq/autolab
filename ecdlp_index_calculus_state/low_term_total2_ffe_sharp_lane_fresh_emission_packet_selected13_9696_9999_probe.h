#ifndef LOW_TERM_TOTAL2_FFE_SHARP_LANE_FRESH_EMISSION_PACKET_H
#define LOW_TERM_TOTAL2_FFE_SHARP_LANE_FRESH_EMISSION_PACKET_H

#include <stdint.h>

#define SELECTED13_FRESH_EMISSION_TARGET_COUNT 2
#define SELECTED13_FRESH_EMISSION_ROW_SLOT_COUNT 6
#define SELECTED13_FRESH_EMISSION_FAMILY_LANE_COUNT 6
#define SELECTED13_FRESH_EMISSION_ACCEPTED_EXPORT_COUNT 0

#define SELECTED13_REPLAY_RESULT_SOURCE_SECRET_ONLY 1ULL
#define SELECTED13_REPLAY_RESULT_INCONSISTENT 2ULL
#define SELECTED13_REPLAY_RESULT_NO_UNIQUE_SECRET 3ULL
#define SELECTED13_REPLAY_RESULT_UNVERIFIED_NEW_DERIVATION 4ULL

typedef struct {
  uint64_t transfer_index;
  uint64_t priority_order;
  uint64_t row_slot_count;
  uint64_t family_lane_count;
  uint64_t full_family_row_id_u64;
  uint64_t full_family_row_hash_u64;
  uint64_t salt0;
  uint64_t salt1;
  uint64_t salt_gap;
  uint64_t full_gate_replay_result_code;
  uint64_t accepted_backfill_export_count;
  uint64_t fresh_emission_required;
} selected13_fresh_emission_target_t;

typedef struct {
  uint64_t transfer_index;
  uint64_t family_mask;
  uint64_t source_tier_code;
  uint64_t source_secret_count;
  uint64_t candidate_form_count;
  uint64_t replay_result_code;
  uint64_t same_row_key_hint;
  uint64_t one_salt_neighbor_hint;
  uint64_t fresh_source_solve_required;
  uint64_t lane_hash_u64;
} selected13_fresh_emission_family_lane_t;

static const selected13_fresh_emission_target_t SELECTED13_FRESH_EMISSION_TARGETS[] = {
  {9981ULL, 0ULL, 3ULL, 3ULL, 85214203672131ULL, 9466797536760925517ULL, 171ULL, 173ULL, 2ULL, 2ULL, 0ULL, 1ULL},
  {9943ULL, 1ULL, 3ULL, 3ULL, 84151286056248ULL, 430448500522037891ULL, 167ULL, 175ULL, 8ULL, 2ULL, 0ULL, 1ULL},
};

static const selected13_fresh_emission_family_lane_t SELECTED13_FRESH_EMISSION_FAMILY_LANES[] = {
  {9981ULL, 33ULL, 1ULL, 1ULL, 2ULL, 1ULL, 1ULL, 0ULL, 0ULL, 15625142965677174875ULL},
  {9981ULL, 17408ULL, 1ULL, 1ULL, 2ULL, 1ULL, 1ULL, 0ULL, 0ULL, 1868107030282002951ULL},
  {9981ULL, 34816ULL, 3ULL, 1ULL, 2ULL, 1ULL, 0ULL, 0ULL, 1ULL, 3634773673468780329ULL},
  {9943ULL, 33ULL, 3ULL, 1ULL, 2ULL, 1ULL, 0ULL, 0ULL, 1ULL, 9375066655858740447ULL},
  {9943ULL, 17408ULL, 2ULL, 1ULL, 2ULL, 1ULL, 0ULL, 1ULL, 0ULL, 12968638412138502884ULL},
  {9943ULL, 34816ULL, 2ULL, 1ULL, 2ULL, 1ULL, 0ULL, 1ULL, 0ULL, 16027564389323554833ULL},
};

#endif
