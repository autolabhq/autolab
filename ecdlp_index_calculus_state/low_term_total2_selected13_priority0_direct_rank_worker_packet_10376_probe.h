#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_DIRECT_RANK_WORKER_PACKET_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_DIRECT_RANK_WORKER_PACKET_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_TRANSFER 10376
#define SELECTED13_PRIORITY0_ROW_REQUEST_U64 15227462883308766969ULL
#define SELECTED13_PRIORITY0_SELECTED_SUPPORT_MASK 65533ULL
#define SELECTED13_PRIORITY0_HINT_COUNT 3
#define SELECTED13_PRIORITY0_FAMILY_LANE_COUNT 3
#define SELECTED13_PRIORITY0_CONTROL_COUNT 3
#define SELECTED13_PRIORITY0_RELATION_EXPORT_COUNT 0
#define SELECTED13_PRIORITY0_RELATION_DERIVED_ECDLP 0

typedef struct {
  uint64_t priority_rank;
  uint64_t priority_score;
  uint64_t packet_index;
  uint64_t global_row_index;
  uint64_t transfer_index;
  uint64_t row_request_id_u64;
  uint64_t selected_support_mask;
  uint64_t target_salts[2];
  uint64_t direct_ops_over_rho_scaled;
  uint64_t worker_class_code;
  uint64_t classification_code;
} selected13_priority0_target_t;

typedef struct {
  uint64_t family_index;
  uint64_t family_mask;
  uint64_t selected_support_covers_family;
} selected13_priority0_family_lane_t;

typedef struct {
  uint64_t hint_local_index;
  uint64_t hint_hash_u64;
  uint64_t source_row_hash_u64;
  uint64_t source_selector_u64;
  uint64_t source_selected_support_mask;
  uint64_t missing_target_support_mask;
  uint64_t extra_source_support_mask;
  uint64_t support_delta_popcount;
  uint64_t support_overlap_count;
  uint64_t support_jaccard_scaled_1e6;
  uint64_t salt_overlap_count;
  uint64_t salt_delta_min;
  uint64_t direct_public_key_verified;
  uint64_t public_product_gate_selected;
  uint64_t shared_product_public_key_verified;
  uint64_t direct_ops_over_rho_scaled;
} selected13_priority0_hint_lane_t;

typedef struct {
  uint64_t transfer_index;
  uint64_t row_request_id_u64;
  uint64_t selected_support_mask;
  uint64_t derived_secret;
  uint64_t relation_derived_ecdlp;
} selected13_priority0_control_t;

static const selected13_priority0_target_t SELECTED13_PRIORITY0_TARGET = {
  0ULL,
  1551239ULL,
  10ULL,
  30ULL,
  10376ULL,
  15227462883308766969ULL,
  65533ULL,
  {168ULL, 173ULL},
  68613139ULL,
  3ULL,
  5ULL
};

static const selected13_priority0_family_lane_t SELECTED13_PRIORITY0_FAMILY_LANES[] = {
  {0ULL, 34816ULL, 1ULL},
  {1ULL, 17408ULL, 1ULL},
  {2ULL, 33ULL, 1ULL},
};

static const selected13_priority0_hint_lane_t SELECTED13_PRIORITY0_HINT_LANES[] = {
  {0ULL, 13039068490116701655ULL, 5275705598190304756ULL, 7740325808165506689ULL, 52465ULL, 13068ULL, 0ULL, 6ULL, 9ULL, 600000ULL, 2ULL, 0ULL, 1ULL, 0ULL, 0ULL, 78832117ULL},
  {1ULL, 3019703627563812167ULL, 3798422369699625013ULL, 11751282872521667808ULL, 52401ULL, 13132ULL, 0ULL, 7ULL, 8ULL, 533333ULL, 2ULL, 0ULL, 1ULL, 1ULL, 1ULL, 69343066ULL},
  {2ULL, 13390195903203261877ULL, 3695383160683521145ULL, 11751282872521667808ULL, 52401ULL, 13132ULL, 0ULL, 7ULL, 8ULL, 533333ULL, 2ULL, 0ULL, 1ULL, 1ULL, 1ULL, 69343066ULL},
};

static const selected13_priority0_control_t SELECTED13_PRIORITY0_CONTROLS[] = {
  {10595ULL, 11582100495532818341ULL, 65533ULL, 9742ULL, 1ULL},
  {10595ULL, 7111504223407624559ULL, 15309ULL, 9742ULL, 1ULL},
  {10595ULL, 11828427395712818429ULL, 15309ULL, 9742ULL, 1ULL},
};

#endif
