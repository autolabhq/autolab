#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_FFE_DIRECT_RANK_EMISSION_PACKET_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_FFE_DIRECT_RANK_EMISSION_PACKET_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_FFE_TRANSFER 10376ULL
#define SELECTED13_PRIORITY0_FFE_ROW_REQUEST_U64 15227462883308766969ULL
#define SELECTED13_PRIORITY0_FFE_SELECTED_SUPPORT_MASK 65533ULL
#define SELECTED13_PRIORITY0_FFE_RESIDUAL_LANE_COUNT 3
#define SELECTED13_PRIORITY0_FFE_FAMILY_LANE_COUNT 3
#define SELECTED13_PRIORITY0_FFE_CONTROL_COUNT 3
#define SELECTED13_PRIORITY0_FFE_ACCEPTED_EXPORT_COUNT 0ULL
#define SELECTED13_PRIORITY0_FFE_RELATION_DERIVED_ECDLP 0ULL
#define SELECTED13_PRIORITY0_FFE_FAMILY_MASK_RANK 3ULL
#define SELECTED13_PRIORITY0_FFE_RESIDUAL_MASK_RANK 2ULL
#define SELECTED13_PRIORITY0_FFE_COMBINED_MASK_RANK 5ULL

typedef struct {
  uint64_t transfer_index;
  uint64_t row_request_id_u64;
  uint64_t selected_support_mask;
  uint64_t target_salts[2];
  uint64_t direct_ops_over_rho_scaled;
  uint64_t family_mask_rank;
  uint64_t residual_mask_rank;
  uint64_t combined_mask_rank;
  uint64_t phase_code;
} selected13_priority0_ffe_target_t;

typedef struct {
  uint64_t family_index;
  uint64_t family_mask;
  uint64_t family_lane_hash_u64;
  uint64_t selected_support_covers_family;
} selected13_priority0_ffe_family_lane_t;

typedef struct {
  uint64_t hint_local_index;
  uint64_t hint_hash_u64;
  uint64_t source_row_hash_u64;
  uint64_t residual_lane_hash_u64;
  uint64_t source_selected_support_mask;
  uint64_t missing_target_support_mask;
  uint64_t extra_source_support_mask;
  uint64_t source_selector_u64;
  uint64_t source_top_k;
  uint64_t same_salt_pair;
  uint64_t direct_public_key_verified;
  uint64_t target_gap_exact;
} selected13_priority0_ffe_residual_lane_t;

typedef struct {
  uint64_t control_index;
  uint64_t transfer_index;
  uint64_t row_request_id_u64;
  uint64_t selected_support_mask;
  uint64_t derived_secret;
  uint64_t control_hash_u64;
  uint64_t relation_derived_ecdlp;
} selected13_priority0_ffe_control_t;

static const selected13_priority0_ffe_target_t SELECTED13_PRIORITY0_FFE_TARGET = {
  10376ULL,
  15227462883308766969ULL,
  65533ULL,
  {168ULL, 173ULL},
  68613139ULL,
  3ULL,
  2ULL,
  5ULL,
  1ULL
};

static const selected13_priority0_ffe_family_lane_t SELECTED13_PRIORITY0_FFE_FAMILY_LANES[] = {
  {0ULL, 34816ULL, 3970563750847567121ULL, 1ULL},
  {1ULL, 17408ULL, 5877233894344903382ULL, 1ULL},
  {2ULL, 33ULL, 1547686404788704459ULL, 1ULL},
};

static const selected13_priority0_ffe_residual_lane_t SELECTED13_PRIORITY0_FFE_RESIDUAL_LANES[] = {
  {0ULL, 13039068490116701655ULL, 5275705598190304756ULL, 7139530377084802544ULL, 52465ULL, 13068ULL, 0ULL, 7740325808165506689ULL, 12ULL, 1ULL, 1ULL, 1ULL},
  {1ULL, 3019703627563812167ULL, 3798422369699625013ULL, 18228329592615618854ULL, 52401ULL, 13132ULL, 0ULL, 11751282872521667808ULL, 12ULL, 1ULL, 1ULL, 1ULL},
  {2ULL, 13390195903203261877ULL, 3695383160683521145ULL, 8540841909473775939ULL, 52401ULL, 13132ULL, 0ULL, 11751282872521667808ULL, 16ULL, 1ULL, 1ULL, 1ULL},
};

static const selected13_priority0_ffe_control_t SELECTED13_PRIORITY0_FFE_CONTROLS[] = {
  {0ULL, 10595ULL, 11582100495532818341ULL, 65533ULL, 9742ULL, 13703686158880693475ULL, 1ULL},
  {1ULL, 10595ULL, 7111504223407624559ULL, 15309ULL, 9742ULL, 13732467143193834822ULL, 1ULL},
  {2ULL, 10595ULL, 11828427395712818429ULL, 15309ULL, 9742ULL, 2363463747439152101ULL, 1ULL},
};

#endif
