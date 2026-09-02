#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_RESIDUAL_COEFF_GAP_MANIFEST_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_RESIDUAL_COEFF_GAP_MANIFEST_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_COEFF_GAP_TRANSFER 10376ULL
#define SELECTED13_PRIORITY0_COEFF_GAP_ROW_REQUEST_U64 15227462883308766969ULL
#define SELECTED13_PRIORITY0_COEFF_GAP_LANE_COUNT 3
#define SELECTED13_PRIORITY0_COEFF_GAP_FORM_COUNT 2
#define SELECTED13_PRIORITY0_COEFF_GAP_ACCEPTED_EXPORT_COUNT 0ULL
#define SELECTED13_PRIORITY0_COEFF_GAP_RELATION_DERIVED_ECDLP 0ULL

#define SELECTED13_SOURCE_CLASS_DIRECT_CERT 1ULL
#define SELECTED13_SOURCE_CLASS_SHARED_PRODUCT_MISSING_COEFFS 2ULL
#define SELECTED13_SOURCE_CLASS_DIRECT_MISSING_COEFFS 3ULL
#define SELECTED13_SOURCE_CLASS_UNVERIFIED 4ULL

typedef struct {
  uint64_t hint_local_index;
  uint64_t source_class_code;
  uint64_t hint_hash_u64;
  uint64_t source_row_hash_u64;
  uint64_t residual_mask;
  uint64_t coefficient_union_mask;
  uint64_t residual_overlap_union_mask;
  uint64_t uncovered_residual_mask;
  uint64_t coefficient_form_count;
  uint64_t coefficient_rank_mod_order;
  uint64_t order;
  uint64_t form_start;
  uint64_t form_count;
  uint64_t same_salt_pair;
  uint64_t direct_public_key_verified;
} selected13_priority0_coeff_gap_lane_t;

typedef struct {
  uint64_t hint_local_index;
  uint64_t form_index;
  uint64_t form_hash_u64;
  uint64_t form_support_mask;
  uint64_t residual_overlap_mask;
  uint64_t rhs;
} selected13_priority0_coeff_gap_form_t;

static const selected13_priority0_coeff_gap_lane_t SELECTED13_PRIORITY0_COEFF_GAP_LANES[] = {
  {0ULL, 1ULL, 13039068490116701655ULL, 5275705598190304756ULL, 13068ULL, 2560ULL, 512ULL, 12556ULL, 2ULL, 2ULL, 11779ULL, 0ULL, 2ULL, 1ULL, 1ULL},
  {1ULL, 2ULL, 3019703627563812167ULL, 3798422369699625013ULL, 13132ULL, 0ULL, 0ULL, 13132ULL, 0ULL, 0ULL, 0ULL, 2ULL, 0ULL, 1ULL, 1ULL},
  {2ULL, 2ULL, 13390195903203261877ULL, 3695383160683521145ULL, 13132ULL, 0ULL, 0ULL, 13132ULL, 0ULL, 0ULL, 0ULL, 2ULL, 0ULL, 1ULL, 1ULL},
};

static const selected13_priority0_coeff_gap_form_t SELECTED13_PRIORITY0_COEFF_GAP_FORMS[] = {
  {0ULL, 0ULL, 7535820120347288277ULL, 2560ULL, 512ULL, 6812ULL},
  {0ULL, 1ULL, 6514666582442510976ULL, 2560ULL, 512ULL, 8173ULL},
};

#endif
