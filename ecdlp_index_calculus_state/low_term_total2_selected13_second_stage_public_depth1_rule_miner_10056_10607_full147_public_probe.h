#ifndef LOW_TERM_TOTAL2_SELECTED13_SALT_CONDITIONED_SECOND_STAGE_RULE_MINER_H
#define LOW_TERM_TOTAL2_SELECTED13_SALT_CONDITIONED_SECOND_STAGE_RULE_MINER_H

#include <stdint.h>

#define SELECTED13_SECOND_STAGE_RULE_SELECTOR_COUNT 3
#define SELECTED13_SECOND_STAGE_RULE_BEST_HELDOUT_ACCEPTED_BELOW_COUNT 80
#define SELECTED13_SECOND_STAGE_RULE_BEST_NO_LEAF_HELDOUT_ACCEPTED_BELOW_COUNT 80
#define SELECTED13_SECOND_STAGE_RULE_BEST_LOTO_HELDOUT_ACCEPTED_BELOW_COUNT 77

typedef struct {
  uint64_t selector_index;
  uint64_t selector_id_u64;
  uint64_t allow_leaf_field;
  uint64_t max_depth;
  uint64_t rule_count;
  uint64_t accepted_below_rho_transfer_count;
  uint64_t heldout_accepted_below_rho_transfer_count;
  uint64_t loto_accepted_below_rho_transfer_count;
  uint64_t loto_heldout_accepted_below_rho_transfer_count;
} selected13_second_stage_rule_selector_t;

static const selected13_second_stage_rule_selector_t SELECTED13_SECOND_STAGE_RULE_SELECTORS[] = {
  {0ULL, 10323921819458462629ULL, 0ULL, 1ULL, 1ULL, 80ULL, 80ULL, 77ULL, 77ULL},
  {1ULL, 13626295238546658417ULL, 0ULL, 2ULL, 1ULL, 80ULL, 80ULL, 77ULL, 77ULL},
  {2ULL, 11813352428996949014ULL, 0ULL, 3ULL, 1ULL, 80ULL, 80ULL, 77ULL, 77ULL},
};

#endif
