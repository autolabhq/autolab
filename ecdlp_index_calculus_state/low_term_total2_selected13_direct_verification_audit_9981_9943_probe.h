#ifndef LOW_TERM_TOTAL2_SELECTED13_DIRECT_VERIFICATION_AUDIT_H
#define LOW_TERM_TOTAL2_SELECTED13_DIRECT_VERIFICATION_AUDIT_H

#include <stdint.h>

#define SELECTED13_DIRECT_VERIFICATION_AUDIT_TARGET_COUNT 2
#define SELECTED13_DIRECT_VERIFICATION_AUDIT_VERIFIED_COUNT 0
#define SELECTED13_DIRECT_VERIFICATION_AUDIT_RELATION_EXPORT_COUNT 0
#define SELECTED13_DIRECT_VERIFICATION_AUDIT_RELATION_DERIVED_ECDLP 0

#define SELECTED13_DIRECT_CLASS_CERT_PRESENT 1ULL
#define SELECTED13_DIRECT_CLASS_NO_RELATIONS 2ULL
#define SELECTED13_DIRECT_CLASS_RANK_DEFICIENT 3ULL
#define SELECTED13_DIRECT_CLASS_SECRET_MISSING 4ULL
#define SELECTED13_DIRECT_CLASS_EVIDENCE_MISSING 5ULL

typedef struct {
  uint64_t transfer_index;
  uint64_t backfill_row_id_u64;
  uint64_t backfill_row_check_hash_u64;
  uint64_t classification_code;
  uint64_t direct_public_key_verified;
  uint64_t direct_certificate_count;
  uint64_t best_product_rank;
  uint64_t best_product_relation_count;
  uint64_t best_source_rank;
  uint64_t best_source_relation_count;
  uint64_t fresh_direct_verification_required;
} selected13_direct_verification_audit_target_t;

static const selected13_direct_verification_audit_target_t SELECTED13_DIRECT_VERIFICATION_AUDIT_TARGETS[] = {
  {9981ULL, 85214203672131ULL, 9466797536760925517ULL, 3ULL, 0ULL, 0ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL},
  {9943ULL, 84151286056248ULL, 430448500522037891ULL, 2ULL, 0ULL, 0ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL},
};

#endif
