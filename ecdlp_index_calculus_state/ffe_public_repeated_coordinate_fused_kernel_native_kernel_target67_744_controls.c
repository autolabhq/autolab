#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

enum {
  OP_KIND_ADD = 0,
  OP_KIND_DOUBLE = 1
};

typedef struct {
  const char *source_name;
  const char *event_key;
  int fanout;
  int reused;
  int op_kind;
  uint64_t p;
  uint64_t a1;
  uint64_t a2;
  uint64_t a3;
  uint64_t a4;
  uint64_t a6;
  uint64_t left_x;
  uint64_t left_y;
  uint64_t right_x;
  uint64_t right_y;
  uint64_t expected_slope_numerator;
  uint64_t expected_slope_denominator;
  uint64_t expected_slope_denominator_inverse;
  uint64_t expected_slope;
  uint64_t expected_intercept;
  uint64_t expected_x;
  uint64_t expected_y;
} case_t;

typedef struct {
  uint64_t slope_numerator;
  uint64_t slope_denominator;
  uint64_t slope_denominator_inverse;
  uint64_t slope;
  uint64_t intercept;
  uint64_t x;
  uint64_t y;
} kernel_out_t;

typedef struct {
  uint64_t add;
  uint64_t sub;
  uint64_t mul;
  uint64_t neg;
  uint64_t inv;
} op_counts_t;

static uint64_t mod_add(uint64_t a, uint64_t b, uint64_t p, op_counts_t *counts) {
  counts->add += 1;
  return (a % p + b % p) % p;
}

static uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t p, op_counts_t *counts) {
  counts->sub += 1;
  return (a % p + p - (b % p)) % p;
}

static uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t p, op_counts_t *counts) {
  counts->mul += 1;
  return ((a % p) * (b % p)) % p;
}

static uint64_t mod_neg(uint64_t a, uint64_t p, op_counts_t *counts) {
  counts->neg += 1;
  return (p - (a % p)) % p;
}

static uint64_t mod_inv(uint64_t a, uint64_t p, int *ok, op_counts_t *counts) {
  int64_t t = 0;
  int64_t new_t = 1;
  int64_t r = (int64_t)p;
  int64_t new_r = (int64_t)(a % p);
  counts->inv += 1;

  while (new_r != 0) {
    int64_t q = r / new_r;
    int64_t next_t = t - q * new_t;
    int64_t next_r = r - q * new_r;
    t = new_t;
    new_t = next_t;
    r = new_r;
    new_r = next_r;
  }
  if (r != 1) {
    *ok = 0;
    return 0;
  }
  if (t < 0) {
    t += (int64_t)p;
  }
  *ok = 1;
  return (uint64_t)t;
}

static int fused_candidate_point_kernel(const case_t *c, kernel_out_t *out, op_counts_t *counts) {
  int ok = 1;
  uint64_t slope_numerator = 0;
  uint64_t slope_denominator = 0;

  if (c->op_kind == OP_KIND_ADD) {
    slope_numerator = mod_sub(c->right_y, c->left_y, c->p, counts);
    slope_denominator = mod_sub(c->right_x, c->left_x, c->p, counts);
  } else if (c->op_kind == OP_KIND_DOUBLE) {
    uint64_t x_sq = mod_mul(c->left_x, c->left_x, c->p, counts);
    uint64_t three_x_sq = mod_mul(3, x_sq, c->p, counts);
    uint64_t two_a2_x = mod_mul((2 * c->a2) % c->p, c->left_x, c->p, counts);
    uint64_t numerator_tmp = mod_add(three_x_sq, two_a2_x, c->p, counts);
    numerator_tmp = mod_add(numerator_tmp, c->a4, c->p, counts);
    uint64_t a1_y = mod_mul(c->a1, c->left_y, c->p, counts);
    slope_numerator = mod_sub(numerator_tmp, a1_y, c->p, counts);
    uint64_t two_y = mod_mul(2, c->left_y, c->p, counts);
    uint64_t a1_x = mod_mul(c->a1, c->left_x, c->p, counts);
    uint64_t denominator_tmp = mod_add(two_y, a1_x, c->p, counts);
    slope_denominator = mod_add(denominator_tmp, c->a3, c->p, counts);
  } else {
    return 0;
  }

  uint64_t slope_denominator_inverse = mod_inv(slope_denominator, c->p, &ok, counts);
  if (!ok) {
    return 0;
  }
  uint64_t slope = mod_mul(slope_numerator, slope_denominator_inverse, c->p, counts);
  uint64_t slope_x1 = mod_mul(slope, c->left_x, c->p, counts);
  uint64_t intercept = mod_sub(c->left_y, slope_x1, c->p, counts);
  uint64_t slope_sq = mod_mul(slope, slope, c->p, counts);
  uint64_t a1_slope = mod_mul(c->a1, slope, c->p, counts);
  uint64_t x_tmp = mod_add(slope_sq, a1_slope, c->p, counts);
  x_tmp = mod_sub(x_tmp, c->a2, c->p, counts);
  x_tmp = mod_sub(x_tmp, c->left_x, c->p, counts);
  uint64_t x3 = mod_sub(x_tmp, c->right_x, c->p, counts);
  uint64_t slope_plus_a1 = mod_add(slope, c->a1, c->p, counts);
  uint64_t y_tmp = mod_mul(slope_plus_a1, x3, c->p, counts);
  y_tmp = mod_neg(y_tmp, c->p, counts);
  y_tmp = mod_sub(y_tmp, intercept, c->p, counts);
  uint64_t y3 = mod_sub(y_tmp, c->a3, c->p, counts);

  out->slope_numerator = slope_numerator;
  out->slope_denominator = slope_denominator;
  out->slope_denominator_inverse = slope_denominator_inverse;
  out->slope = slope;
  out->intercept = intercept;
  out->x = x3;
  out->y = y3;
  return 1;
}

static const case_t cases[] = {
  {"coord1114_744", "12:17:218", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 7537ULL, 8016ULL, 2128ULL, 2558ULL, 6741ULL, 3059ULL, 8989ULL, 2705ULL, 9753ULL},
  {"coord1114_744", "12:17:469", 2, 1, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 7537ULL, 8016ULL, 2128ULL, 2558ULL, 6741ULL, 3059ULL, 8989ULL, 2705ULL, 9753ULL},
  {"coord1114_744", "12:30:218", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 7537ULL, 8016ULL, 2128ULL, 2558ULL, 6741ULL, 3059ULL, 8989ULL, 2705ULL, 9753ULL},
  {"coord1114_744", "12:30:469", 2, 1, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 7537ULL, 8016ULL, 2128ULL, 2558ULL, 6741ULL, 3059ULL, 8989ULL, 2705ULL, 9753ULL},
  {"coord1114_744", "12:174:218", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 7537ULL, 8016ULL, 4102ULL, 2558ULL, 6741ULL, 7122ULL, 927ULL, 9251ULL, 9216ULL},
  {"coord1114_744", "12:174:469", 2, 1, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 7537ULL, 8016ULL, 4102ULL, 2558ULL, 6741ULL, 7122ULL, 927ULL, 9251ULL, 9216ULL},
  {"coord1114_744", "12:210:218", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 7537ULL, 8016ULL, 4102ULL, 2558ULL, 6741ULL, 7122ULL, 927ULL, 9251ULL, 9216ULL},
  {"coord1114_744", "12:210:469", 2, 1, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 7537ULL, 8016ULL, 4102ULL, 2558ULL, 6741ULL, 7122ULL, 927ULL, 9251ULL, 9216ULL},
  {"coord161_720", "2:1:149", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 2476ULL, 3749ULL, 9638ULL, 7300ULL, 7833ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_720", "2:1:186", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 2476ULL, 3749ULL, 9638ULL, 7300ULL, 7833ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_720", "2:1:296", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 2476ULL, 3749ULL, 9638ULL, 7300ULL, 7833ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_720", "2:130:149", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:130:186", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:130:296", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:144:149", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:144:186", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:144:296", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:331:149", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 2476ULL, 3749ULL, 9638ULL, 7300ULL, 7833ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_720", "2:331:186", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 2476ULL, 3749ULL, 9638ULL, 7300ULL, 7833ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_720", "2:331:296", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 2476ULL, 3749ULL, 9638ULL, 7300ULL, 7833ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_728", "2:4:204", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:4:377", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:4:429", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:4:466", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:202:204", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 2476ULL, 3749ULL, 9638ULL, 7300ULL, 7833ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_728", "2:202:377", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 2476ULL, 3749ULL, 9638ULL, 7300ULL, 7833ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_728", "2:202:429", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 2476ULL, 3749ULL, 9638ULL, 7300ULL, 7833ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_728", "2:202:466", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 3914ULL, 2476ULL, 3749ULL, 9638ULL, 7300ULL, 7833ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_728", "2:226:204", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:226:377", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:226:429", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:226:466", 1, 0, 0, 9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 5888ULL, 2476ULL, 3749ULL, 7664ULL, 7300ULL, 7833ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
};

int main(void) {
  size_t case_count = sizeof(cases) / sizeof(cases[0]);
  size_t verified_count = 0;
  size_t failure_count = 0;
  size_t result_match_count = 0;
  size_t register_match_count = 0;
  size_t reused_case_count = 0;
  op_counts_t total_counts = {0, 0, 0, 0, 0};

  for (size_t i = 0; i < case_count; ++i) {
    const case_t *c = &cases[i];
    kernel_out_t out = {0, 0, 0, 0, 0, 0, 0};
    op_counts_t counts = {0, 0, 0, 0, 0};
    int ok = fused_candidate_point_kernel(c, &out, &counts);
    int case_failed = 0;
    if (c->reused) {
      reused_case_count += 1;
    }

    total_counts.add += counts.add;
    total_counts.sub += counts.sub;
    total_counts.mul += counts.mul;
    total_counts.neg += counts.neg;
    total_counts.inv += counts.inv;

    if (!ok) {
      fprintf(stderr, "kernel failed source=%s event=%s\n", c->source_name, c->event_key);
      case_failed = 1;
      failure_count += 1;
    }

    if (
      out.slope_numerator == c->expected_slope_numerator &&
      out.slope_denominator == c->expected_slope_denominator &&
      out.slope_denominator_inverse == c->expected_slope_denominator_inverse &&
      out.slope == c->expected_slope &&
      out.intercept == c->expected_intercept
    ) {
      register_match_count += 1;
    } else {
      fprintf(
        stderr,
        "register mismatch source=%s event=%s got=(%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ")\n",
        c->source_name,
        c->event_key,
        out.slope_numerator,
        out.slope_denominator,
        out.slope_denominator_inverse,
        out.slope,
        out.intercept
      );
      case_failed = 1;
      failure_count += 1;
    }

    if (out.x == c->expected_x && out.y == c->expected_y) {
      result_match_count += 1;
    } else {
      fprintf(
        stderr,
        "result mismatch source=%s event=%s expected=(%" PRIu64 ",%" PRIu64 ") got=(%" PRIu64 ",%" PRIu64 ")\n",
        c->source_name,
        c->event_key,
        c->expected_x,
        c->expected_y,
        out.x,
        out.y
      );
      case_failed = 1;
      failure_count += 1;
    }

    if (!case_failed) {
      verified_count += 1;
    }
  }

  uint64_t total_ops = total_counts.add + total_counts.sub + total_counts.mul + total_counts.neg + total_counts.inv;
  printf(
    "{"
    "\"case_count\":%zu,"
    "\"verified_count\":%zu,"
    "\"failure_count\":%zu,"
    "\"result_match_count\":%zu,"
    "\"register_match_count\":%zu,"
    "\"reused_case_count\":%zu,"
    "\"field_op_counts\":{"
    "\"add\":%" PRIu64 ","
    "\"sub\":%" PRIu64 ","
    "\"mul\":%" PRIu64 ","
    "\"neg\":%" PRIu64 ","
    "\"inv\":%" PRIu64 ","
    "\"total\":%" PRIu64
    "}"
    "}\n",
    case_count,
    verified_count,
    failure_count,
    result_match_count,
    register_match_count,
    reused_case_count,
    total_counts.add,
    total_counts.sub,
    total_counts.mul,
    total_counts.neg,
    total_counts.inv,
    total_ops
  );
  return failure_count == 0 ? 0 : 1;
}
