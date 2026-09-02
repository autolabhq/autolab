#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  uint64_t p;
  uint64_t a1;
  uint64_t a2;
  uint64_t a3;
  uint64_t a4;
  uint64_t a6;
  uint64_t left_x;
  uint64_t right_x;
  uint64_t right_y;
  uint64_t expected_slope_denominator;
  uint64_t expected_slope_denominator_inverse;
  size_t case_start;
  size_t case_count;
} shared_group_t;

typedef struct {
  const char *source_name;
  const char *event_key;
  int fanout;
  int reused;
  uint64_t left_y;
  uint64_t expected_slope_numerator;
  uint64_t expected_slope;
  uint64_t expected_intercept;
  uint64_t expected_x;
  uint64_t expected_y;
} case_t;

typedef struct {
  uint64_t slope_denominator;
  uint64_t slope_denominator_inverse;
} shared_state_t;

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

static int prepare_shared_denominator(const shared_group_t *g, shared_state_t *state, op_counts_t *counts) {
  int ok = 1;
  state->slope_denominator = mod_sub(g->right_x, g->left_x, g->p, counts);
  state->slope_denominator_inverse = mod_inv(state->slope_denominator, g->p, &ok, counts);
  return ok;
}

static int shared_denominator_candidate_kernel(
    const shared_group_t *g,
    const shared_state_t *state,
    const case_t *c,
    kernel_out_t *out,
    op_counts_t *counts) {
  uint64_t slope_numerator = mod_sub(g->right_y, c->left_y, g->p, counts);
  uint64_t slope = mod_mul(slope_numerator, state->slope_denominator_inverse, g->p, counts);
  uint64_t slope_x1 = mod_mul(slope, g->left_x, g->p, counts);
  uint64_t intercept = mod_sub(c->left_y, slope_x1, g->p, counts);
  uint64_t slope_sq = mod_mul(slope, slope, g->p, counts);
  uint64_t a1_slope = mod_mul(g->a1, slope, g->p, counts);
  uint64_t x_tmp = mod_add(slope_sq, a1_slope, g->p, counts);
  x_tmp = mod_sub(x_tmp, g->a2, g->p, counts);
  x_tmp = mod_sub(x_tmp, g->left_x, g->p, counts);
  uint64_t x3 = mod_sub(x_tmp, g->right_x, g->p, counts);
  uint64_t slope_plus_a1 = mod_add(slope, g->a1, g->p, counts);
  uint64_t y_tmp = mod_mul(slope_plus_a1, x3, g->p, counts);
  y_tmp = mod_neg(y_tmp, g->p, counts);
  y_tmp = mod_sub(y_tmp, intercept, g->p, counts);
  uint64_t y3 = mod_sub(y_tmp, g->a3, g->p, counts);

  out->slope_numerator = slope_numerator;
  out->slope_denominator = state->slope_denominator;
  out->slope_denominator_inverse = state->slope_denominator_inverse;
  out->slope = slope;
  out->intercept = intercept;
  out->x = x3;
  out->y = y3;
  return 1;
}

static const shared_group_t groups[] = {
  {9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 2476ULL, 3749ULL, 7300ULL, 7833ULL, 0, 39},
  {9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 79ULL, 3352ULL, 428ULL, 3273ULL, 4289ULL, 39, 10},
  {9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 7537ULL, 8016ULL, 2558ULL, 6741ULL, 49, 8},
  {9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 4979ULL, 7593ULL, 7605ULL, 2614ULL, 4894ULL, 57, 8},
  {9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 9245ULL, 4927ULL, 1187ULL, 5485ULL, 9719ULL, 65, 8},
  {9803ULL, 0ULL, 1ULL, 1ULL, 9791ULL, 9782ULL, 2614ULL, 473ULL, 2497ULL, 7662ULL, 8814ULL, 73, 4},
};

static const case_t cases[] = {
  {"coord161_672", "2:114:135", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_672", "2:114:217", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_672", "2:114:53", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_672", "2:322:135", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_672", "2:322:217", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_672", "2:322:53", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_672", "2:76:135", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_672", "2:76:217", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_672", "2:76:53", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:130:149", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:130:186", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:130:296", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:144:149", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:144:186", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:144:296", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_720", "2:1:149", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_720", "2:1:186", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_720", "2:1:296", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_720", "2:331:149", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_720", "2:331:186", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_720", "2:331:296", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_728", "2:202:204", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_728", "2:202:377", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_728", "2:202:429", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_728", "2:202:466", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_728", "2:226:204", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:226:377", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:226:429", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:226:466", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:4:204", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:4:377", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:4:429", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_728", "2:4:466", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_736", "2:150:14", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_736", "2:150:299", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_736", "2:150:96", 1, 0, 5888ULL, 7664ULL, 8343ULL, 1402ULL, 6696ULL, 1166ULL},
  {"coord161_736", "2:19:14", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_736", "2:19:299", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord161_736", "2:19:96", 1, 0, 3914ULL, 9638ULL, 1551ULL, 6249ULL, 6213ULL, 3539ULL},
  {"coord55_704", "0:210:109", 1, 0, 6558ULL, 3673ULL, 76ULL, 554ULL, 2344ULL, 7558ULL},
  {"coord55_704", "0:210:503", 1, 0, 6558ULL, 3673ULL, 76ULL, 554ULL, 2344ULL, 7558ULL},
  {"coord55_704", "0:294:109", 1, 0, 3244ULL, 6987ULL, 9275ULL, 5744ULL, 868ULL, 1621ULL},
  {"coord55_704", "0:294:503", 1, 0, 3244ULL, 6987ULL, 9275ULL, 5744ULL, 868ULL, 1621ULL},
  {"coord55_704", "0:37:109", 1, 0, 6558ULL, 3673ULL, 76ULL, 554ULL, 2344ULL, 7558ULL},
  {"coord55_704", "0:37:503", 1, 0, 6558ULL, 3673ULL, 76ULL, 554ULL, 2344ULL, 7558ULL},
  {"coord55_712", "0:11:29", 1, 0, 6558ULL, 3673ULL, 76ULL, 554ULL, 2344ULL, 7558ULL},
  {"coord55_712", "0:11:442", 1, 0, 6558ULL, 3673ULL, 76ULL, 554ULL, 2344ULL, 7558ULL},
  {"coord55_712", "0:170:29", 1, 0, 3244ULL, 6987ULL, 9275ULL, 5744ULL, 868ULL, 1621ULL},
  {"coord55_712", "0:170:442", 1, 0, 3244ULL, 6987ULL, 9275ULL, 5744ULL, 868ULL, 1621ULL},
  {"coord1114_744", "12:174:218", 1, 0, 3914ULL, 4102ULL, 7122ULL, 927ULL, 9251ULL, 9216ULL},
  {"coord1114_744", "12:174:469", 2, 1, 3914ULL, 4102ULL, 7122ULL, 927ULL, 9251ULL, 9216ULL},
  {"coord1114_744", "12:17:218", 1, 0, 5888ULL, 2128ULL, 3059ULL, 8989ULL, 2705ULL, 9753ULL},
  {"coord1114_744", "12:17:469", 2, 1, 5888ULL, 2128ULL, 3059ULL, 8989ULL, 2705ULL, 9753ULL},
  {"coord1114_744", "12:210:218", 1, 0, 3914ULL, 4102ULL, 7122ULL, 927ULL, 9251ULL, 9216ULL},
  {"coord1114_744", "12:210:469", 2, 1, 3914ULL, 4102ULL, 7122ULL, 927ULL, 9251ULL, 9216ULL},
  {"coord1114_744", "12:30:218", 1, 0, 5888ULL, 2128ULL, 3059ULL, 8989ULL, 2705ULL, 9753ULL},
  {"coord1114_744", "12:30:469", 2, 1, 5888ULL, 2128ULL, 3059ULL, 8989ULL, 2705ULL, 9753ULL},
  {"coord757_712", "7:191:233", 1, 0, 3914ULL, 3691ULL, 6628ULL, 0ULL, 371ULL, 1564ULL},
  {"coord757_712", "7:191:473", 1, 0, 3914ULL, 3691ULL, 6628ULL, 0ULL, 371ULL, 1564ULL},
  {"coord757_712", "7:19:233", 1, 0, 3914ULL, 3691ULL, 6628ULL, 0ULL, 371ULL, 1564ULL},
  {"coord757_712", "7:19:473", 1, 0, 3914ULL, 3691ULL, 6628ULL, 0ULL, 371ULL, 1564ULL},
  {"coord757_712", "7:245:233", 1, 0, 5888ULL, 1717ULL, 1827ULL, 6439ULL, 2139ULL, 6807ULL},
  {"coord757_712", "7:245:473", 1, 0, 5888ULL, 1717ULL, 1827ULL, 6439ULL, 2139ULL, 6807ULL},
  {"coord757_712", "7:347:233", 1, 0, 5888ULL, 1717ULL, 1827ULL, 6439ULL, 2139ULL, 6807ULL},
  {"coord757_712", "7:347:473", 1, 0, 5888ULL, 1717ULL, 1827ULL, 6439ULL, 2139ULL, 6807ULL},
  {"coord861_688", "9:151:336", 1, 0, 5708ULL, 5282ULL, 7250ULL, 2569ULL, 4247ULL, 7706ULL},
  {"coord861_688", "9:151:359", 1, 0, 5708ULL, 5282ULL, 7250ULL, 2569ULL, 4247ULL, 7706ULL},
  {"coord861_688", "9:296:336", 1, 0, 4094ULL, 6896ULL, 8916ULL, 9101ULL, 7962ULL, 4835ULL},
  {"coord861_688", "9:296:359", 1, 0, 4094ULL, 6896ULL, 8916ULL, 9101ULL, 7962ULL, 4835ULL},
  {"coord861_688", "9:5:336", 1, 0, 5708ULL, 5282ULL, 7250ULL, 2569ULL, 4247ULL, 7706ULL},
  {"coord861_688", "9:5:359", 1, 0, 5708ULL, 5282ULL, 7250ULL, 2569ULL, 4247ULL, 7706ULL},
  {"coord861_688", "9:69:336", 1, 0, 4094ULL, 6896ULL, 8916ULL, 9101ULL, 7962ULL, 4835ULL},
  {"coord861_688", "9:69:359", 1, 0, 4094ULL, 6896ULL, 8916ULL, 9101ULL, 7962ULL, 4835ULL},
  {"coord117_688", "1:229:286", 1, 0, 2638ULL, 9662ULL, 2207ULL, 7507ULL, 5473ULL, 680ULL},
  {"coord117_688", "1:229:42", 1, 0, 2638ULL, 9662ULL, 2207ULL, 7507ULL, 5473ULL, 680ULL},
  {"coord117_688", "1:68:286", 1, 0, 7164ULL, 5136ULL, 8253ULL, 422ULL, 7480ULL, 6431ULL},
  {"coord117_688", "1:68:42", 1, 0, 7164ULL, 5136ULL, 8253ULL, 422ULL, 7480ULL, 6431ULL},
};

int main(void) {
  size_t group_count = sizeof(groups) / sizeof(groups[0]);
  size_t case_count = sizeof(cases) / sizeof(cases[0]);
  size_t verified_count = 0;
  size_t failure_count = 0;
  size_t result_match_count = 0;
  size_t register_match_count = 0;
  size_t shared_prepare_match_count = 0;
  size_t reused_case_count = 0;
  op_counts_t total_counts = {0, 0, 0, 0, 0};

  for (size_t i = 0; i < group_count; ++i) {
    const shared_group_t *g = &groups[i];
    shared_state_t state = {0, 0};
    op_counts_t prepare_counts = {0, 0, 0, 0, 0};
    if (!prepare_shared_denominator(g, &state, &prepare_counts)) {
      fprintf(stderr, "shared denominator inverse failed group=%zu\n", i);
      failure_count += 1;
      continue;
    }
    total_counts.add += prepare_counts.add;
    total_counts.sub += prepare_counts.sub;
    total_counts.mul += prepare_counts.mul;
    total_counts.neg += prepare_counts.neg;
    total_counts.inv += prepare_counts.inv;

    if (
      state.slope_denominator == g->expected_slope_denominator &&
      state.slope_denominator_inverse == g->expected_slope_denominator_inverse
    ) {
      shared_prepare_match_count += 1;
    } else {
      fprintf(
        stderr,
        "shared prepare mismatch group=%zu expected=(%" PRIu64 ",%" PRIu64 ") got=(%" PRIu64 ",%" PRIu64 ")\n",
        i,
        g->expected_slope_denominator,
        g->expected_slope_denominator_inverse,
        state.slope_denominator,
        state.slope_denominator_inverse
      );
      failure_count += 1;
    }

    for (size_t j = 0; j < g->case_count; ++j) {
      const case_t *c = &cases[g->case_start + j];
      kernel_out_t out = {0, 0, 0, 0, 0, 0, 0};
      op_counts_t counts = {0, 0, 0, 0, 0};
      int case_failed = 0;
      if (c->reused) {
        reused_case_count += 1;
      }

      if (!shared_denominator_candidate_kernel(g, &state, c, &out, &counts)) {
        fprintf(stderr, "kernel failed source=%s event=%s\n", c->source_name, c->event_key);
        case_failed = 1;
        failure_count += 1;
      }

      total_counts.add += counts.add;
      total_counts.sub += counts.sub;
      total_counts.mul += counts.mul;
      total_counts.neg += counts.neg;
      total_counts.inv += counts.inv;

      if (
        out.slope_numerator == c->expected_slope_numerator &&
        out.slope_denominator == g->expected_slope_denominator &&
        out.slope_denominator_inverse == g->expected_slope_denominator_inverse &&
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
  }

  uint64_t total_ops = total_counts.add + total_counts.sub + total_counts.mul + total_counts.neg + total_counts.inv;
  printf(
    "{"
    "\"group_count\":%zu,"
    "\"case_count\":%zu,"
    "\"verified_count\":%zu,"
    "\"failure_count\":%zu,"
    "\"shared_prepare_match_count\":%zu,"
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
    group_count,
    case_count,
    verified_count,
    failure_count,
    shared_prepare_match_count,
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
