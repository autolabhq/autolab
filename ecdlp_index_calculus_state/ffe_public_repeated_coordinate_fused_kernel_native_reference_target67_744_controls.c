#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef enum {
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_NEG,
  OP_INV
} op_t;

typedef struct {
  op_t op;
  uint64_t a;
  uint64_t b;
  uint64_t expected;
  int out_kind;
} instr_t;

typedef struct {
  const char *source_name;
  const char *event_key;
  int fanout;
  int reused;
  uint64_t p;
  uint64_t expected_x;
  uint64_t expected_y;
  const instr_t *instructions;
  size_t instruction_count;
} case_t;

static uint64_t mod_add(uint64_t a, uint64_t b, uint64_t p) {
  return (a % p + b % p) % p;
}

static uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t p) {
  return (a % p + p - (b % p)) % p;
}

static uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t p) {
  return ((a % p) * (b % p)) % p;
}

static uint64_t mod_neg(uint64_t a, uint64_t p) {
  return (p - (a % p)) % p;
}

static uint64_t mod_inv(uint64_t a, uint64_t p, int *ok) {
  int64_t t = 0;
  int64_t new_t = 1;
  int64_t r = (int64_t)p;
  int64_t new_r = (int64_t)(a % p);

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

static const instr_t instr_0[] = {
  {OP_SUB, 8016ULL, 5888ULL, 2128ULL, 0},
  {OP_SUB, 7537ULL, 4979ULL, 2558ULL, 0},
  {OP_INV, 2558ULL, 0ULL, 6741ULL, 0},
  {OP_MUL, 2128ULL, 6741ULL, 3059ULL, 0},
  {OP_MUL, 3059ULL, 4979ULL, 6702ULL, 0},
  {OP_SUB, 5888ULL, 6702ULL, 8989ULL, 0},
  {OP_MUL, 3059ULL, 3059ULL, 5419ULL, 0},
  {OP_MUL, 0ULL, 3059ULL, 0ULL, 0},
  {OP_ADD, 5419ULL, 0ULL, 5419ULL, 0},
  {OP_SUB, 5419ULL, 1ULL, 5418ULL, 0},
  {OP_SUB, 5418ULL, 4979ULL, 439ULL, 0},
  {OP_SUB, 439ULL, 7537ULL, 2705ULL, 1},
  {OP_ADD, 3059ULL, 0ULL, 3059ULL, 0},
  {OP_MUL, 3059ULL, 2705ULL, 863ULL, 0},
  {OP_NEG, 863ULL, 0ULL, 8940ULL, 0},
  {OP_SUB, 8940ULL, 8989ULL, 9754ULL, 0},
  {OP_SUB, 9754ULL, 1ULL, 9753ULL, 2},
};

static const instr_t instr_1[] = {
  {OP_SUB, 8016ULL, 5888ULL, 2128ULL, 0},
  {OP_SUB, 7537ULL, 4979ULL, 2558ULL, 0},
  {OP_INV, 2558ULL, 0ULL, 6741ULL, 0},
  {OP_MUL, 2128ULL, 6741ULL, 3059ULL, 0},
  {OP_MUL, 3059ULL, 4979ULL, 6702ULL, 0},
  {OP_SUB, 5888ULL, 6702ULL, 8989ULL, 0},
  {OP_MUL, 3059ULL, 3059ULL, 5419ULL, 0},
  {OP_MUL, 0ULL, 3059ULL, 0ULL, 0},
  {OP_ADD, 5419ULL, 0ULL, 5419ULL, 0},
  {OP_SUB, 5419ULL, 1ULL, 5418ULL, 0},
  {OP_SUB, 5418ULL, 4979ULL, 439ULL, 0},
  {OP_SUB, 439ULL, 7537ULL, 2705ULL, 1},
  {OP_ADD, 3059ULL, 0ULL, 3059ULL, 0},
  {OP_MUL, 3059ULL, 2705ULL, 863ULL, 0},
  {OP_NEG, 863ULL, 0ULL, 8940ULL, 0},
  {OP_SUB, 8940ULL, 8989ULL, 9754ULL, 0},
  {OP_SUB, 9754ULL, 1ULL, 9753ULL, 2},
};

static const instr_t instr_2[] = {
  {OP_SUB, 8016ULL, 5888ULL, 2128ULL, 0},
  {OP_SUB, 7537ULL, 4979ULL, 2558ULL, 0},
  {OP_INV, 2558ULL, 0ULL, 6741ULL, 0},
  {OP_MUL, 2128ULL, 6741ULL, 3059ULL, 0},
  {OP_MUL, 3059ULL, 4979ULL, 6702ULL, 0},
  {OP_SUB, 5888ULL, 6702ULL, 8989ULL, 0},
  {OP_MUL, 3059ULL, 3059ULL, 5419ULL, 0},
  {OP_MUL, 0ULL, 3059ULL, 0ULL, 0},
  {OP_ADD, 5419ULL, 0ULL, 5419ULL, 0},
  {OP_SUB, 5419ULL, 1ULL, 5418ULL, 0},
  {OP_SUB, 5418ULL, 4979ULL, 439ULL, 0},
  {OP_SUB, 439ULL, 7537ULL, 2705ULL, 1},
  {OP_ADD, 3059ULL, 0ULL, 3059ULL, 0},
  {OP_MUL, 3059ULL, 2705ULL, 863ULL, 0},
  {OP_NEG, 863ULL, 0ULL, 8940ULL, 0},
  {OP_SUB, 8940ULL, 8989ULL, 9754ULL, 0},
  {OP_SUB, 9754ULL, 1ULL, 9753ULL, 2},
};

static const instr_t instr_3[] = {
  {OP_SUB, 8016ULL, 5888ULL, 2128ULL, 0},
  {OP_SUB, 7537ULL, 4979ULL, 2558ULL, 0},
  {OP_INV, 2558ULL, 0ULL, 6741ULL, 0},
  {OP_MUL, 2128ULL, 6741ULL, 3059ULL, 0},
  {OP_MUL, 3059ULL, 4979ULL, 6702ULL, 0},
  {OP_SUB, 5888ULL, 6702ULL, 8989ULL, 0},
  {OP_MUL, 3059ULL, 3059ULL, 5419ULL, 0},
  {OP_MUL, 0ULL, 3059ULL, 0ULL, 0},
  {OP_ADD, 5419ULL, 0ULL, 5419ULL, 0},
  {OP_SUB, 5419ULL, 1ULL, 5418ULL, 0},
  {OP_SUB, 5418ULL, 4979ULL, 439ULL, 0},
  {OP_SUB, 439ULL, 7537ULL, 2705ULL, 1},
  {OP_ADD, 3059ULL, 0ULL, 3059ULL, 0},
  {OP_MUL, 3059ULL, 2705ULL, 863ULL, 0},
  {OP_NEG, 863ULL, 0ULL, 8940ULL, 0},
  {OP_SUB, 8940ULL, 8989ULL, 9754ULL, 0},
  {OP_SUB, 9754ULL, 1ULL, 9753ULL, 2},
};

static const instr_t instr_4[] = {
  {OP_SUB, 8016ULL, 3914ULL, 4102ULL, 0},
  {OP_SUB, 7537ULL, 4979ULL, 2558ULL, 0},
  {OP_INV, 2558ULL, 0ULL, 6741ULL, 0},
  {OP_MUL, 4102ULL, 6741ULL, 7122ULL, 0},
  {OP_MUL, 7122ULL, 4979ULL, 2987ULL, 0},
  {OP_SUB, 3914ULL, 2987ULL, 927ULL, 0},
  {OP_MUL, 7122ULL, 7122ULL, 2162ULL, 0},
  {OP_MUL, 0ULL, 7122ULL, 0ULL, 0},
  {OP_ADD, 2162ULL, 0ULL, 2162ULL, 0},
  {OP_SUB, 2162ULL, 1ULL, 2161ULL, 0},
  {OP_SUB, 2161ULL, 4979ULL, 6985ULL, 0},
  {OP_SUB, 6985ULL, 7537ULL, 9251ULL, 1},
  {OP_ADD, 7122ULL, 0ULL, 7122ULL, 0},
  {OP_MUL, 7122ULL, 9251ULL, 9462ULL, 0},
  {OP_NEG, 9462ULL, 0ULL, 341ULL, 0},
  {OP_SUB, 341ULL, 927ULL, 9217ULL, 0},
  {OP_SUB, 9217ULL, 1ULL, 9216ULL, 2},
};

static const instr_t instr_5[] = {
  {OP_SUB, 8016ULL, 3914ULL, 4102ULL, 0},
  {OP_SUB, 7537ULL, 4979ULL, 2558ULL, 0},
  {OP_INV, 2558ULL, 0ULL, 6741ULL, 0},
  {OP_MUL, 4102ULL, 6741ULL, 7122ULL, 0},
  {OP_MUL, 7122ULL, 4979ULL, 2987ULL, 0},
  {OP_SUB, 3914ULL, 2987ULL, 927ULL, 0},
  {OP_MUL, 7122ULL, 7122ULL, 2162ULL, 0},
  {OP_MUL, 0ULL, 7122ULL, 0ULL, 0},
  {OP_ADD, 2162ULL, 0ULL, 2162ULL, 0},
  {OP_SUB, 2162ULL, 1ULL, 2161ULL, 0},
  {OP_SUB, 2161ULL, 4979ULL, 6985ULL, 0},
  {OP_SUB, 6985ULL, 7537ULL, 9251ULL, 1},
  {OP_ADD, 7122ULL, 0ULL, 7122ULL, 0},
  {OP_MUL, 7122ULL, 9251ULL, 9462ULL, 0},
  {OP_NEG, 9462ULL, 0ULL, 341ULL, 0},
  {OP_SUB, 341ULL, 927ULL, 9217ULL, 0},
  {OP_SUB, 9217ULL, 1ULL, 9216ULL, 2},
};

static const instr_t instr_6[] = {
  {OP_SUB, 8016ULL, 3914ULL, 4102ULL, 0},
  {OP_SUB, 7537ULL, 4979ULL, 2558ULL, 0},
  {OP_INV, 2558ULL, 0ULL, 6741ULL, 0},
  {OP_MUL, 4102ULL, 6741ULL, 7122ULL, 0},
  {OP_MUL, 7122ULL, 4979ULL, 2987ULL, 0},
  {OP_SUB, 3914ULL, 2987ULL, 927ULL, 0},
  {OP_MUL, 7122ULL, 7122ULL, 2162ULL, 0},
  {OP_MUL, 0ULL, 7122ULL, 0ULL, 0},
  {OP_ADD, 2162ULL, 0ULL, 2162ULL, 0},
  {OP_SUB, 2162ULL, 1ULL, 2161ULL, 0},
  {OP_SUB, 2161ULL, 4979ULL, 6985ULL, 0},
  {OP_SUB, 6985ULL, 7537ULL, 9251ULL, 1},
  {OP_ADD, 7122ULL, 0ULL, 7122ULL, 0},
  {OP_MUL, 7122ULL, 9251ULL, 9462ULL, 0},
  {OP_NEG, 9462ULL, 0ULL, 341ULL, 0},
  {OP_SUB, 341ULL, 927ULL, 9217ULL, 0},
  {OP_SUB, 9217ULL, 1ULL, 9216ULL, 2},
};

static const instr_t instr_7[] = {
  {OP_SUB, 8016ULL, 3914ULL, 4102ULL, 0},
  {OP_SUB, 7537ULL, 4979ULL, 2558ULL, 0},
  {OP_INV, 2558ULL, 0ULL, 6741ULL, 0},
  {OP_MUL, 4102ULL, 6741ULL, 7122ULL, 0},
  {OP_MUL, 7122ULL, 4979ULL, 2987ULL, 0},
  {OP_SUB, 3914ULL, 2987ULL, 927ULL, 0},
  {OP_MUL, 7122ULL, 7122ULL, 2162ULL, 0},
  {OP_MUL, 0ULL, 7122ULL, 0ULL, 0},
  {OP_ADD, 2162ULL, 0ULL, 2162ULL, 0},
  {OP_SUB, 2162ULL, 1ULL, 2161ULL, 0},
  {OP_SUB, 2161ULL, 4979ULL, 6985ULL, 0},
  {OP_SUB, 6985ULL, 7537ULL, 9251ULL, 1},
  {OP_ADD, 7122ULL, 0ULL, 7122ULL, 0},
  {OP_MUL, 7122ULL, 9251ULL, 9462ULL, 0},
  {OP_NEG, 9462ULL, 0ULL, 341ULL, 0},
  {OP_SUB, 341ULL, 927ULL, 9217ULL, 0},
  {OP_SUB, 9217ULL, 1ULL, 9216ULL, 2},
};

static const instr_t instr_8[] = {
  {OP_SUB, 3749ULL, 3914ULL, 9638ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 9638ULL, 7833ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 4979ULL, 7468ULL, 0},
  {OP_SUB, 3914ULL, 7468ULL, 6249ULL, 0},
  {OP_MUL, 1551ULL, 1551ULL, 3866ULL, 0},
  {OP_MUL, 0ULL, 1551ULL, 0ULL, 0},
  {OP_ADD, 3866ULL, 0ULL, 3866ULL, 0},
  {OP_SUB, 3866ULL, 1ULL, 3865ULL, 0},
  {OP_SUB, 3865ULL, 4979ULL, 8689ULL, 0},
  {OP_SUB, 8689ULL, 2476ULL, 6213ULL, 1},
  {OP_ADD, 1551ULL, 0ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 6213ULL, 14ULL, 0},
  {OP_NEG, 14ULL, 0ULL, 9789ULL, 0},
  {OP_SUB, 9789ULL, 6249ULL, 3540ULL, 0},
  {OP_SUB, 3540ULL, 1ULL, 3539ULL, 2},
};

static const instr_t instr_9[] = {
  {OP_SUB, 3749ULL, 3914ULL, 9638ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 9638ULL, 7833ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 4979ULL, 7468ULL, 0},
  {OP_SUB, 3914ULL, 7468ULL, 6249ULL, 0},
  {OP_MUL, 1551ULL, 1551ULL, 3866ULL, 0},
  {OP_MUL, 0ULL, 1551ULL, 0ULL, 0},
  {OP_ADD, 3866ULL, 0ULL, 3866ULL, 0},
  {OP_SUB, 3866ULL, 1ULL, 3865ULL, 0},
  {OP_SUB, 3865ULL, 4979ULL, 8689ULL, 0},
  {OP_SUB, 8689ULL, 2476ULL, 6213ULL, 1},
  {OP_ADD, 1551ULL, 0ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 6213ULL, 14ULL, 0},
  {OP_NEG, 14ULL, 0ULL, 9789ULL, 0},
  {OP_SUB, 9789ULL, 6249ULL, 3540ULL, 0},
  {OP_SUB, 3540ULL, 1ULL, 3539ULL, 2},
};

static const instr_t instr_10[] = {
  {OP_SUB, 3749ULL, 3914ULL, 9638ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 9638ULL, 7833ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 4979ULL, 7468ULL, 0},
  {OP_SUB, 3914ULL, 7468ULL, 6249ULL, 0},
  {OP_MUL, 1551ULL, 1551ULL, 3866ULL, 0},
  {OP_MUL, 0ULL, 1551ULL, 0ULL, 0},
  {OP_ADD, 3866ULL, 0ULL, 3866ULL, 0},
  {OP_SUB, 3866ULL, 1ULL, 3865ULL, 0},
  {OP_SUB, 3865ULL, 4979ULL, 8689ULL, 0},
  {OP_SUB, 8689ULL, 2476ULL, 6213ULL, 1},
  {OP_ADD, 1551ULL, 0ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 6213ULL, 14ULL, 0},
  {OP_NEG, 14ULL, 0ULL, 9789ULL, 0},
  {OP_SUB, 9789ULL, 6249ULL, 3540ULL, 0},
  {OP_SUB, 3540ULL, 1ULL, 3539ULL, 2},
};

static const instr_t instr_11[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_12[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_13[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_14[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_15[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_16[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_17[] = {
  {OP_SUB, 3749ULL, 3914ULL, 9638ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 9638ULL, 7833ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 4979ULL, 7468ULL, 0},
  {OP_SUB, 3914ULL, 7468ULL, 6249ULL, 0},
  {OP_MUL, 1551ULL, 1551ULL, 3866ULL, 0},
  {OP_MUL, 0ULL, 1551ULL, 0ULL, 0},
  {OP_ADD, 3866ULL, 0ULL, 3866ULL, 0},
  {OP_SUB, 3866ULL, 1ULL, 3865ULL, 0},
  {OP_SUB, 3865ULL, 4979ULL, 8689ULL, 0},
  {OP_SUB, 8689ULL, 2476ULL, 6213ULL, 1},
  {OP_ADD, 1551ULL, 0ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 6213ULL, 14ULL, 0},
  {OP_NEG, 14ULL, 0ULL, 9789ULL, 0},
  {OP_SUB, 9789ULL, 6249ULL, 3540ULL, 0},
  {OP_SUB, 3540ULL, 1ULL, 3539ULL, 2},
};

static const instr_t instr_18[] = {
  {OP_SUB, 3749ULL, 3914ULL, 9638ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 9638ULL, 7833ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 4979ULL, 7468ULL, 0},
  {OP_SUB, 3914ULL, 7468ULL, 6249ULL, 0},
  {OP_MUL, 1551ULL, 1551ULL, 3866ULL, 0},
  {OP_MUL, 0ULL, 1551ULL, 0ULL, 0},
  {OP_ADD, 3866ULL, 0ULL, 3866ULL, 0},
  {OP_SUB, 3866ULL, 1ULL, 3865ULL, 0},
  {OP_SUB, 3865ULL, 4979ULL, 8689ULL, 0},
  {OP_SUB, 8689ULL, 2476ULL, 6213ULL, 1},
  {OP_ADD, 1551ULL, 0ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 6213ULL, 14ULL, 0},
  {OP_NEG, 14ULL, 0ULL, 9789ULL, 0},
  {OP_SUB, 9789ULL, 6249ULL, 3540ULL, 0},
  {OP_SUB, 3540ULL, 1ULL, 3539ULL, 2},
};

static const instr_t instr_19[] = {
  {OP_SUB, 3749ULL, 3914ULL, 9638ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 9638ULL, 7833ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 4979ULL, 7468ULL, 0},
  {OP_SUB, 3914ULL, 7468ULL, 6249ULL, 0},
  {OP_MUL, 1551ULL, 1551ULL, 3866ULL, 0},
  {OP_MUL, 0ULL, 1551ULL, 0ULL, 0},
  {OP_ADD, 3866ULL, 0ULL, 3866ULL, 0},
  {OP_SUB, 3866ULL, 1ULL, 3865ULL, 0},
  {OP_SUB, 3865ULL, 4979ULL, 8689ULL, 0},
  {OP_SUB, 8689ULL, 2476ULL, 6213ULL, 1},
  {OP_ADD, 1551ULL, 0ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 6213ULL, 14ULL, 0},
  {OP_NEG, 14ULL, 0ULL, 9789ULL, 0},
  {OP_SUB, 9789ULL, 6249ULL, 3540ULL, 0},
  {OP_SUB, 3540ULL, 1ULL, 3539ULL, 2},
};

static const instr_t instr_20[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_21[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_22[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_23[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_24[] = {
  {OP_SUB, 3749ULL, 3914ULL, 9638ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 9638ULL, 7833ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 4979ULL, 7468ULL, 0},
  {OP_SUB, 3914ULL, 7468ULL, 6249ULL, 0},
  {OP_MUL, 1551ULL, 1551ULL, 3866ULL, 0},
  {OP_MUL, 0ULL, 1551ULL, 0ULL, 0},
  {OP_ADD, 3866ULL, 0ULL, 3866ULL, 0},
  {OP_SUB, 3866ULL, 1ULL, 3865ULL, 0},
  {OP_SUB, 3865ULL, 4979ULL, 8689ULL, 0},
  {OP_SUB, 8689ULL, 2476ULL, 6213ULL, 1},
  {OP_ADD, 1551ULL, 0ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 6213ULL, 14ULL, 0},
  {OP_NEG, 14ULL, 0ULL, 9789ULL, 0},
  {OP_SUB, 9789ULL, 6249ULL, 3540ULL, 0},
  {OP_SUB, 3540ULL, 1ULL, 3539ULL, 2},
};

static const instr_t instr_25[] = {
  {OP_SUB, 3749ULL, 3914ULL, 9638ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 9638ULL, 7833ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 4979ULL, 7468ULL, 0},
  {OP_SUB, 3914ULL, 7468ULL, 6249ULL, 0},
  {OP_MUL, 1551ULL, 1551ULL, 3866ULL, 0},
  {OP_MUL, 0ULL, 1551ULL, 0ULL, 0},
  {OP_ADD, 3866ULL, 0ULL, 3866ULL, 0},
  {OP_SUB, 3866ULL, 1ULL, 3865ULL, 0},
  {OP_SUB, 3865ULL, 4979ULL, 8689ULL, 0},
  {OP_SUB, 8689ULL, 2476ULL, 6213ULL, 1},
  {OP_ADD, 1551ULL, 0ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 6213ULL, 14ULL, 0},
  {OP_NEG, 14ULL, 0ULL, 9789ULL, 0},
  {OP_SUB, 9789ULL, 6249ULL, 3540ULL, 0},
  {OP_SUB, 3540ULL, 1ULL, 3539ULL, 2},
};

static const instr_t instr_26[] = {
  {OP_SUB, 3749ULL, 3914ULL, 9638ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 9638ULL, 7833ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 4979ULL, 7468ULL, 0},
  {OP_SUB, 3914ULL, 7468ULL, 6249ULL, 0},
  {OP_MUL, 1551ULL, 1551ULL, 3866ULL, 0},
  {OP_MUL, 0ULL, 1551ULL, 0ULL, 0},
  {OP_ADD, 3866ULL, 0ULL, 3866ULL, 0},
  {OP_SUB, 3866ULL, 1ULL, 3865ULL, 0},
  {OP_SUB, 3865ULL, 4979ULL, 8689ULL, 0},
  {OP_SUB, 8689ULL, 2476ULL, 6213ULL, 1},
  {OP_ADD, 1551ULL, 0ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 6213ULL, 14ULL, 0},
  {OP_NEG, 14ULL, 0ULL, 9789ULL, 0},
  {OP_SUB, 9789ULL, 6249ULL, 3540ULL, 0},
  {OP_SUB, 3540ULL, 1ULL, 3539ULL, 2},
};

static const instr_t instr_27[] = {
  {OP_SUB, 3749ULL, 3914ULL, 9638ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 9638ULL, 7833ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 4979ULL, 7468ULL, 0},
  {OP_SUB, 3914ULL, 7468ULL, 6249ULL, 0},
  {OP_MUL, 1551ULL, 1551ULL, 3866ULL, 0},
  {OP_MUL, 0ULL, 1551ULL, 0ULL, 0},
  {OP_ADD, 3866ULL, 0ULL, 3866ULL, 0},
  {OP_SUB, 3866ULL, 1ULL, 3865ULL, 0},
  {OP_SUB, 3865ULL, 4979ULL, 8689ULL, 0},
  {OP_SUB, 8689ULL, 2476ULL, 6213ULL, 1},
  {OP_ADD, 1551ULL, 0ULL, 1551ULL, 0},
  {OP_MUL, 1551ULL, 6213ULL, 14ULL, 0},
  {OP_NEG, 14ULL, 0ULL, 9789ULL, 0},
  {OP_SUB, 9789ULL, 6249ULL, 3540ULL, 0},
  {OP_SUB, 3540ULL, 1ULL, 3539ULL, 2},
};

static const instr_t instr_28[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_29[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_30[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const instr_t instr_31[] = {
  {OP_SUB, 3749ULL, 5888ULL, 7664ULL, 0},
  {OP_SUB, 2476ULL, 4979ULL, 7300ULL, 0},
  {OP_INV, 7300ULL, 0ULL, 7833ULL, 0},
  {OP_MUL, 7664ULL, 7833ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 4979ULL, 4486ULL, 0},
  {OP_SUB, 5888ULL, 4486ULL, 1402ULL, 0},
  {OP_MUL, 8343ULL, 8343ULL, 4349ULL, 0},
  {OP_MUL, 0ULL, 8343ULL, 0ULL, 0},
  {OP_ADD, 4349ULL, 0ULL, 4349ULL, 0},
  {OP_SUB, 4349ULL, 1ULL, 4348ULL, 0},
  {OP_SUB, 4348ULL, 4979ULL, 9172ULL, 0},
  {OP_SUB, 9172ULL, 2476ULL, 6696ULL, 1},
  {OP_ADD, 8343ULL, 0ULL, 8343ULL, 0},
  {OP_MUL, 8343ULL, 6696ULL, 7234ULL, 0},
  {OP_NEG, 7234ULL, 0ULL, 2569ULL, 0},
  {OP_SUB, 2569ULL, 1402ULL, 1167ULL, 0},
  {OP_SUB, 1167ULL, 1ULL, 1166ULL, 2},
};

static const case_t cases[] = {
  {"coord1114_744", "12:17:218", 1, 0, 9803ULL, 2705ULL, 9753ULL, instr_0, sizeof(instr_0) / sizeof(instr_0[0])},
  {"coord1114_744", "12:17:469", 2, 1, 9803ULL, 2705ULL, 9753ULL, instr_1, sizeof(instr_1) / sizeof(instr_1[0])},
  {"coord1114_744", "12:30:218", 1, 0, 9803ULL, 2705ULL, 9753ULL, instr_2, sizeof(instr_2) / sizeof(instr_2[0])},
  {"coord1114_744", "12:30:469", 2, 1, 9803ULL, 2705ULL, 9753ULL, instr_3, sizeof(instr_3) / sizeof(instr_3[0])},
  {"coord1114_744", "12:174:218", 1, 0, 9803ULL, 9251ULL, 9216ULL, instr_4, sizeof(instr_4) / sizeof(instr_4[0])},
  {"coord1114_744", "12:174:469", 2, 1, 9803ULL, 9251ULL, 9216ULL, instr_5, sizeof(instr_5) / sizeof(instr_5[0])},
  {"coord1114_744", "12:210:218", 1, 0, 9803ULL, 9251ULL, 9216ULL, instr_6, sizeof(instr_6) / sizeof(instr_6[0])},
  {"coord1114_744", "12:210:469", 2, 1, 9803ULL, 9251ULL, 9216ULL, instr_7, sizeof(instr_7) / sizeof(instr_7[0])},
  {"coord161_720", "2:1:149", 1, 0, 9803ULL, 6213ULL, 3539ULL, instr_8, sizeof(instr_8) / sizeof(instr_8[0])},
  {"coord161_720", "2:1:186", 1, 0, 9803ULL, 6213ULL, 3539ULL, instr_9, sizeof(instr_9) / sizeof(instr_9[0])},
  {"coord161_720", "2:1:296", 1, 0, 9803ULL, 6213ULL, 3539ULL, instr_10, sizeof(instr_10) / sizeof(instr_10[0])},
  {"coord161_720", "2:130:149", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_11, sizeof(instr_11) / sizeof(instr_11[0])},
  {"coord161_720", "2:130:186", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_12, sizeof(instr_12) / sizeof(instr_12[0])},
  {"coord161_720", "2:130:296", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_13, sizeof(instr_13) / sizeof(instr_13[0])},
  {"coord161_720", "2:144:149", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_14, sizeof(instr_14) / sizeof(instr_14[0])},
  {"coord161_720", "2:144:186", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_15, sizeof(instr_15) / sizeof(instr_15[0])},
  {"coord161_720", "2:144:296", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_16, sizeof(instr_16) / sizeof(instr_16[0])},
  {"coord161_720", "2:331:149", 1, 0, 9803ULL, 6213ULL, 3539ULL, instr_17, sizeof(instr_17) / sizeof(instr_17[0])},
  {"coord161_720", "2:331:186", 1, 0, 9803ULL, 6213ULL, 3539ULL, instr_18, sizeof(instr_18) / sizeof(instr_18[0])},
  {"coord161_720", "2:331:296", 1, 0, 9803ULL, 6213ULL, 3539ULL, instr_19, sizeof(instr_19) / sizeof(instr_19[0])},
  {"coord161_728", "2:4:204", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_20, sizeof(instr_20) / sizeof(instr_20[0])},
  {"coord161_728", "2:4:377", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_21, sizeof(instr_21) / sizeof(instr_21[0])},
  {"coord161_728", "2:4:429", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_22, sizeof(instr_22) / sizeof(instr_22[0])},
  {"coord161_728", "2:4:466", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_23, sizeof(instr_23) / sizeof(instr_23[0])},
  {"coord161_728", "2:202:204", 1, 0, 9803ULL, 6213ULL, 3539ULL, instr_24, sizeof(instr_24) / sizeof(instr_24[0])},
  {"coord161_728", "2:202:377", 1, 0, 9803ULL, 6213ULL, 3539ULL, instr_25, sizeof(instr_25) / sizeof(instr_25[0])},
  {"coord161_728", "2:202:429", 1, 0, 9803ULL, 6213ULL, 3539ULL, instr_26, sizeof(instr_26) / sizeof(instr_26[0])},
  {"coord161_728", "2:202:466", 1, 0, 9803ULL, 6213ULL, 3539ULL, instr_27, sizeof(instr_27) / sizeof(instr_27[0])},
  {"coord161_728", "2:226:204", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_28, sizeof(instr_28) / sizeof(instr_28[0])},
  {"coord161_728", "2:226:377", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_29, sizeof(instr_29) / sizeof(instr_29[0])},
  {"coord161_728", "2:226:429", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_30, sizeof(instr_30) / sizeof(instr_30[0])},
  {"coord161_728", "2:226:466", 1, 0, 9803ULL, 6696ULL, 1166ULL, instr_31, sizeof(instr_31) / sizeof(instr_31[0])},
};

int main(void) {
  size_t case_count = sizeof(cases) / sizeof(cases[0]);
  size_t verified_count = 0;
  size_t failure_count = 0;
  size_t instruction_count = 0;
  size_t result_match_count = 0;
  size_t reused_case_count = 0;
  size_t op_add = 0;
  size_t op_sub = 0;
  size_t op_mul = 0;
  size_t op_neg = 0;
  size_t op_inv = 0;

  for (size_t i = 0; i < case_count; ++i) {
    const case_t *c = &cases[i];
    uint64_t x3 = UINT64_MAX;
    uint64_t y3 = UINT64_MAX;
    int case_failed = 0;
    if (c->reused) {
      reused_case_count += 1;
    }

    for (size_t j = 0; j < c->instruction_count; ++j) {
      const instr_t *instr = &c->instructions[j];
      uint64_t value = 0;
      int ok = 1;
      switch (instr->op) {
        case OP_ADD:
          value = mod_add(instr->a, instr->b, c->p);
          op_add += 1;
          break;
        case OP_SUB:
          value = mod_sub(instr->a, instr->b, c->p);
          op_sub += 1;
          break;
        case OP_MUL:
          value = mod_mul(instr->a, instr->b, c->p);
          op_mul += 1;
          break;
        case OP_NEG:
          value = mod_neg(instr->a, c->p);
          op_neg += 1;
          break;
        case OP_INV:
          value = mod_inv(instr->a, c->p, &ok);
          op_inv += 1;
          break;
      }

      instruction_count += 1;
      if (!ok || value != instr->expected) {
        if (!case_failed) {
          fprintf(
            stderr,
            "case failure source=%s event=%s instr=%zu expected=%" PRIu64 " got=%" PRIu64 "\n",
            c->source_name,
            c->event_key,
            j,
            instr->expected,
            value
          );
        }
        case_failed = 1;
        failure_count += 1;
      }
      if (instr->out_kind == 1) {
        x3 = value;
      } else if (instr->out_kind == 2) {
        y3 = value;
      }
    }

    if (x3 == c->expected_x && y3 == c->expected_y) {
      result_match_count += 1;
    } else {
      if (!case_failed) {
        fprintf(
          stderr,
          "result failure source=%s event=%s expected=(%" PRIu64 ",%" PRIu64 ") got=(%" PRIu64 ",%" PRIu64 ")\n",
          c->source_name,
          c->event_key,
          c->expected_x,
          c->expected_y,
          x3,
          y3
        );
      }
      case_failed = 1;
      failure_count += 1;
    }

    if (!case_failed) {
      verified_count += 1;
    }
  }

  printf(
    "{"
    "\"case_count\":%zu,"
    "\"verified_count\":%zu,"
    "\"failure_count\":%zu,"
    "\"instruction_count\":%zu,"
    "\"result_match_count\":%zu,"
    "\"reused_case_count\":%zu,"
    "\"field_op_counts\":{"
    "\"add\":%zu,"
    "\"sub\":%zu,"
    "\"mul\":%zu,"
    "\"neg\":%zu,"
    "\"inv\":%zu,"
    "\"total\":%zu"
    "}"
    "}\n",
    case_count,
    verified_count,
    failure_count,
    instruction_count,
    result_match_count,
    reused_case_count,
    op_add,
    op_sub,
    op_mul,
    op_neg,
    op_inv,
    instruction_count
  );

  return failure_count == 0 ? 0 : 1;
}
