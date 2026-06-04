#ifndef SOLVE_H
#define SOLVE_H

/*
 * Compute the Levenshtein edit distance between string a (length la)
 * and string b (length lb).
 *
 * Both strings contain only lowercase ASCII letters a-z.
 * Benchmark calls use lengths in [16, 64]. Correctness tests may include
 * empty strings, so implementations must handle la and lb in [0, 64].
 *
 * Returns 0 if the strings are identical.
 */
int levenshtein(const char *a, int la, const char *b, int lb);

#endif /* SOLVE_H */
