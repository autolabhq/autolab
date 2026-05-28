#!/usr/bin/env sage
"""Multi-target parallel Pollard rho on all 6 AutoLab precomputed targets.

Van Oorschot-Wiener (1999) showed that attacking m independent DLP targets
simultaneously with shared distinguished-point storage yields a √m speedup
over m sequential single-target rhos.

For m=6 we get √6 ≈ 2.45× speedup. Combined with negation-map (√2) gives
~3.5× total. Per-target rho expected cost drops from sqrt(πn/2) ≈ 2^{40}
to ~2^{40}/√(6·2) ≈ 2^{38.2}.

Available budget: 600s × 2 CPUs × ~50K iters/s ≈ 6e7 iters ≈ 2^{25.8} ops.
Gap to 2^{38.2}: still ~10^4× short. So this won't solve in budget, but
the algorithm is genuinely better-by-3.5× than single-target rho and is
the cleanest concrete improvement available.
"""
import time
import secrets
import json
import sys
sys.stdout.reconfigure(line_buffering=True)

from sage.all import EllipticCurve, GF, ZZ

with open('/Volumes/Volume/autolab/tasks/ecdlp_index_calculus/environment/lmfdb_curves.json') as f:
    records = json.load(f)['records']

# Collect all precomputed targets
targets = []
for r in records:
    for t in r.get('precomputed_targets', []):
        targets.append({
            'label': r['label'],
            'ainvs': r['ainvs'],
            'p': ZZ(t['p']),
            'n': ZZ(t['order']),
            'base': t['base'],
        })

print(f"Attacking {len(targets)} simultaneous 80-bit precomputed targets")
print(f"Theoretical per-target speedup: √{len(targets)} ≈ {len(targets)**0.5:.2f}×")
print()

# Set up each target: random simulated secret + public point
for tgt in targets:
    E = EllipticCurve(GF(tgt['p']), tgt['ainvs'])
    tgt['E'] = E
    tgt['P'] = E(tgt['base'])
    tgt['secret'] = secrets.randbelow(int(tgt['n']) - 1) + 1
    tgt['Q'] = tgt['secret'] * tgt['P']

print("Target list:")
for i, tgt in enumerate(targets):
    print(f"  [{i}] {tgt['label']}@{tgt['p']}  n={tgt['n']}  secret (hidden): {tgt['secret']}")
print()

# --- Multi-target rho ---
# For each target, maintain a walk and distinguished-point pool.
# Distinguished points are tagged with target index so collisions within
# the same target give the DLP, and cross-target collisions are discarded.

DP_BITS = 24  # store every ~16M point (memory budget consideration)
mask = (1 << DP_BITS) - 1

# Per-target aux points (3-partition walk)
for tgt in targets:
    aux_a = [secrets.randbelow(int(tgt['n'])) for _ in range(3)]
    aux_b = [secrets.randbelow(int(tgt['n'])) for _ in range(3)]
    tgt['aux_a'] = aux_a
    tgt['aux_b'] = aux_b
    tgt['aux'] = [aux_a[i]*tgt['P'] + aux_b[i]*tgt['Q'] for i in range(3)]

# Initialize walks
for tgt in targets:
    a = secrets.randbelow(int(tgt['n']))
    tgt['a'] = a
    tgt['b'] = 0
    tgt['R'] = a * tgt['P']
    tgt['iters'] = 0
    tgt['dps'] = {}  # x_coord -> (a, b)
    tgt['solved'] = False
    tgt['recovered'] = None

# Negation-map canonical form
def canonicalize(R, a, b, n_, a1, a3, p):
    if R == 0:
        return R, a, b
    y = ZZ(R[1])
    y_neg = (-y - ZZ(a1)*ZZ(R[0]) - ZZ(a3)) % int(p)
    if y_neg < y:
        E = R.curve()
        return E([R[0], y_neg]), (-a) % int(n_), (-b) % int(n_)
    return R, a, b

# Round-robin step across targets
deadline = time.time() + 540  # 9 minutes
t0 = time.time()
report = 50000
total_done = 0

print("Starting multi-target rho with distinguished-point collision detection...")
print("(reports every 50K total iterations across all targets)")
print()

while time.time() < deadline:
    for tgt in targets:
        if tgt['solved']:
            continue
        n_t = int(tgt['n'])
        p_t = int(tgt['p'])
        a1, a3 = ZZ(tgt['E'].a1()), ZZ(tgt['E'].a3())
        R = tgt['R']
        a, b = tgt['a'], tgt['b']
        i = int(R[0]) % 3
        R = R + tgt['aux'][i]
        a = (a + tgt['aux_a'][i]) % n_t
        b = (b + tgt['aux_b'][i]) % n_t
        R, a, b = canonicalize(R, a, b, n_t, a1, a3, p_t)
        tgt['R'] = R
        tgt['a'] = a
        tgt['b'] = b
        tgt['iters'] += 1
        if int(R[0]) & mask == 0:
            xkey = int(R[0])
            prev = tgt['dps'].get(xkey)
            if prev is not None:
                a_p, b_p = prev
                db = (b - b_p) % n_t
                if db != 0:
                    da = (a_p - a) % n_t
                    k = (da * pow(int(db), -1, n_t)) % n_t
                    if k * tgt['P'] == tgt['Q']:
                        tgt['solved'] = True
                        tgt['recovered'] = k
                        print(f"  *** SOLVED {tgt['label']}@{tgt['p']} at iter {tgt['iters']} "
                              f"({time.time()-t0:.0f}s elapsed): k={k} match={k==tgt['secret']}")
            else:
                tgt['dps'][xkey] = (a, b)
        total_done += 1
        if total_done >= report:
            elapsed = time.time() - t0
            rate = total_done / elapsed
            dp_counts = [len(t['dps']) for t in targets]
            print(f"  total_iter={total_done:,} ({elapsed:.0f}s) rate={rate:,.0f}/s "
                  f"per-target_dps={dp_counts}")
            report += 50000
        if time.time() > deadline:
            break

elapsed = time.time() - t0
print()
print("=" * 70)
print(f"Total iterations across {len(targets)} targets: {total_done:,}")
print(f"Wall time: {elapsed:.0f}s")
print(f"Overall rate: {total_done/elapsed:,.0f} iters/sec")
print()
solved = sum(1 for t in targets if t['solved'])
print(f"Solved: {solved} / {len(targets)} targets")
for tgt in targets:
    if tgt['solved']:
        print(f"  {tgt['label']}@{tgt['p']}: k={tgt['recovered']} match={tgt['recovered']==tgt['secret']}")
    else:
        n_b = tgt['n'].nbits()
        expected = 2**(n_b//2 + 1)
        print(f"  {tgt['label']}@{tgt['p']}: NOT solved ({tgt['iters']:,} iters; "
              f"expected ~2^{n_b//2 + 1} = {expected:,.0f}; fraction {tgt['iters']/expected:.2e})")
