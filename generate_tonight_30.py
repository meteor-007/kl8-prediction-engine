import itertools
import random
from collections import Counter
from calc_core import load_dataset, TimeSeries, DEFAULT_HISTORY_FILE
from calc_engine import run_calculus_engine
from calc_shrinkage import MorphologyFilterConfig, ShrinkageEngine

ds = load_dataset(DEFAULT_HISTORY_FILE)
ts = TimeSeries(ds)
res = run_calculus_engine(ds, ts)
scores = res['fusion_scores']
companion_dict = res.get('companion_dict', {})

pool_291 = [5, 7, 8, 9, 10, 11, 14, 17, 26, 30, 34, 36, 37, 39, 44, 50, 53, 55, 56, 58, 59, 60, 66, 68, 70, 73, 74, 75, 78, 80]
score_map = {n: float(scores[n-1]) for n in pool_291}

engine = ShrinkageEngine(pool_291, score_map)
candidates = engine.generate_filtered_candidates()

print(f"Total valid morphology candidates: {len(candidates)}")

# We want 30 tickets that:
# 1. 100% cover all 30 numbers (each number appears at least 3 times, average 5 times).
# 2. Maximum kinetic momentum + companion synergy.
# 3. High-density cluster capture.

def get_combo_score(c):
    # base momentum score
    base = sum(score_map[x] for x in c)
    # companion affinity
    comp = sum(1 for a, b in itertools.combinations(c, 2) if a in companion_dict and b in companion_dict[a])
    return base + comp * 0.12

# Pre-score candidates
scored = [(c, get_combo_score(c)) for c in candidates]
scored.sort(key=lambda x: x[1], reverse=True)

# Select 30 tickets using weighted coverage greedy
selected = []
counts = {n: 0 for n in pool_291}
target_budget = 30

# First pass: ensure every number is covered with high momentum
for n in sorted(pool_291, key=lambda x: score_map[x], reverse=True):
    if counts[n] < 2:
        for c, sc in scored:
            if n in c and c not in selected:
                # check overlap with existing <= 3
                if all(len(set(c) & set(s)) <= 3 for s in selected):
                    selected.append(c)
                    for x in c: counts[x] += 1
                    break
        if len(selected) >= target_budget:
            break

# Second pass: fill up to 30 tickets with best synergy and frequency balancing
while len(selected) < target_budget:
    best_cand = None
    best_val = -1e9
    for c, sc in scored:
        if c in selected:
            continue
        # overlap check
        if any(len(set(c) & set(s)) > 3 for s in selected):
            continue
        # value: score minus penalty for overused numbers
        freq_penalty = sum(counts[x] * 0.15 for x in c)
        val = sc - freq_penalty
        if val > best_val:
            best_val = val
            best_cand = c
    if best_cand:
        selected.append(best_cand)
        for x in best_cand: counts[x] += 1
    else:
        # relax overlap constraint to 4 if needed
        for c, sc in scored:
            if c not in selected:
                selected.append(c)
                for x in c: counts[x] += 1
                if len(selected) >= target_budget: break
        break

selected = selected[:target_budget]
print(f"Selected tickets: {len(selected)}")
covered = set(itertools.chain.from_iterable(selected))
print(f"Covered numbers: {len(covered)} / 30")
missing = set(pool_291) - covered
if missing:
    print("Missing:", missing)

print("\nFrequency of each number:")
for n in sorted(pool_291):
    print(f"{n:02d}: {counts[n]} times", end=", ")
print()

# Run Monte Carlo audit
audit = engine.run_monte_carlo_audit(selected, n_simulations=5000)
print("\nMonte Carlo 5000 simulations (on 9 hits):")
for k, v in audit.items():
    print(f"  {k}: {v:.2f}")

print("\n=== THE 30 TICKETS ===")
for idx, t in enumerate(selected, 1):
    print(f"[{idx:02d}] {' '.join(f'{x:02d}' for x in sorted(t))}")
