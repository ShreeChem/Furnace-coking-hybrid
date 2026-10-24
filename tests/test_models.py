"""
Checks for Step 4 (fast: uses the cached learning curves in data/derived).
Run:  python3 tests/test_models.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import models as M
fails = 0

def check(name, ok):
    global fails
    fails += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name}")

# 1. the one-standard-error rule picks the simplest option within 1 SE of the best
errs = [np.array([5.0, 5.2, 4.8, 5.1]), np.array([4.5, 5.3, 4.6, 5.2]), np.array([1.0, 1.1, 0.9, 1.0])]
check("1-SE rule keeps the simplest option when the others are not clearly better", M.one_se_choice(errs[:2]) == 0)
check("1-SE rule takes a complex option when it is clearly better", M.one_se_choice(errs) == 2)

# 2. learning curves exist and show the expected sparse-data behaviour
for q in "abc":
    lc = M.learning_curve_cached(q).groupby(["n_train", "model"]).mae.mean().unstack()
    check(f"({q}) with 2 campaigns, tuned physics beats ML-only by a wide margin",
          lc.loc[2, "physics"] < 0.8 * min(lc.loc[2, "ml_gbm"], lc.loc[2, "ml_ridge"]))
    check(f"({q}) cautious hybrids are never more than 1.5 days worse than physics",
          (lc[["hybrid_param", "hybrid_residual"]].max(axis=1) - lc["physics"]).max() < 1.5)
    check(f"({q}) ML-only improves with more data", lc.loc[30, "ml_gbm"] < lc.loc[2, "ml_gbm"])

print("\nALL PASSED" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
