"""
Checks for Step 5 (economics). Fast, no model training.
Run:  python3 tests/test_value.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import value as V
fails = 0

def check(name, ok):
    global fails
    fails += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name}")

check(f"15 c/lb margin = {V.margin_eur_per_t(15):.0f} EUR/t (22.05 $/t per c/lb, 1.152 $/EUR)", abs(V.margin_eur_per_t(15) - 287) < 1)
check("furnace-limited: longer campaigns earn more per year",
      V.annual_profit(60) > V.annual_profit(40) > V.annual_profit(20))
check("spare furnace: only the decoke cost depends on run length",
      abs((V.annual_profit(60, dict(spare_furnace=True)) - V.annual_profit(30, dict(spare_furnace=True)))
          - V.ECON["decoke_cost"] * (365 / 32 - 365 / 62)) < 1)
check("no early decoke -> no money lost", V.value_of_early_days(np.array([60.0]), np.array([0.0])) == 0)
err = np.array([1.0, -2.0, 3.0, 0.5, -1.0])
check("normal margin = mean + 1.645 x SD", abs(V.safety_margin(err) - (err.mean() + 1.645 * err.std(ddof=1))) < 1e-9)
check("margin is never negative", V.safety_margin(np.array([-5.0, -6.0, -4.0])) == 0.0)
planned, early, late = V.plan_decoke(np.array([30.0, 30.0]), 5.0, np.array([28.0, 20.0]))
check("planning: decoke 5 days before the prediction; late only if the truth ends first",
      list(planned) == [25.0, 25.0] and list(early) == [3.0, 0.0] and list(late) == [False, True])
print("\nALL PASSED" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
