"""
Checks for the hidden-truth plant and its synthetic data.
Run after `python3 plant_truth.py`:   python3 tests/test_truth.py
"""
import csv, glob, os, sys
from collections import Counter
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
fails = 0

def check(name, ok):
    global fails
    fails += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name}")

# 1. leakage guard: model code must never touch the hidden truth
model_files = [f for f in glob.glob(os.path.join(ROOT, "*.py")) if os.path.basename(f) != "plant_truth.py"]
PATTERNS = ["plant_truth", '"hidden"', "'hidden'", "data/hidden"]
leaks = [os.path.basename(f) for f in model_files if any(p in open(f).read() for p in PATTERNS)]
check(f"no model file imports plant_truth or reads data/hidden ({', '.join(leaks) or 'none'})", not leaks)

# 2. datasets exist and look like a real furnace history
for name, n in [("history", 30), ("test", 20)]:
    runs = list(csv.DictReader(open(os.path.join(ROOT, "data", name, "runs.csv"))))
    rl = [float(r["run_length_days"]) for r in runs]
    check(f"{name}: {n} campaigns, run lengths {min(rl):.0f}-{max(rl):.0f} d within 20-150 d", len(runs) == n and 20 < min(rl) and max(rl) < 150)
    pyro = list(csv.DictReader(open(os.path.join(ROOT, "data", name, "pyrometer_weekly.csv"))))
    daily = list(csv.DictReader(open(os.path.join(ROOT, "data", name, "dcs_daily.csv"))))
    check(f"{name}: pyrometer is sparse (~1 reading per 7 DCS days: {len(daily)/len(pyro):.1f})", 6 < len(daily) / len(pyro) < 8)
    per_run = Counter(r["run_id"] for r in daily)
    check(f"{name}: every campaign has DCS data up to its decoke day",
          all(abs(per_run[r["run_id"]] - float(r["run_length_days"])) <= 2 for r in runs))

# 3. truth ends each run at the TMT limit
truth = list(csv.DictReader(open(os.path.join(ROOT, "data", "hidden", "test_true_daily.csv"))))
last = {}
for t in truth:
    last[t["run_id"]] = float(t["tmt_max"])
check("true max TMT reaches the 1100 C limit on the last day of every campaign", all(v >= 1100 for v in last.values()))

print("\nALL PASSED" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
