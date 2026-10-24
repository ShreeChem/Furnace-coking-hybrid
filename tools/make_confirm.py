"""Generates a fresh CONFIRMATION set (20 campaigns, new seed) used exactly once to check Step 4 conclusions."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plant_truth as pt
ROOT = os.path.join(os.path.dirname(__file__), "..")
A = json.load(open(os.path.join(ROOT, "data", "hidden", "truth_params.json")))["A_true"]
daily, pyro, runs, truth_rows = pt.generate(20, 303, A)
for name, rows in [("dcs_daily", daily), ("pyrometer_weekly", pyro), ("runs", runs)]:
    pt.write_csv(os.path.join(ROOT, "data", "confirm", f"{name}.csv"), rows)
pt.write_csv(os.path.join(ROOT, "data", "hidden", "confirm_true_daily.csv"), truth_rows)
print("confirm set:", len(runs), "campaigns")
