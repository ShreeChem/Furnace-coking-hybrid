"""
Physics sanity checks for the coke / run-length model.
Run:  python3 tests/test_coke.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import coil, coke

A = coke.calibrate_A_c()
run = lambda feed, ratio, cot: coke.simulate_run(lambda d: (feed, ratio, cot), A)[0]
fails = 0

def check(name, ok):
    global fails
    fails += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name}")

base = run(32, 0.32, 850)
check(f"design run length hits the 60-day calibration target ({base:.1f} d)", abs(base - 60) < 0.5)
check("hotter COT -> shorter run", run(32, 0.32, 860) < base < run(32, 0.32, 840))
check("more dilution steam -> longer run", run(32, 0.28, 850) < base < run(32, 0.40, 850))

op = coke.operating_point(32, 0.32, 850)
rc = [coke.coke_rate(T, C, A) for T, C in zip(op["prof"]["T_C"], op["prof"]["conc_C4H6"])]
check("almost no coke in the first 20 % of the coil (as Sundaram et al. 1981 report)",
      max(rc[: coil.NCELL // 5]) < 0.01 * max(rc))
check("coke rate is highest at the outlet", rc[-1] == max(rc))
print("\nALL PASSED" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
