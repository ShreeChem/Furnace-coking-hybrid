"""
Checks the Python coil model against the ORIGINAL JavaScript (engine.js).
Reference numbers come from tests/js_reference.json, produced by:
    node tests/js_reference.js <path>/Cracking-Furnace-Trainer/src/engine.js > tests/js_reference.json
Run:  python3 tests/test_coil.py
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import coil

ref = json.load(open(os.path.join(os.path.dirname(__file__), "js_reference.json")))
cal = coil.calibrate()
nE0, nW0 = coil.design_flows()
fails = 0

def check(name, py, js, tol):
    global fails
    ok = abs(py - js) <= tol
    fails += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name:<28} python {py:14.6f}   js {js:14.6f}   diff {py - js:+.2e}")

print("Calibration")
check("V (m3)", cal["V"], ref["CAL0"]["V"], 1e-9)
check("Q (W)", cal["Q"], ref["CAL0"]["Q"], 1e-3)
check("X at design", cal["X"], ref["CAL0"]["X"], 1e-12)

print("\nOff-design cases (feed x, steam x, heat x)")
for c in ref["cases"]:
    r = coil.solve_coil(nE0 * c["fe"], nW0 * c["fw"], ref["CAL0"]["Q"] * c["fq"], ref["CAL0"]["V"])
    tag = f"{c['fe']}/{c['fw']}/{c['fq']}"
    check(f"COT  {tag}", r["cot"], c["cot"], 1e-6)
    check(f"X    {tag}", r["X"], c["X"], 1e-9)

# physics sanity: profiles must be monotonic along the coil
p = coil.solve_coil(nE0, nW0, cal["Q"], cal["V"], profiles=True)["profiles"]
mono = all(b > a for a, b in zip(p["X"], p["X"][1:]))
print(f"\n{'OK  ' if mono else 'FAIL'} conversion rises along the coil")
fails += not mono

print("\nALL PASSED" if fails == 0 else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
