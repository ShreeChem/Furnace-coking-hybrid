"""
plant_truth.py - The "hidden truth" furnace that produces our synthetic plant data.

Why this file exists
--------------------
To judge physics-only, ML-only and hybrid models fairly, we need to know the
RIGHT answer. Real plant data never tells you that. So we build a "true"
furnace here, let it run for many campaigns, and record only what a real
plant would record: noisy daily DCS values and sparse weekly pyrometer
readings. The models in Step 4 see only those records, never this file.

How the truth differs from our physics model (coke.py)
-------------------------------------------------------
If the truth were just coke.py with other numbers, the hybrid would win by
construction and any reviewer would call it rigged. So the truth contains
effects the physics model does NOT have:

 1. Initial fast coking. Sundaram et al. (1981) report coking rates that are
    "initially high but rapidly decrease" to an asymptote. coke.py uses only
    the asymptote; the truth adds a fast first layer over the first days.
 2. Feed propane. Real "ethane" feed carries some propane, which forms more
    coke precursors. Each campaign gets its own propane level. coke.py assumes
    pure ethane. The lab measures propane weekly (noisy).
 3. Different numbers where the literature is uncertain: activation energy,
    coke conductivity, clean-tube film drop. coke.py uses literature values;
    the truth uses other plausible values inside the published spread.
 4. Coke narrows the tube, so coil inlet pressure (CIP) rises. coke.py ignores
    this; the truth records CIP daily, a signal ML can exploit.
 5. Sensors: DCS values carry noise; pyrometer surveys are weekly, noisy and
    have a per-survey bias (emissivity, angle, operator).

What is shared: the gas-phase chemistry (coil.py). We assume the cracking
kinetics are known and the uncertainty sits in the coking, which is the
realistic situation for a furnace engineer.

Engineering prototype on synthetic, illustrative values. Not plant data.
"""
import json, math, os
import numpy as np
import coil, coke

# ---------------- hidden truth parameters (models never read these) ----------------
TRUTH = dict(
    E_c=128e3,             # J/mol      (coke.py: 118.1e3, from the 1981 paper)
    k_coke=5.6,            # W/m/K      (coke.py: 6.46)
    dtf_design=172.0,      # K          (coke.py: 165)
    propane_gain=8.0,      # coke rate multiplier = 1 + gain * propane mole fraction
    init_mm=0.9,           # mm of fast initial coke at the outlet (scaled along the coil)
    init_tau_days=3.0,     # time constant of the initial coking phase
    run_design_days=55.0,  # calibration target: design conditions with 2 % propane
    tmt_limit=1100.0,      # C, decoke when the TRUE max tube metal temperature reaches this
)
# sensor behaviour
SENSORS = dict(
    feed_rel_sd=0.005, ratio_sd=0.004, cot_sd=1.0, cip_sd=0.02,   # daily DCS averages
    propane_lab_sd=0.003, lab_every_days=7,                       # weekly lab sample (mole fraction)
    pyro_every_days=7, pyro_noise_sd=8.0, pyro_bias_sd=6.0,       # weekly pyrometer survey (K)
)
CIP_OUTLET = 1.8           # bar a at the coil outlet
DP_CLEAN = 1.2             # bar, clean coil pressure drop at design mass flow


def _coke_rate_true(T_C, c4h6, propane, A):
    return A * math.exp(-TRUTH["E_c"] / (coil.RG * (T_C + 273.15))) * c4h6 * (1 + TRUTH["propane_gain"] * propane)


def run_campaign(cond_fn, A, propane_fn, max_days=200, record=True):
    """
    Run one campaign of the TRUE furnace from clean tubes until decoke.
      cond_fn(day)    -> (feed_tph, steam_ratio, cot_sp)  true operating conditions
      propane_fn(day) -> true propane mole fraction in the feed
    Returns (run_length_days, list of daily true states).
    """
    n = coil.NCELL
    asym = np.zeros(n)                       # slow (asymptotic) coke, m
    init_shape = None                        # where the fast initial coke sits
    days, prev = [], None
    for day in range(max_days):
        feed, ratio, cot = cond_fn(day)
        x_c3 = propane_fn(day)
        op = coke.operating_point(feed, ratio, cot)          # shared gas-phase physics
        p = op["prof"]
        T = np.array(p["T_C"]); c4 = np.array(p["conc_C4H6"])
        rate = np.array([_coke_rate_true(T[i], c4[i], x_c3, A) for i in range(n)])   # g/m2/s
        if init_shape is None:
            init_shape = rate / rate.max()               # fast layer follows the same hot-end shape
        init = TRUTH["init_mm"] * 1e-3 * init_shape * (1 - math.exp(-day / TRUTH["init_tau_days"]))
        thick = asym + init
        dtf = op["dtf"] * TRUTH["dtf_design"] / coke.DTF_DESIGN
        tmt = T + dtf + op["q_flux"] * thick / TRUTH["k_coke"]
        tmt_max = float(tmt.max())
        # pressure drop rises as coke narrows the tube (dP ~ 1/d^4.8 for turbulent flow)
        g = feed * (1 + ratio) / (coil.FEED_TPH * (1 + coil.STEAM_RATIO))   # mass flow / design mass flow
        narrowing = float(np.mean((coke.D_IN / (coke.D_IN - 2 * thick)) ** 4.8))
        cip = CIP_OUTLET + DP_CLEAN * g ** 1.8 * narrowing
        if record:
            days.append(dict(day=day, feed=feed, ratio=ratio, cot=cot, propane=x_c3, tmt_max=tmt_max,
                             cip=cip, coke_out_mm=float(thick[-1] * 1e3), ethylene_tph=op["ethylene_tph"]))
        if tmt_max >= TRUTH["tmt_limit"]:
            if prev is None:
                return 0.0, days
            return day - 1 + (TRUTH["tmt_limit"] - prev) / (tmt_max - prev), days
        prev = tmt_max
        asym += rate * 1e-3 * 86400 / coke.RHO_COKE
    return float(max_days), days


def calibrate_truth():
    """Find the true pre-exponential factor so the design campaign (2 % propane) lasts run_design_days."""
    design = lambda d: (coil.FEED_TPH, coil.STEAM_RATIO, coil.COT_DESIGN)
    lo, hi = 1e2, 1e7                                     # bisection on log A (initial coke makes it non-linear)
    for _ in range(40):
        A = math.sqrt(lo * hi)
        run, _ = run_campaign(design, A, lambda d: 0.02, record=False)
        if run > TRUTH["run_design_days"]:
            lo = A
        else:
            hi = A
    return math.sqrt(lo * hi)


def make_operating_plan(rng):
    """Random but realistic operating policy for one campaign."""
    plan = dict(cot=rng.uniform(843, 858), ratio=rng.uniform(0.30, 0.38), feed=rng.uniform(29.0, 34.0),
                propane=rng.uniform(0.005, 0.04))
    walk = np.clip(np.cumsum(rng.normal(0, 0.6, 400)), -5, 5)   # slow COT drift by operators, within +-5 K
    cuts = rng.random(400) < 0.04                         # ~4 % of days: throughput cut (downstream limit)
    def cond(day):
        f = plan["feed"] * (0.85 if cuts[day] else 1.0)
        return (f, plan["ratio"] + 0.01 * math.sin(day / 9), plan["cot"] + walk[day])
    prop_noise = rng.normal(0, 0.003, 400)
    def propane(day):
        return max(0.0, plan["propane"] + prop_noise[day])
    return plan, cond, propane


def generate(n_runs, seed, A):
    """Simulate n_runs campaigns. Returns (daily rows, pyrometer rows, run rows, true daily rows)."""
    rng = np.random.default_rng(seed)
    daily, pyro, runs, truth_rows = [], [], [], []
    for r in range(n_runs):
        plan, cond, propane = make_operating_plan(rng)
        run_len, days = run_campaign(cond, A, propane)
        rid = f"{seed}-{r:02d}"
        for d in days:
            truth_rows.append(dict(run_id=rid, **d))
            # what the DCS historian records (daily averages with sensor noise)
            daily.append(dict(
                run_id=rid, day=d["day"],
                feed_tph=round(d["feed"] * (1 + rng.normal(0, SENSORS["feed_rel_sd"])), 3),
                steam_ratio=round(d["ratio"] + rng.normal(0, SENSORS["ratio_sd"]), 4),
                cot_c=round(d["cot"] + rng.normal(0, SENSORS["cot_sd"]), 2),
                cip_bara=round(d["cip"] + rng.normal(0, SENSORS["cip_sd"]), 3),
                propane_lab=(round(max(0.0, d["propane"] + rng.normal(0, SENSORS["propane_lab_sd"])), 4)
                             if d["day"] % SENSORS["lab_every_days"] == 3 else None),
            ))
            # weekly pyrometer survey of the hottest tube point
            if d["day"] % SENSORS["pyro_every_days"] == 2:
                pyro.append(dict(run_id=rid, day=d["day"], tmt_max_meas=round(
                    d["tmt_max"] + rng.normal(0, SENSORS["pyro_bias_sd"]) + rng.normal(0, SENSORS["pyro_noise_sd"]), 1)))
        runs.append(dict(run_id=rid, run_length_days=round(run_len, 2),
                         cot_plan=round(plan["cot"], 1), steam_ratio_plan=round(plan["ratio"], 3),
                         feed_plan_tph=round(plan["feed"], 2)))
    return daily, pyro, runs, truth_rows


def write_csv(path, rows):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    A = calibrate_truth()
    print(f"true A = {A:.3e}")
    for name, n, seed in [("history", 30, 101), ("test", 20, 202)]:
        daily, pyro, runs, truth_rows = generate(n, seed, A)
        write_csv(os.path.join(here, "data", name, "dcs_daily.csv"), daily)
        write_csv(os.path.join(here, "data", name, "pyrometer_weekly.csv"), pyro)
        write_csv(os.path.join(here, "data", name, "runs.csv"), runs)
        write_csv(os.path.join(here, "data", "hidden", f"{name}_true_daily.csv"), truth_rows)
        rl = [r["run_length_days"] for r in runs]
        print(f"{name}: {n} campaigns, run length {min(rl):.0f}-{max(rl):.0f} d (mean {np.mean(rl):.0f}), "
              f"{len(daily)} DCS days, {len(pyro)} pyrometer readings")
    json.dump(dict(TRUTH=TRUTH, SENSORS=SENSORS, A_true=A), open(os.path.join(here, "data", "hidden", "truth_params.json"), "w"), indent=2)
    print("hidden truth written to data/hidden/ (models must not read it)")
