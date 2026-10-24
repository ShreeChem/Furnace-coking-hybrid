"""
coke.py - Coke growth and run length for one radiant coil (physics model).

The idea in plain words
-----------------------
1. Each day, solve the coil (coil.py) at that day's feed, steam ratio and COT.
   This gives the gas temperature and the butadiene concentration in every cell.
2. Coke grows on the tube wall in every cell at the rate
        r_c = A_c * exp(-E_c / (R * T_gas)) * C_C4H6           [g coke / m2 / s]
   Form and E_c follow Model III of Sundaram, Van Damme & Froment (1981):
   coke from the C4+ fraction, rate taken at the GAS temperature, and the
   asymptotic (long-run) rate. We use butadiene as the C4+ proxy.
3. The coke layer insulates the tube. To keep the same COT the firebox must
   push the same heat through a thicker layer, so the tube metal gets hotter:
        TMT = T_gas + (clean film + metal drop) + q * thickness / k_coke
4. When the hottest point of the tube reaches the TMT limit, the run is over:
   that day is the RUN LENGTH, and the furnace goes to decoking.

What is published and what is calibrated
----------------------------------------
- E_c = 118.1 kJ/mol (28.25 kcal/mol): Model III, Table 2 of the 1981 paper
  (AIChE J. 27(6), 946-951), as read from a partly garbled scan.
  The scanned table is partly garbled, but its kcal and kJ columns agree
  (28.25 x 4.184 = 118.2), so we trust E_c.
- A_c: the scanned value is not readable, and our butadiene level differs from
  theirs anyway. A_c is therefore CALIBRATED so the design case runs
  RUN_DESIGN_DAYS. This is a stated assumption, not a published value.
- Coke density and conductivity: literature values vary (1200-1680 kg/m3;
  0.2-6.5 W/m/K for soft vs hard coke). We use the hard-coke values used in
  run-length studies: 6.46 W/m/K and 1680 kg/m3 from "Estimation of kinetic
  parameters of coking ...", International Journal of Engineering (IJE)
  Transactions B 17(4), 2004, p. 319 ff. (Table 5; mapping of values read from layout).

Engineering prototype on synthetic, illustrative values. Not plant data.
"""
import math
import coil

RG = coil.RG
E_C = 118.1e3              # J/mol, Sundaram, Van Damme & Froment (1981), Model III
RHO_COKE = 1680.0          # kg/m3
K_COKE = 6.46              # W/m/K
D_IN = 0.11                # m, assumed tube inner diameter (gives about 108 kW/m2 heat flux)
DTF_DESIGN = 165.0         # K, clean tube wall minus gas at the outlet at design (trainer: 1015 - 850)
TMT_LIMIT = 1100.0         # C, end-of-run limit: 1100 C for a Technip ethane furnace in "Mathematical Modeling of Ethane
                           # Cracking Furnace of Olefin Plant with Coke Formation Approach"; Sundaram et al. 1981 used 1030 C
RUN_DESIGN_DAYS = 60.0     # calibration target for the design case (modern ethane furnaces: 60-90 days)

CAL = coil.calibrate(full_scheme=True)     # coil volume and design heat with the 5-reaction scheme
AREA = 4 * CAL["V"] / D_IN                 # inner tube area of one pass, m2
NE0, NW0 = coil.design_flows()


def q_for_cot(nE, nW, cot_sp):
    """Heat (W) the coil must absorb to reach a COT setpoint (what the COT controller does)."""
    lo, hi = 2e6, 2e7
    for _ in range(30):
        m = (lo + hi) / 2
        if coil.solve_coil(nE, nW, m, CAL["V"], full_scheme=True)["cot"] < cot_sp:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def operating_point(feed_tph, ratio, cot_sp):
    """Solve the coil for one day's conditions. Returns profiles and heat flux."""
    nE, nW = coil.design_flows(feed_tph, ratio)
    Q = q_for_cot(nE, nW, cot_sp)
    r = coil.solve_coil(nE, nW, Q, CAL["V"], profiles=True, full_scheme=True)
    q_flux = Q / AREA                                     # W/m2, uniform along the coil
    g_ratio = (nE * coil.MW["e"] + nW * coil.MW["w"]) / (NE0 * coil.MW["e"] + NW0 * coil.MW["w"])
    dtf = DTF_DESIGN * (Q / CAL["Q"]) / g_ratio ** 0.8    # film drop: ~flux / h, with h ~ G^0.8 (as in the trainer)
    return dict(prof=r["profiles"], q_flux=q_flux, dtf=dtf, X=r["X"],
                ethylene_tph=coil.ethylene_tph(r["y"]), cot=r["cot"])


def coke_rate(T_C, c_c4h6, A_c):
    """Coke growth rate in g/m2/s at gas temperature T_C (C) and butadiene concentration (mol/m3)."""
    return A_c * math.exp(-E_C / (RG * (T_C + 273.15))) * c_c4h6


_OP_CACHE = {}     # coil solutions depend only on conditions, not on A_c, so share them across runs


def simulate_run(days_conditions, A_c, dt_days=1.0, max_days=400):
    """
    Run the furnace from clean tubes until the TMT limit.
      days_conditions(day) -> (feed_tph, steam_ratio, cot_sp)
    Returns the run length (days, interpolated) and a daily history.
    """
    thick = [0.0] * coil.NCELL                   # coke thickness per cell, m
    hist = []
    prev_tmt = None
    day = 0.0
    cache = _OP_CACHE
    while day < max_days:
        cond = days_conditions(day)
        key = tuple(round(c, 4) for c in cond)
        if key not in cache:                     # same conditions -> same coil solution
            cache[key] = operating_point(*cond)
        op = cache[key]
        p = op["prof"]
        tmt = [p["T_C"][i] + op["dtf"] + op["q_flux"] * thick[i] / K_COKE for i in range(coil.NCELL)]
        tmt_max = max(tmt)
        hist.append(dict(day=day, feed=cond[0], ratio=cond[1], cot_sp=cond[2], tmt_max=tmt_max,
                         thick_max_mm=max(thick) * 1e3, ethylene_tph=op["ethylene_tph"], X=op["X"]))
        if tmt_max >= TMT_LIMIT:
            # linear interpolation between yesterday and today for the crossing day
            if prev_tmt is None:
                return 0.0, hist
            frac = (TMT_LIMIT - prev_tmt) / (tmt_max - prev_tmt)
            return day - dt_days + frac * dt_days, hist
        prev_tmt = tmt_max
        # grow coke for one time step
        for i in range(coil.NCELL):
            rc = coke_rate(p["T_C"][i], p["conc_C4H6"][i], A_c)       # g/m2/s
            thick[i] += rc * 1e-3 * dt_days * 86400 / RHO_COKE         # m
        day += dt_days
    return float("inf"), hist


def calibrate_A_c(target_days=RUN_DESIGN_DAYS):
    """
    At constant conditions the coke thickness grows linearly with A_c, so the
    run length is exactly proportional to 1/A_c. One trial run and one
    correction are enough; a second pass removes the daily-step rounding.
    """
    design = lambda d: (coil.FEED_TPH, coil.STEAM_RATIO, coil.COT_DESIGN)
    A = 1e6
    for _ in range(3):
        run, _ = simulate_run(design, A, dt_days=0.25)
        A *= run / target_days
    return A


if __name__ == "__main__":
    A_c = calibrate_A_c()
    design = lambda d: (coil.FEED_TPH, coil.STEAM_RATIO, coil.COT_DESIGN)
    run, hist = simulate_run(design, A_c)
    op = operating_point(coil.FEED_TPH, coil.STEAM_RATIO, coil.COT_DESIGN)
    print(f"Calibrated A_c = {A_c:.3e} g/m2/s per (mol/m3)   (E_c = {E_C/1e3:.1f} kJ/mol, published)")
    print(f"Heat flux = {op['q_flux']/1e3:.0f} kW/m2, clean outlet TMT = {op['prof']['T_C'][-1] + op['dtf']:.0f} C")
    print(f"Design run length = {run:.1f} days, max coke at end = {hist[-1]['thick_max_mm']:.1f} mm")
