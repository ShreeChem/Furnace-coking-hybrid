"""
coil.py - Radiant coil model for an ethane cracking furnace (physics core).

Ported line by line from the Cracking Furnace Trainer (src/engine.js, solveCoil
and calibrate) so both projects share ONE physics core. tests/test_coil.py
checks that this Python version gives the same numbers as the JavaScript one.

What the model does, in plain words
-----------------------------------
One pass of the furnace is a long tube. Ethane + dilution steam enter at the
crossover temperature (640 C). The tube is cut into 40 cells. In each cell:
  1. heat from the firebox flows in (same amount in every cell = uniform flux),
  2. three reactions convert some ethane (Sundaram & Froment 1977 kinetics),
  3. the gas temperature changes by (heat in - heat used by reactions) / heat capacity.
The gas leaving the last cell gives the coil outlet temperature (COT) and the
ethane conversion X.

New compared with the trainer: solve_coil() also returns the AXIAL PROFILES
(temperature and species along the tube). The coking model in Step 2 needs
them, because coke grows fastest where the gas is hottest and richest in coke
precursors (butadiene in this model, see coke.py) - near the outlet.

Engineering prototype on synthetic, illustrative values. Not plant data.
"""
import math

# ---------- constants (identical to engine.js) ----------
RG = 8.314                # gas constant, J/mol/K
CAL = 4.184               # J per cal
PB = 2.4e5                # mean coil pressure, Pa (2.4 bar abs)
T_XO = 640.0              # crossover (coil inlet) temperature, C
NCELL = 40                # cells along the coil
NSUB = 3                  # small sub-steps inside each cell for stability

# species keys: e ethane, y ethylene, h hydrogen, m methane, p propane, r propylene, w steam,
#               a acetylene, b butadiene (a and b only appear with full_scheme=True)
MW = dict(e=30.07, y=28.05, h=2.016, m=16.04, p=44.1, r=42.08, w=18.02, a=26.04, b=54.09)  # g/mol
CP = dict(e=122, y=94, h=30.2, m=72, p=174, r=150, w=41.3, a=55, b=155)               # J/mol/K near 1000 K

# Sundaram & Froment (1977) molecular ethane scheme, published A and E.
# The 1e-3 converts L/mol/s to m3/mol/s for the two bimolecular reactions.
RX = [
    dict(A=4.65e13,        E=65210 * CAL, dH=144e3),    # R1  C2H6 <=> C2H4 + H2
    dict(A=3.85e11 * 1e-3, E=65250 * CAL, dH=-11.6e3),  # R2  2 C2H6 -> C3H8 + CH4
    dict(A=7.08e13 * 1e-3, E=60430 * CAL, dH=-22.9e3),  # R3  C2H4 + C2H6 -> C3H6 + CH4
]
# Two more reactions of the same 1977 scheme. The trainer leaves them out; Step 2 needs
# them because they make butadiene (C4H6), our proxy for the C4+ coke precursor of
# Sundaram, Van Damme & Froment (1981). dH from standard heats of formation at 298 K.
RX_FULL = [
    dict(A=9.81e8,         E=36920 * CAL, dH=131.8e3),  # R4  C3H6 -> C2H2 + CH4
    dict(A=1.03e12 * 1e-3, E=41260 * CAL, dH=-168.9e3), # R5  C2H2 + C2H4 -> C4H6
]

# ---------- design point (synthetic, identical to engine.js) ----------
FEED_TPH = 32.0           # ethane feed to the furnace, t/h (4 passes)
STEAM_RATIO = 0.32        # kg steam per kg ethane
COT_DESIGN = 850.0        # C
X_DESIGN = 0.65           # ethane conversion at design
N_PASS = 4


def design_flows(feed_tph=FEED_TPH, ratio=STEAM_RATIO):
    """Ethane and steam molar flows (mol/s) into ONE pass."""
    nE = feed_tph * 1e6 / 3600 / N_PASS / MW["e"]
    nW = feed_tph * ratio * 1e6 / 3600 / N_PASS / MW["w"]
    return nE, nW


def solve_coil(nE, nW, Q, V, profiles=False, full_scheme=False):
    """
    March along one coil.
      nE, nW : ethane and steam into the coil, mol/s
      Q      : heat absorbed by the coil, W
      V      : coil internal volume, m3
    Returns a dict with cot (C), X (conversion), y (ethylene mol/s), n (all flows)
    and, if profiles=True, lists along the coil: z (0..1), T_C, conc_C2H4, conc_C4H6 (mol/m3), X.
    full_scheme=False reproduces the trainer exactly (3 reactions);
    full_scheme=True adds R4 and R5 so butadiene is formed.
    """
    n = dict(e=nE, y=0.0, h=0.0, m=0.0, p=0.0, r=0.0, w=nW, a=0.0, b=0.0)
    T = T_XO + 273.15
    if nE + nW < 1e-3:                       # no flow: same shortcut as the trainer
        return dict(cot=T_XO + Q / 2000, X=0.0, y=0.0, n=n)

    vol = V / NCELL / NSUB                   # volume of one sub-step, m3
    dq = Q / NCELL / NSUB                    # heat into one sub-step, W
    prof = dict(z=[], T_C=[], conc_C2H4=[], conc_C4H6=[], X=[]) if profiles else None

    for c in range(NCELL):
        for _ in range(NSUB):
            nt = sum(n.values())
            Cf = PB / (RG * T) / nt          # converts a flow (mol/s) to a concentration (mol/m3)
            k1, k2, k3 = (rx["A"] * math.exp(-rx["E"] / (RG * T)) for rx in RX)
            Kp = math.exp(-(144e3 - T * 136) / (RG * T)) * 1e5   # approximate R1 equilibrium, Pa
            Kc = Kp / (RG * T)                                   # same in mol/m3
            Ce, Cy, Ch = n["e"] * Cf, n["y"] * Cf, n["h"] * Cf

            # reaction extents in this sub-step, mol/s (rate x volume)
            x1 = max(-0.5 * n["y"], k1 * (Ce - Cy * Ch / Kc) * vol)
            x2 = k2 * Ce * Ce * vol
            x3 = k3 * Ce * Cy * vol
            use = x1 + 2 * x2 + x3           # never consume more than half the ethane in one step
            if use > 0.5 * n["e"]:
                f = 0.5 * n["e"] / use
                x1, x2, x3 = x1 * f, x2 * f, x3 * f

            # extra reactions (full scheme only), clipped like the others
            x4 = x5 = 0.0
            if full_scheme:
                k4 = RX_FULL[0]["A"] * math.exp(-RX_FULL[0]["E"] / (RG * T))
                k5 = RX_FULL[1]["A"] * math.exp(-RX_FULL[1]["E"] / (RG * T))
                x4 = min(0.5 * n["r"], k4 * n["r"] * Cf * vol)
                x5 = min(0.5 * n["a"], k5 * n["a"] * Cf * Cy * vol)

            # mole balances
            n["e"] -= x1 + 2 * x2 + x3
            n["y"] += x1 - x3 - x5
            n["h"] += x1
            n["p"] += x2
            n["m"] += x2 + x3 + x4
            n["r"] += x3 - x4
            n["a"] += x4 - x5
            n["b"] += x5

            # energy balance: heat in minus heat consumed by reactions
            cp_tot = sum(n[k] * CP[k] for k in n)
            T += (dq - x1 * RX[0]["dH"] - x2 * RX[1]["dH"] - x3 * RX[2]["dH"]
                  - x4 * RX_FULL[0]["dH"] - x5 * RX_FULL[1]["dH"]) / cp_tot

        if profiles:
            Cf = PB / (RG * T) / sum(n.values())
            prof["z"].append((c + 1) / NCELL)
            prof["T_C"].append(T - 273.15)
            prof["conc_C2H4"].append(n["y"] * Cf)
            prof["conc_C4H6"].append(n["b"] * Cf)
            prof["X"].append(1 - n["e"] / nE)

    out = dict(cot=T - 273.15, X=1 - n["e"] / nE if nE > 0 else 0.0, y=n["y"], n=n)
    if profiles:
        out["profiles"] = prof
    return out


def calibrate(full_scheme=False):
    """
    Find coil volume V and design heat Q so that the design feed gives
    COT = 850 C at 65 % conversion. Same two nested bisections as engine.js:
      inner: for a given V, find the Q that gives COT 850 C
      outer: find the V whose COT-850 point has X = 0.65
    """
    nE0, nW0 = design_flows()

    def q_for(V):
        lo, hi = 1e6, 4e7
        for _ in range(40):
            m = (lo + hi) / 2
            if solve_coil(nE0, nW0, m, V, full_scheme=full_scheme)["cot"] < COT_DESIGN:
                lo = m
            else:
                hi = m
        return (lo + hi) / 2

    lo, hi = 0.05, 20.0
    for _ in range(40):
        V = math.sqrt(lo * hi)                # geometric bisection: V spans orders of magnitude
        if solve_coil(nE0, nW0, q_for(V), V, full_scheme=full_scheme)["X"] < X_DESIGN:
            lo = V
        else:
            hi = V
    V = math.sqrt(lo * hi)
    Q = q_for(V)
    r = solve_coil(nE0, nW0, Q, V, full_scheme=full_scheme)
    return dict(V=V, Q=Q, X=r["X"], Y0=r["y"])


def ethylene_tph(y_mol_s_per_pass):
    """Ethylene production of the whole furnace, t/h, from one pass's ethylene flow."""
    return y_mol_s_per_pass * N_PASS * MW["y"] * 3.6e-3


if __name__ == "__main__":
    cal = calibrate()
    nE0, nW0 = design_flows()
    r = solve_coil(nE0, nW0, cal["Q"], cal["V"], profiles=True)
    print(f"Coil volume per pass  V = {cal['V']:.4f} m3")
    print(f"Heat absorbed / pass  Q = {cal['Q']/1e6:.3f} MW")
    print(f"COT = {r['cot']:.1f} C   X = {r['X']:.3f}")
    print(f"Ethylene = {ethylene_tph(r['y']):.2f} t/h  "
          f"(yield {ethylene_tph(r['y']) / FEED_TPH * 100:.1f} wt% on ethane feed)")
