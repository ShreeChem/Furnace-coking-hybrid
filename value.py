"""
value.py - Step 5: what is a better run-length prediction worth, in euros?

The idea in plain words
-----------------------
A prediction is only worth money if it changes a decision. Two decisions:

 CASE A - "When do we schedule the decoke?"  (compares the models)
   On day 21 the model predicts the days left. Because the prediction can be
   wrong, planners decoke EARLY by a safety margin. The margin is set so that
   only about 1 in 20 campaigns would hit the tube limit before the planned
   decoke. A more accurate model needs a smaller margin -> longer campaigns ->
   fewer decokes per year. Fewer decokes = less downtime and less decoke cost.

 CASE B - "Which COT should we run?"  (what the model is FOR)
   Hotter COT makes more ethylene per hour but cokes faster. The physics model
   turns each COT into a run length, so we can find the COT that earns most
   per day over the whole cycle (cracking days + decoke days).

Where the numbers come from (all editable in ECON)
---------------------------------------------------
 margin  Ethane-cracker cash margin, RBN Energy: long-term average ~15 c/lb,
         ~19 c/lb in June 2026, ~30 c/lb at the April 2026 peak.
         1 c/lb = 22.05 $/t; 1 EUR = 1.152 USD (OPIS, April 2026 settlement).
         This is a US Gulf ethane margin; European crackers differ.
 decoke  1-3 days per decoke (Imubit; Prismecs). Base 2 days.
 cost    Direct cost of one decoke (steam, air, fuel, crew, tube life used):
         NO PUBLIC SOURCE FOUND. Base 50 kEUR is an ASSUMPTION; we show 20-150 kEUR.
 spare   Plants run 6-12 furnaces with one in decoke and one on hot standby
         (Imubit). If a spare furnace covers the decoke, no production is lost
         and only the direct cost counts. We show both cases.
 recycle Unconverted ethane is recycled, so the margin per tonne of ethylene is
         taken as constant and more conversion = more ethylene from the furnace.
         (Simplification; fuel cost of a hotter COT is ignored.)
 limits  Case A compares with perfect foresight and gives a campaign that
         hits the tube limit before the planned decoke NO extra cost (in a
         plant it would mean an unplanned decoke and tube stress). The annual
         figure averages per-campaign annual profit (an approximation).
"""
import numpy as np
import coil, coke

CENT_PER_LB_TO_USD_PER_T = 22.0462
EUR_USD = 1.152


def margin_eur_per_t(cents_per_lb):
    return cents_per_lb * CENT_PER_LB_TO_USD_PER_T / EUR_USD


ECON = dict(
    margin=margin_eur_per_t(15),      # EUR per t ethylene (long-term average; conservative base)
    decoke_days=2.0,                  # days per decoke
    decoke_cost=50_000.0,             # EUR per decoke (ASSUMPTION)
    spare_furnace=False,              # False = furnace-limited plant: decoke days lose production
    ethylene_tpd=None,                # t ethylene per day of this furnace (None = from the coil model)
)


def design_ethylene_tpd():
    op = coke.operating_point(coil.FEED_TPH, coil.STEAM_RATIO, coil.COT_DESIGN)
    return op["ethylene_tph"] * 24


def annual_profit(run_days, econ=ECON):
    """
    EUR per furnace-year for campaigns of `run_days` cracking days.
    One cycle = run_days cracking + decoke_days decoking.
    Furnace-limited: production happens only on cracking days.
    Spare furnace:   production continues; only the decoke cost counts.
    """
    e = dict(ECON, **econ)
    tpd = e["ethylene_tpd"] or design_ethylene_tpd()
    run_days = np.asarray(run_days, dtype=float)
    cycles = 365.0 / (run_days + e["decoke_days"])
    if e["spare_furnace"]:
        production = 365.0 * tpd * e["margin"]
    else:
        production = 365.0 * run_days / (run_days + e["decoke_days"]) * tpd * e["margin"]
    return production - cycles * e["decoke_cost"]


# ------------------------------------------------------------ CASE A
def safety_margin(errors, risk=0.05, method="normal"):
    """
    errors = predicted - true remaining days on campaigns we already know.
    Planning at (prediction - margin) is late only when error > margin, so the
    margin is the (1 - risk) point of the error distribution. Never negative.
      method="normal":    mean + 1.645 x standard deviation (assumes roughly normal errors;
                          stable with few campaigns)  <- main method
      method="empirical": the 95 % quantile of the errors themselves (no assumption, but with
                          20 campaigns it is close to "the second-worst case" and very noisy)
    """
    errors = np.asarray(errors, dtype=float)
    if method == "normal":
        z = {0.05: 1.645, 0.10: 1.282, 0.01: 2.326}[risk]
        return max(0.0, float(errors.mean() + z * errors.std(ddof=1)))
    return max(0.0, float(np.quantile(errors, 1 - risk)))


def plan_decoke(pred_remaining, margin, true_remaining):
    """Planned remaining days, days decoked too early, and whether the tube limit was hit first."""
    planned = np.maximum(pred_remaining - margin, 0.0)
    early = np.maximum(true_remaining - planned, 0.0)
    late = planned > true_remaining
    return planned, early, late


def value_of_early_days(true_run_days, early_days, econ=ECON):
    """
    EUR per furnace-year lost because campaigns end `early_days` before they had to.
    Compares the actual campaign length with the perfect-foresight length.
    """
    perfect = annual_profit(true_run_days, econ)
    actual = annual_profit(np.maximum(true_run_days - early_days, 1.0), econ)
    return float(np.mean(perfect - actual))


def case_a(pred_err_sets, econ=ECON, method="normal"):
    """
    Cross-fitting over two held-back sets: the margin is set on one set and judged on the other, then swapped.
      pred_err_sets = {"A": (pred, true_remaining, true_run_length), "B": (...)}
    Returns mean days decoked too early, number of campaigns that hit the tube limit first,
    and EUR per furnace-year lost compared with perfect foresight.
    """
    early, runs, late, margins = [], [], 0, []
    keys = list(pred_err_sets)
    for a, b in [(keys[0], keys[1]), (keys[1], keys[0])]:
        pa, ta, _ = pred_err_sets[a]
        pb, tb, lb = pred_err_sets[b]
        m = safety_margin(pa - ta, method=method)
        _, e, lt = plan_decoke(pb, m, tb)
        early += list(e); runs += list(lb); late += int(lt.sum()); margins.append(m)
    early, runs = np.array(early), np.array(runs)
    return dict(margin_days=float(np.mean(margins)), early_days=float(early.mean()), late=late, n=len(early),
                eur_lost=value_of_early_days(runs, early, econ))


# ------------------------------------------------------------ CASE B
def cot_tradeoff(cots, A_c, feed=coil.FEED_TPH, ratio=coil.STEAM_RATIO, econ=ECON):
    """Run length, ethylene per day and EUR per furnace-year for each COT (physics model, constant conditions)."""
    e = dict(ECON, **econ)
    rows = []
    for cot in cots:
        run, hist = coke.simulate_run(lambda d: (feed, ratio, cot), A_c)
        tpd = hist[0]["ethylene_tph"] * 24
        rows.append(dict(cot=cot, run_days=run, ethylene_tpd=tpd,
                         eur_per_year=float(annual_profit(run, dict(e, ethylene_tpd=tpd)))))
    return rows
