"""
models.py - Step 4: physics-only, ML-only and hybrid run-length models.

Two questions an operator asks
------------------------------
 (a) START: "How long will this campaign last?" asked at the start of a run,
     knowing only the operating plan and the first propane lab sample (day 3).
 (b) DAY 21: "How many days are left?" asked on day 21, knowing the daily DCS
     data so far, three weekly pyrometer readings and three propane samples.
     Future operation is unknown: we assume last week's conditions continue.
 (c) WHAT-IF on day 21: same as (b), but the models are TOLD the conditions the
     furnace will run at for the rest of the campaign (in the test set: the
     conditions it actually ran at). This separates "is the model right?" from
     "can it guess what operators will do next?".

Five models, all trained on the same campaigns and tested on the same 20
held-back campaigns:
  physics   coke.py with ONE number fitted on the training campaigns: a factor
            on the coking rate (what an engineer does when "tuning" a model).
  ml_ridge  linear regression with ridge penalty on measured features only.
  ml_gbm    gradient boosting (trees) on the same features. Strong with lots
            of data, weak with little.
  hybrid_residual  physics prediction + a small ridge model that learns the
            physics ERROR (the residual) from a few features. "Correct the output."
  hybrid_param     ML predicts the physics coking factor for each campaign from
            a few features (propane, tube temperature offset), then the physics
            model runs with it. "Learn the physics parameter" (simulator as prior).

How penalties are chosen (same rule for every model):
  leave-one-out on the TRAINING campaigns (each campaign is left out once and
  predicted by the others), then the ONE-STANDARD-ERROR RULE: take the simplest
  option whose error is within one standard error of the best (Hastie,
  Tibshirani & Friedman, Elements of Statistical Learning, sec. 7.10).
  "Simplest" means the strongest penalty; for the hybrids it includes "no
  correction at all", so with little data they fall back to physics.
  The test campaigns are never used for any choice.
(History of this rule: a fixed penalty made the residual hybrid worse than
physics with 2 campaigns; plain leave-one-out still picked complex options on
ties. Both flaws are why the 1-SE rule is used.)
This file reads only the plant records (data/history, data/test, data/confirm),
never the truth folder.
"""
import json, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import coke

ROOT = os.path.dirname(os.path.abspath(__file__))
CUT = 21                                         # day of questions (b) and (c)
KEY = dict(a="start", b="total_b", c="total_c")  # which pre-run physics result each question uses
MULTS = np.exp(np.linspace(np.log(0.4), np.log(2.5), 13))   # coking-rate factors the physics is pre-run with
A_C = None                                       # nominal physics coking constant (set on first use)


def load(split):
    d = os.path.join(ROOT, "data", split)
    return (pd.read_csv(f"{d}/dcs_daily.csv"), pd.read_csv(f"{d}/pyrometer_weekly.csv"),
            pd.read_csv(f"{d}/runs.csv"))


# ---------------------------------------------------------------- physics runs
def _physics_for_campaign(daily, pyro, plan):
    """Pre-run the physics model at every factor in MULTS, for both questions."""
    global A_C
    if A_C is None:
        A_C = coke.calibrate_A_c()
    plan_cond = (plan.feed_plan_tph, plan.steam_ratio_plan, plan.cot_plan)
    # (b) measured conditions for days 0..CUT-1, then assume the last week's average continues
    seen = daily[daily.day < CUT]
    meas = list(zip(seen.feed_tph.round(2), seen.steam_ratio.round(3), seen.cot_c.round(1)))
    wk = seen[seen.day >= CUT - 7]
    last = (round(wk.feed_tph.mean(), 2), round(wk.steam_ratio.mean(), 3), round(wk.cot_c.mean(), 1))
    cond_b = lambda day: meas[int(day)] if day < CUT else last
    allm = list(zip(daily.feed_tph.round(2), daily.steam_ratio.round(3), daily.cot_c.round(1)))
    cond_c = lambda day: allm[min(int(day), len(allm) - 1)]           # actual conditions, held after the last day
    out = dict(start=[], total_b=[], total_c=[])
    tmt_at_pyro = None
    for m in MULTS:
        run_a, _ = coke.simulate_run(lambda day: plan_cond, A_C * m)
        run_b, hist = coke.simulate_run(cond_b, A_C * m)
        out["start"].append(run_a)
        out["total_b"].append(run_b)
        out["total_c"].append(coke.simulate_run(cond_c, A_C * m)[0])
        if np.isclose(m, 1.0):                           # nominal physics TMT, used by the hybrid
            tmt_at_pyro = {h["day"]: h["tmt_max"] for h in hist}
    p = pyro[pyro.day < CUT]
    out["tmt_offset"] = float(np.mean([r.tmt_max_meas - tmt_at_pyro[float(r.day)] for r in p.itertuples()]))
    return out


def physics_table(split):
    """Physics results for every campaign of a split (cached in data/derived)."""
    path = os.path.join(ROOT, "data", "derived", f"physics_{split}.json")
    if os.path.exists(path):
        return json.load(open(path))
    daily, pyro, runs = load(split)
    res = {}
    for plan in runs.itertuples():
        dd, pp = daily[daily.run_id == plan.run_id], pyro[pyro.run_id == plan.run_id]
        res[plan.run_id] = _physics_for_campaign(dd, pp, plan)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w"))
    return res


def physics_predict(ph, run_ids, question, m):
    """Physics prediction at coking factor m, interpolated between the pre-run factors (in log space)."""
    key = KEY[question]
    pred = np.array([np.interp(np.log(m), np.log(MULTS), ph[r][key]) for r in run_ids])
    return pred if question == "a" else np.maximum(pred - CUT, 0.0)


def fit_physics_factor(ph, run_ids, y, question):
    """The ONE physics parameter: the coking-rate factor with the smallest squared error on the training campaigns."""
    grid = np.exp(np.linspace(np.log(MULTS[0]), np.log(MULTS[-1]), 300))
    errs = [np.mean((physics_predict(ph, run_ids, question, m) - y) ** 2) for m in grid]
    return float(grid[int(np.argmin(errs))])


# ---------------------------------------------------------------- features
def features(split):
    """One row per campaign with measured features and both targets."""
    daily, pyro, runs = load(split)
    rows = []
    for plan in runs.itertuples():
        d = daily[daily.run_id == plan.run_id]
        p = pyro[(pyro.run_id == plan.run_id) & (pyro.day < CUT)]
        lab = d.propane_lab.dropna()
        seen = d[d.day < CUT]
        lab_seen = seen.propane_lab.dropna()
        cip_slope = np.polyfit(seen.day, seen.cip_bara, 1)[0]
        tmt_slope = np.polyfit(p.day, p.tmt_max_meas, 1)[0]
        rows.append(dict(
            run_id=plan.run_id,
            # (a) known at the start
            cot_plan=plan.cot_plan, ratio_plan=plan.steam_ratio_plan, feed_plan=plan.feed_plan_tph,
            propane_first=lab.iloc[0],
            # (b) known on day 21
            cot_mean=seen.cot_c.mean(), ratio_mean=seen.steam_ratio.mean(), feed_mean=seen.feed_tph.mean(),
            cip_slope=cip_slope, tmt_last=p.tmt_max_meas.iloc[-1], tmt_slope=tmt_slope,
            propane_mean=lab_seen.mean(),
            # (c) the what-if scenario: conditions for the rest of the campaign
            cot_future=d[d.day >= CUT].cot_c.mean(), ratio_future=d[d.day >= CUT].steam_ratio.mean(),
            feed_future=d[d.day >= CUT].feed_tph.mean(),
            # targets
            run_length=plan.run_length_days, remaining=plan.run_length_days - CUT))
    return pd.DataFrame(rows)


FEAT = dict(
    a=dict(ml=["cot_plan", "ratio_plan", "feed_plan", "propane_first"], hybrid=["propane_first"]),
    b=dict(ml=["cot_mean", "ratio_mean", "feed_mean", "cip_slope", "tmt_last", "tmt_slope", "propane_mean"],
           hybrid=["propane_mean", "tmt_offset"]),
)
FEAT["c"] = dict(ml=FEAT["b"]["ml"] + ["cot_future", "ratio_future", "feed_future"], hybrid=FEAT["b"]["hybrid"])
TARGET = dict(a="run_length", b="remaining", c="remaining")


# ---------------------------------------------------------------- the five models
ALPHAS = np.logspace(4, -2, 13)          # ridge penalties, strongest (simplest) first


RULE = "1se"     # main rule. "min" = plain lowest leave-one-out error (sensitivity check only)


def one_se_choice(loo_errors):
    """loo_errors: list of arrays (one per option, ordered simplest first). Returns index of the chosen option."""
    means = [np.mean(e) for e in loo_errors]
    best = int(np.argmin(means))                 # ties go to the simplest option (it comes first)
    if RULE == "min":
        return best
    se = np.std(loo_errors[best], ddof=1) / np.sqrt(len(loo_errors[best])) if len(loo_errors[best]) > 1 else np.inf
    return next(i for i, m in enumerate(means) if m <= means[best] + se)


def ridge_1se(X, y, extra_null=False):
    """
    Ridge with its penalty chosen by leave-one-out + 1-SE rule.
    extra_null=True adds the simplest option of all: predict 0 (used for residuals = "no correction").
    Returns a fitted predictor function.
    """
    opts = (["null"] if extra_null else []) + list(ALPHAS)
    errs = []
    for o in opts:
        e = []
        for i in range(len(y)):
            k = np.arange(len(y)) != i
            if o == "null":
                p = 0.0
            else:
                p = make_pipeline(StandardScaler(), Ridge(alpha=o)).fit(X[k], y[k]).predict(X[[i]])[0]
            e.append(abs(p - y[i]))
        errs.append(np.array(e))
    o = opts[one_se_choice(errs)]
    if o == "null":
        return lambda Xn: np.zeros(len(Xn))
    m = make_pipeline(StandardScaler(), Ridge(alpha=o)).fit(X, y)
    return m.predict
LAMBDAS = [np.inf, 1000, 100, 10, 1, 0.1]   # parameter-hybrid penalties, simplest first; inf = no correction (pure physics)


def _param_pred(ph, run_ids, Z, theta, question):
    """Physics prediction where each campaign gets its own coking factor m_i = exp(theta0 + theta1*z1 + ...)."""
    logm = theta[0] + Z @ theta[1:]
    key = KEY[question]
    pred = np.array([np.interp(lm, np.log(MULTS), ph[r][key]) for lm, r in zip(logm, run_ids)])
    return pred if question == "a" else np.maximum(pred - CUT, 0.0)


def _fit_param(ph, run_ids, Z, y, question, lam):
    p = Z.shape[1]
    if np.isinf(lam):                                  # no correction: only the common factor theta0 is fitted
        loss = lambda t: np.mean((_param_pred(ph, run_ids, Z, np.r_[t, np.zeros(p)], question) - y) ** 2)
        grid = np.linspace(np.log(MULTS[0]), np.log(MULTS[-1]), 200)
        return np.r_[grid[int(np.argmin([loss(t) for t in grid]))], np.zeros(p)]
    loss = lambda th: np.mean((_param_pred(ph, run_ids, Z, th, question) - y) ** 2) + lam * np.sum(th[1:] ** 2)
    best = None
    for start in (-0.5, 0.0, 0.5):                     # a few starting points: the interpolation is not smooth
        r = minimize(loss, np.r_[start, np.zeros(p)], method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-4, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def fit_predict_param(ph_tr, ids_tr, Xtr, y, ph_te, ids_te, Xte, question):
    """Parameter hybrid: choose the penalty by leave-one-out on training campaigns, refit, predict test."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    ids_tr = list(ids_tr)
    def loo(lam):
        err = []
        for i in range(len(y)):
            k = [j for j in range(len(y)) if j != i]
            th = _fit_param(ph_tr, [ids_tr[j] for j in k], Ztr[k], y[k], question, lam)
            err.append(abs(_param_pred(ph_tr, [ids_tr[i]], Ztr[[i]], th, question)[0] - y[i]))
        return np.array(err)
    lam = LAMBDAS[one_se_choice([loo(l) for l in LAMBDAS])]
    th = _fit_param(ph_tr, ids_tr, Ztr, y, question, lam)
    fit_predict_param.last = dict(lam=lam, theta=th, mu=mu, sd=sd)      # kept for explanation plots
    return _param_pred(ph_te, list(ids_te), Zte, th, question)
def fit_predict(train, test, ph_tr, ph_te, question):
    """Train all four models on `train` campaigns, return predictions for `test` campaigns."""
    y = train[TARGET[question]].values
    out = {}
    # physics: one fitted factor
    m = fit_physics_factor(ph_tr, train.run_id, y, question)
    phys_tr = physics_predict(ph_tr, train.run_id, question, m)
    phys_te = physics_predict(ph_te, test.run_id, question, m)
    out["physics"] = phys_te
    # ML only: measured features, no physics
    Xtr, Xte = train[FEAT[question]["ml"]].values, test[FEAT[question]["ml"]].values
    out["ml_ridge"] = ridge_1se(Xtr, y)(Xte)
    gbm = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0).fit(Xtr, y)
    out["ml_gbm"] = gbm.predict(Xte)
    # hybrid: physics + ridge on the physics residual
    def hyb_X(df, ph):
        cols = FEAT[question]["hybrid"]
        X = df[[c for c in cols if c != "tmt_offset"]].values
        if "tmt_offset" in cols:
            X = np.column_stack([X, [ph[r]["tmt_offset"] for r in df.run_id]])
        return X
    Htr, Hte = hyb_X(train, ph_tr), hyb_X(test, ph_te)
    out["hybrid_residual"] = phys_te + ridge_1se(Htr, y - phys_tr, extra_null=True)(Hte)
    out["hybrid_param"] = fit_predict_param(ph_tr, train.run_id, Htr, y, ph_te, test.run_id, Hte, question)
    if question in ("b", "c"):                           # remaining days cannot be negative
        out = {k: np.maximum(v, 0.0) for k, v in out.items()}
    return out, m


def learning_curve(question, sizes=(2, 4, 8, 16, 30), repeats=20, seed=0, test_split="test"):
    """Mean absolute error on held-back campaigns vs number of training campaigns."""
    H, T = features("history"), features(test_split)
    ph_h, ph_t = physics_table("history"), physics_table(test_split)
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        reps = 1 if n == len(H) else repeats
        for rep in range(reps):
            tr = H.iloc[rng.choice(len(H), n, replace=False)]
            preds, m = fit_predict(tr, T, ph_h, ph_t, question)
            active = bool(np.isfinite(fit_predict_param.last["lam"]))
            for model, p in preds.items():
                rows.append(dict(n_train=n, rep=rep, model=model,
                                 mae=float(np.mean(np.abs(p - T[TARGET[question]].values))), factor=m,
                                 hybrid_correction_on=active))
    return pd.DataFrame(rows)


def learning_curve_cached(question, rule="1se", test_split="test", sizes=(2, 4, 8, 16, 30)):
    """Learning curve for one question and selection rule, cached in data/derived (takes minutes to compute)."""
    global RULE
    suffix = ("" if rule == "1se" else f"_{rule}") + ("" if test_split == "test" else f"_{test_split}")
    path = os.path.join(ROOT, "data", "derived", f"learning_curve_{question}{suffix}.csv")
    if not os.path.exists(path):
        RULE = rule
        try:
            learning_curve(question, sizes=sizes, test_split=test_split).to_csv(path, index=False)
        finally:
            RULE = "1se"
    return pd.read_csv(path)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    for rule in ["1se", "min"]:
        for q in ["a", "b", "c"]:
            lc = learning_curve_cached(q, rule)
            print(f"\nRule {rule} | question ({q}): mean absolute error on test campaigns (days)")
            print(lc.groupby(["n_train", "model"]).mae.mean().unstack().round(1))
