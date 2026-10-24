"""
Exports every number the web page shows into web/data.json, straight from the
project code, so the page can never disagree with the notebook.
Audit-only items (true TMT, true propane effect) are read from the truth folder
here, in tools/, never in model code.
Run:  python3 tools/export_web_data.py   (~2 min)
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import coil, coke, models as M, value as V

r = lambda x, n=2: float(round(float(x), n))
out = {}

# --- learning curves: mean and standard error per model and training size
lc = {}
for q in "abc":
    for rule in ["1se", "min"]:
        d = M.learning_curve_cached(q, rule)
        lc[f"{q}_{rule}"] = {
            m: [dict(n=int(n), mae=r(v.mean()), se=r(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0))
                for n, v in dm.groupby("n_train").mae]
            for m, dm in d.groupby("model")}
out["learning_curves"] = lc

# --- predictions on the 40 held-back campaigns (train on all 30 history campaigns)
H, ph = M.features("history"), M.physics_table("history")
RES = {}
for q in ["b", "c"]:
    for rule in ["1se", "min"]:
        M.RULE = rule
        for split in ["test", "confirm"]:
            T = M.features(split)
            preds, _ = M.fit_predict(H, T, ph, M.physics_table(split), q)
            for k, v in preds.items():
                RES[(q, rule, split, k)] = (v, T.remaining.values, T.run_length.values)
M.RULE = "1se"

# paired bootstrap CI of the gain over physics (same procedure as the notebook)
rng = np.random.default_rng(1)
ci = []
for q in ["c", "b"]:
    for rule in ["1se", "min"]:
        # "confirm" = the fresh set made after the results and used once (the clean check);
        # "pooled" = test + confirm (40 campaigns) - but the test set also guided method choices
        for set_name, splits in [("confirm", ["confirm"]), ("pooled", ["test", "confirm"])]:
            e = {k: np.concatenate([np.abs(RES[(q, rule, s, k)][0] - RES[(q, rule, s, k)][1]) for s in splits])
                 for k in ["physics", "hybrid_residual", "hybrid_param", "ml_gbm"]}
            for other in ["hybrid_residual", "hybrid_param", "ml_gbm"]:
                d = e["physics"] - e[other]
                b = [rng.choice(d, len(d)).mean() for _ in range(5000)]
                ci.append(dict(set=set_name, n=len(d), q=q, rule=rule, model=other, mae=r(e[other].mean()), physics=r(e["physics"].mean()),
                               gain=r(d.mean()), lo=r(np.percentile(b, 2.5)), hi=r(np.percentile(b, 97.5))))
out["ci"] = ci

# Case A raw predictions so the page can recompute euros live (what-if question c)
cands = [("1se", "physics"), ("min", "hybrid_param"), ("min", "hybrid_residual"), ("1se", "ml_gbm"), ("1se", "ml_ridge")]
out["case_a"] = {m: {s: dict(pred=[r(x, 6) for x in RES[("c", rule, s, m)][0]], rem=[r(x, 6) for x in RES[("c", rule, s, m)][1]],
                             run=[r(x, 6) for x in RES[("c", rule, s, m)][2]]) for s in ["test", "confirm"]} for rule, m in cands}
# reference numbers from Python, used by tests/test_web.js to prove the page computes the same
ref = {}
for rule, m in cands:
    sets = {s: RES[("c", rule, s, m)] for s in ["test", "confirm"]}
    for method in ["normal", "empirical"]:
        for spare in [False, True]:
            ref[f"{m}|{method}|{spare}"] = r(V.case_a(sets, dict(spare_furnace=spare), method)["eur_lost"], 0)
out["case_a_reference"] = ref

# Case B: physics run length and ethylene per COT
A_c = coke.calibrate_A_c()
out["cot"] = [dict(cot=c["cot"], run=r(c["run_days"], 1), tpd=r(c["ethylene_tpd"], 1)) for c in V.cot_tradeoff(range(836, 868, 2), A_c)]

# Step 2: coke thickness along the coil at the design point
op = coke.operating_point(coil.FEED_TPH, coil.STEAM_RATIO, coil.COT_DESIGN)
rate = [coke.coke_rate(T, C, A_c) for T, C in zip(op["prof"]["T_C"], op["prof"]["conc_C4H6"])]
out["coke_profile"] = dict(z=[r(z, 3) for z in op["prof"]["z"]],
                           days={d: [r(x * 1e-3 * d * 86400 / coke.RHO_COKE * 1e3, 3) for x in rate] for d in [15, 30, 45, 60]})

# Step 3: one campaign as the plant sees it (audit view of the truth)
truth = pd.read_csv(os.path.join(ROOT, "data", "hidden", "history_true_daily.csv"))
pyro = pd.read_csv(os.path.join(ROOT, "data", "history", "pyrometer_weekly.csv"))
rid = "101-04"
t, p = truth[truth.run_id == rid], pyro[pyro.run_id == rid]
out["campaign"] = dict(id=rid, true=[[int(a), r(b, 1)] for a, b in zip(t.day, t.tmt_max)],
                       pyro=[[int(a), r(b, 1)] for a, b in zip(p.day, p.tmt_max_meas)], limit=1100)

# propane effect learned by the parameter hybrid vs truth
M.RULE = "min"
X = np.column_stack([H.propane_mean, [ph[i]["tmt_offset"] for i in H.run_id]])
M.fit_predict_param(ph, H.run_id, X, H.remaining.values, ph, H.run_id, X, "c")
M.RULE = "1se"
fit = M.fit_predict_param.last
tp = json.load(open(os.path.join(ROOT, "data", "hidden", "truth_params.json")))["TRUTH"]
x = H.propane_mean.mean()
# rough uncertainty: refit on 40 bootstrap resamples of the 30 training campaigns (light penalty)
rngp = np.random.default_rng(0); boot = []; ids = np.array(H.run_id)
for _ in range(40):
    k = rngp.choice(len(H), len(H), replace=True); Xb = X[k]; mu, sd = Xb.mean(0), Xb.std(0) + 1e-9
    th = M._fit_param(ph, list(ids[k]), (Xb - mu) / sd, H.remaining.values[k], "c", 0.1)
    boot.append(th[1] / sd[0])
out["propane"] = dict(learned=r(fit["theta"][1] / fit["sd"][0], 2), truth=r(tp["propane_gain"] / (1 + tp["propane_gain"] * x), 2),
                      lo90=r(np.percentile(boot, 5), 1), hi90=r(np.percentile(boot, 95), 1),
                      pyro_coef_per_10K=r(fit["theta"][2] / fit["sd"][1] * 10, 3))

out["econ"] = dict(cents_per_lb=15, usd_per_eur=V.EUR_USD, c_lb_to_usd_t=V.CENT_PER_LB_TO_USD_PER_T,
                   decoke_days=V.ECON["decoke_days"], decoke_cost=V.ECON["decoke_cost"], tpd=r(V.design_ethylene_tpd(), 6))

os.makedirs(os.path.join(ROOT, "web"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "web", "data.json"), "w"), separators=(",", ":"))
print("web/data.json written,", os.path.getsize(os.path.join(ROOT, "web", "data.json")) // 1024, "kB")
print("propane learned/truth (% per mol%):", out["propane"])
