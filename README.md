# Ethane furnace coking: physics vs ML vs hybrid

Predicting when an ethane cracking furnace must be decoked (run length), and what that prediction is worth, from **sparse** plant data.
Three models are compared on the same synthetic data with a **hidden truth**: physics-only, ML-only and a hybrid.

> **Engineering prototype.** All data is synthetic and illustrative. No plant, client or employer data is used.

## Why a hidden truth
Real plant data never tells you the right answer. Here a "true" furnace (`plant_truth.py`) generates what a plant records,
including daily DCS values and weekly pyrometer readings, while the models only ever see those records. The truth contains effects the physics
model lacks (initial fast coking, feed propane and different coke properties change run length; tube narrowing and a biased pyrometer change only the measurements). It still shares the physics model's chemistry and heat transfer, so the results apply to a structurally good physics model; real plants differ more.

## Physics core
- **Coil:** reused from my [Cracking Furnace Trainer](https://github.com/ShreeChem/Cracking-Furnace-Trainer), ported 1:1 to Python and verified against the JavaScript.
  Sundaram & Froment (1977) molecular ethane scheme.
- **Coke:** rate form and activation energy (118.1 kJ/mol) from Sundaram, Van Damme & Froment (1981), Model III (C4+ precursor, gas temperature, asymptotic rate).
  Pre-exponential factor calibrated to a 60-day design run.

## Status
| Step | Content | State |
|---|---|---|
| 1 | Coil model port + verification | done |
| 2 | Coke growth + run length physics | done |
| 3 | Hidden-truth plant + sparse data | done |
| 4 | Physics vs ML vs hybrid | done |
| 5 | € value of better run-length prediction | done |
| 6 | Interactive web page | done |

## Step 4 result (honest version, after an independent review)
- With sparse data, a physics model with **one** tuned factor is the strongest single tool. With 2–8 campaigns ML alone is 1.3–3× worse, and even with 30 campaigns (≈ 5 years of one furnace) it had not clearly caught up.
- A hybrid helped a little in *what-if* use ("if I hold these conditions, how many days are left?"): on the fresh confirmation set the gain is 0.3–0.6 days (7–14 %) and **not proven**. It looked larger on the test set, which also guided method choices.
- The parameter hybrid learns the feed-propane effect on coking (+5.6 % per mol % vs +6.7 % true; rough 90 % range 2–10 %).
- In pure forecasting no model beat physics significantly: what operators do next dominates the error.
- The test furnace shares the physics model's chemistry and heat transfer, so this tests a structurally good physics model. Real plants differ more.

## Step 5 result (€, with stated assumptions)
- Decoking early "to be safe" costs about €130–420k per furnace-year in a plant without a spare furnace (€25–77k with one), compared with perfect foresight, whatever the model.
- The hybrid's extra value over tuned physics is **−€12k to +€123k per furnace-year**, depending on the hybrid type and how the safety margin is sized, so it is reported as a range.
- COT choice is a multi-million € trade-off in this model, but its optimum depends on plant costs the model does not contain (coil life, fuel, compressor limits).
- Inputs: RBN Energy ethane-cracker margin, OPIS exchange rate, Imubit/Prismecs decoke duration; the cost of one decoke is an assumption.

## Interactive page
`docs/index.html` is a single self-contained page: learning curves, confidence intervals, and a live € calculator where you change the margin, decoke days, decoke cost and spare-furnace case. Every number is exported from the Python code (`tools/export_web_data.py`), and `tests/test_web.js` checks that the page's calculator reproduces `value.py` exactly.

## Run
```
python3 tests/test_coil.py && python3 tests/test_coke.py
python3 plant_truth.py && python3 tools/make_confirm.py
python3 tests/test_truth.py
python3 models.py
python3 tests/test_models.py
python3 tests/test_value.py
python3 tools/build_notebook.py
python3 tools/export_web_data.py && python3 tools/build_web.py && node tests/test_web.js
```

## References
- Sundaram, K. M., Froment, G. F. (1977). Modeling of thermal cracking kinetics I. *Chem. Eng. Sci.* 32, 601–608.
- Sundaram, K. M., Van Damme, P. S., Froment, G. F. (1981). Coke deposition in the thermal cracking of ethane. *AIChE J.* 27(6), 946–951.

## Author
Madhushree Bharadwaj, process simulation and OTS engineer.
