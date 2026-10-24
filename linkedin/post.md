# LinkedIn post: furnace coking project

**Upload:** `coking_carousel_D.pdf` as a document post (or the three PNG slides as images).
**First comment:** the public page link, https://shreechem.github.io/furnace-coking-hybrid/ (works once the repository is public and GitHub Pages is on).

## Post text

How many days does a cracking furnace really have left, and what does guessing wrong cost?

I built a test furnace where the right answer is known, then compared three ways to predict run length: physics, machine learning and a hybrid of both.

With little plant data, physics wins. Even 30 campaigns, roughly five years of one furnace, weren't enough for ML alone to clearly catch up.

The hybrid helped a little in what-if use, but on fresh data the gain was smaller than it first looked: 0.3–0.6 days, not yet proven. I report the smaller number.

The money sits in the safety margin. In a plant without a spare furnace, decoking early "to be safe" cost €130–420k per furnace-year with every model (€25–77k with a spare).

Synthetic data, method and limits shown openly. How do you decide when to decoke in your plant?

## Facts behind each claim (for replies)
- Physics vs ML: notebook Step 4, learning curves and the fresh confirmation set (ML boosting 2.4 days worse than physics at 30 campaigns, what-if).
- Hybrid gain on fresh data: 0.3–0.6 days, 95 % intervals include zero (notebook Step 4, "Fresh confirmation set only").
- €130–420k / €25–77k: notebook Step 5, Case A, furnace-limited vs spare furnace, compared with perfect foresight.
- Everything is synthetic and illustrative; the test furnace shares the physics model's chemistry.
