# Bauer–Chernov Stage 0: immutable unfavorable continuation evidence

This is a factual summary of a preserved project outcome record. No academic evaluator, calibration runner or empirical dataset is distributed or executed here.

The academic context is Bauer and Chernov (2024), *Interest Rate Skewness and Biased Beliefs*, [Journal of Finance, DOI 10.1111/jofi.13276](https://doi.org/10.1111/jofi.13276); see the [author's publication page](https://www.michaeldbauer.com/publication/skewness/) and the [San Francisco Fed's Treasury yield-skewness description](https://www.frbsf.org/research-and-insights/data-and-indicators/treasury-yield-skewness/). These links are citations, not redistribution grants for their code/data.

The project's frozen public-data extension used 60 primary holdout predictor months, June 2021 through May 2026. Its preregistered directional conditions were a negative holdout-only augmented-regression ISK coefficient and a lower frozen augmented-model RMSE than the frozen base model.

The preserved outcome record reports:

| Primary quantity | Recorded outcome |
| --- | --- |
| Holdout ISK coefficient | +0.23163600455830943 |
| Frozen base-model RMSE | 0.5290995707023926 |
| Frozen augmented-model RMSE | 0.529977861296233 |
| Both directional conditions | Failed |
| Immutable classification | `unfavorable_continuation_evidence` |

These values are **reported historical results, not reproduced by this release**. The coefficient had the opposite required sign and augmented forecast error was slightly higher. Descriptive R-squared or uncertainty diagnostics do not rescue or reclassify the primary result.

The bounded conclusion concerns this extension's protocol, window and criteria. It does not establish that the original paper as a whole is false, and it is not an intraday trading strategy. The primary holdout is exposed and closed to rerun.

The historical B.4 18-cell calibration comparison remains unfinished. No B.4 cell, primary holdout, sensitivity study or other historical scientific run was executed during candidate preparation. Any future exploratory work would need separate authority and must leave the primary result unchanged.

Journal replication archives, author-supplied R/RData, private correspondence, yield/skewness workbooks and provider-derived market inputs are excluded. Source availability does not grant general rights to redistribute them. This release offers a negative-result reporting example rather than an empirical replication kit.
