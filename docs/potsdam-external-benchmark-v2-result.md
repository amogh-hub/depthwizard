# Potsdam External Benchmark v2 — Result

## Status

External-v2 completed successfully as an execution, but the frozen DepthWizard V4 learned refiner **did not satisfy the predeclared promotion rule**.

Protocol SHA-256:

`3b1ecb2ae9b559d84cc2233af25887e05575889bdbd7b57bd2568b7e98bc7118`

The protocol was sealed before loading any ISPRS Potsdam DSM reference values. The historical external-v1 seal remains preserved separately.

## Frozen per-tile RMSE

| Tile | Calibrated DA3 RMSE | Frozen V4 RMSE | V4 - DA3 |
| --- | ---: | ---: | ---: |
| `2_10` | 4.077 m | 4.080 m | +0.003 m |
| `3_13` | 5.311 m | 5.311 m | -0.000 m |
| `5_11` | 4.912 m | 4.912 m | +0.000 m |
| `6_14` | 7.005 m | 7.018 m | +0.014 m |

All four frozen tiles evaluated successfully.

## External aggregate

- DA3 RMSE: **5.432 m**
- V4 RMSE: **5.437 m**
- RMSE improvement: **-0.09%**
- DA3 MAE: **4.106 m**
- V4 MAE: **4.116 m**
- MAE improvement: **-0.27%**
- Every frozen tile RMSE non-degrading: **NO**
- Frozen external refiner promotion: **NO**

## Scientific interpretation

This is a valid negative external result. On this frozen, independent urban airborne Potsdam benchmark, V4 is effectively tied with calibrated DA3 on most tiles but is slightly worse in aggregate and regresses on at least `2_10` and `6_14`. The predeclared promotion criteria therefore reject V4 for operational promotion.

The result must not be reframed as a learned-model win. It also does not invalidate DepthWizard's calibrated DA3 absolute-elevation pipeline: the independent benchmark establishes a calibrated DA3 aggregate RMSE of 5.432 m and MAE of 4.106 m on these four frozen Potsdam tiles under the specified Copernicus GLO-30 calibration protocol.

## Consequence

DepthWizard should keep calibrated DA3 as the external-safe geometry path while learned-refinement research continues on development data or future genuinely untouched datasets. The consumed Potsdam-v2 benchmark must not be used for post-hoc V4 tuning or threshold selection.

The local machine produced the full machine-readable report at:

`artifacts/evaluation/potsdam-external-v2/potsdam_external_report.json`

That report remains the authoritative source for detailed per-tile metrics and calibration diagnostics beyond the terminal summary recorded here.
