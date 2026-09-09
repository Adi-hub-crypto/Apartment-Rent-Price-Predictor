# Submission notes — apartment rent prediction

**Task.** Predict monthly rent from area, district, floor and distance to the
centre. Scored by MAE.
**Result.** **MAE 776.40 ± 34.92**, 5-fold cross-validation on the training
set → score `1 − 776.40/1500` ≈ **0.482 points**.

## Data

7,000 training rows, ids 0–6,999. No missing values, no duplicate rows, ten
districts.

Two facts shaped every later decision:

**The target is strongly multiplicative.** `rent_price` has skew 6.62; `log(rent_price)`
has skew 0.38. Districts set price *levels* that differ by an order of
magnitude — median rent runs from 1,395 (industrial) to 12,951 (luxury_hills)
— and area scales that level rather than adding to it.

**Area behaves differently in each district.** Regressing log-rent on log-area
*within* districts gives a slope of 1.02 in luxury_hills (r = 0.85) but 0.01 in
central (r = 0.01). A single global area coefficient cannot represent that.

Worth flagging: `sample_submission.csv` is filled with 3193.2396185714283,
which is exactly the training mean. That constant scores MAE 1,509.7 — just
*past* the 1,500 cut-off, so the provided sample submission earns zero points.
The training median is a better constant at 1,391.0, and that is the baseline
this model should be judged against.

## Preprocessing

`district` is categorical, so it is one-hot encoded (ten levels, one held out
as the reference). Because area, distance and floor all act differently per
district, each district also gets its own slope for those three. With
`area`, `floor`, `distance_to_center` and `log(area)` as base terms, that is
**49 features** from four raw columns.

No scaling is needed for the closed-form solver; the gradient-descent solver
standardises internally.

## Model

Multiple linear regression fitted on **log(rent_price)**, then exponentiated.
Three reasons, in order of importance:

1. Log turns the multiplicative process into an additive one, which is what a
   linear model can actually fit.
2. `exp(E[log y])` is the conditional **geometric mean**, which for symmetric
   log-residuals equals the conditional **median** — and the median is exactly
   what minimises MAE. Fitting on the raw scale targets the mean, which is the
   wrong statistic for this metric and is dragged upward by the luxury_hills tail.
3. Coefficients become multipliers, so a prediction reads as
   "base × district × area × distance × floor".

Two solvers minimise the identical objective
`1/n·||Xw + b − log y||² + α/n·||w||²`:

- **OLS** — closed-form normal equations, `w = (XᵀX + αI)⁻¹Xᵀ(y − ȳ)`.
- **Gradient descent** — mini-batch (256) with momentum 0.9 and a decaying
  step size.

**Per-district calibration.** `exp(mean of log)` is the geometric mean, and the
gap between that and the true conditional median differs by district. One
multiplier per district is fitted by minimising MAE — always on *out-of-fold*
predictions, never on the rows it corrects. Factors range from 0.964
(luxury_hills) to 1.084 (industrial).

This step was validated with **nested** cross-validation, refitting the factors
inside every outer fold, so the gain is not an artefact of tuning on the
evaluation rows: 779.65 → 776.40.

## Results

5-fold cross-validation, MAE (lower is better):

| Model | MAE |
|---|---|
| Multiple linear regression — OLS + per-district calibration | **776.40 ± 34.92** |
| Multiple linear regression — gradient descent | 784.01 ± 33.24 |
| Gradient boosting (`HistGradientBoostingRegressor`) | 800.80 ± 30.66 |
| OLS without the calibration step | 779.65 ± 34.16 |
| Always predict the training median | 1,391.02 |

The linear model beating gradient boosting is not an accident: once the
district interactions are in the design matrix, the structure really is
log-linear, and the tree model spends its capacity rediscovering it.

Median absolute error is 539.63 — half of all apartments are priced within
about 540 of their true rent.

Error concentrates where the prices do:

| District | MAE | Rows |
|---|---|---|
| industrial | 492 | 595 |
| residential | 570 | 1,241 |
| green_zone | 633 | 511 |
| suburb | 648 | 993 |
| university | 652 | 821 |
| riverside | 714 | 583 |
| old_town | 773 | 528 |
| business | 813 | 768 |
| central | 1,069 | 690 |
| luxury_hills | 2,767 | 270 |

luxury_hills carries 3.6× the average error on 3.9% of the rows. Its rents run
to 66,217 against a dataset median of 2,566, so in absolute terms it dominates
MAE. Anyone wanting to improve on this figure should start there.

## Submission

`submission.csv` — columns `id,rent_price`, ids matching
`sample_submission.csv` in the same order, CRLF line endings to match the
sample byte-for-byte.

## Reproducing

```bash
pip3 install -r requirements.txt
python3 train.py             # prints every number above, writes submission.csv
python3 validate_submission.py submission.csv
```

Dependencies: pandas, numpy, scipy, scikit-learn. Fixed seeds throughout
(42 for the outer split, 7 for the calibration folds). Training takes ~2 seconds.

## Note on the reported figure

776.40 is a cross-validated estimate on training data, not a leaderboard score.
The fold spread (±34.92) suggests the test figure should land in roughly the
740–810 range, i.e. about 0.46–0.51 points.
