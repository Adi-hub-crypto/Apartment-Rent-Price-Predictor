# Apartment rent estimator

Predicts monthly rent from area, district, floor and distance to the centre.
Metric: MAE.

**Result: MAE 776.40 ± 34.92** (5-fold CV) → about **0.482 points**
(`1 − MAE/1500`). Always predicting the training median scores 1,391.

| Estimator | MAE |
|---|---|
| Multiple linear regression — OLS + per-district calibration | **776.40** |
| Multiple linear regression — gradient descent | 784.01 |
| Gradient boosting (scikit-learn) | 800.80 |

## Run

```bash
pip3 install -r requirements.txt
python3 web_app.py
```

Opens http://127.0.0.1:8000 in your browser. Standard library only — no Flask,
no CDN, no API keys, no network access. `--port N`, `--no-browser`.

```bash
python3 train.py                        # metrics + model.pkl + submission.csv
python3 validate_submission.py FILE     # check a file against the sample format
```

## The interface

**Estimate** — drag area, floor and distance, pick a district, and the price
updates live. Below it, *how that price is built*: the model is linear in log
price, so moving one input from a typical apartment to its actual value
multiplies the rent by exactly `exp(change in fitted log price)`. Those
multipliers are exact and multiply out to the estimate. Factors above 1× are
drawn to the right of the neutral line, below 1× to the left — a diverging
encoding, because the quantity has a natural midpoint.

**Submission** — drop `test.csv` on the upload area; every row is priced, the
output is checked against `data/sample_submission.csv` (same header, same ids,
same order), and `Download submission.csv` saves it in the `id,rent_price`
layout with CRLF endings.

**Model** — MAE and the competition score, the three estimators against the
do-nothing baseline, error by district and median rent by district. Bars run
from 0 to a real reference rather than to the largest value, so a small spread
is not drawn as a large one.

## How it works

Rent is multiplicative: districts set price levels an order of magnitude
apart, and area scales the level rather than adding to it. `rent_price` has
skew 6.62; `log(rent_price)` has 0.38. So the model fits log-rent and
exponentiates.

That also matches the metric. `exp(E[log y])` is the conditional geometric
mean, which for symmetric log-residuals is the conditional **median** — and
the median is what minimises MAE. Fitting on the raw scale targets the mean,
the wrong statistic here, and the luxury_hills tail drags it upward.

Area's effect differs sharply by district — the log-log slope is 1.02 in
luxury_hills but 0.01 in central — so every district gets its own slope for
area, distance and floor alongside its one-hot level. 49 features from four
raw columns.

A final per-district multiplier closes the remaining gap between the geometric
mean and the median. It is fitted only on out-of-fold predictions, and
verified with nested cross-validation: 779.65 → 776.40.

## Files

```
web_app.py              local web server (standard library only)
static/index.html       the browser interface
rent_data.py            loading, feature engineering, data report
rent_model.py           OLS / gradient descent / boosting, calibration, explanation
train.py                cross-validate, write metrics.json + submission.csv
validate_submission.py  check (and --fix) a submission against the sample
data/                   train.csv, sample_submission.csv
```

`data/test.csv` is not included — drop it in and re-run `train.py`, or upload
it in the browser interface.
