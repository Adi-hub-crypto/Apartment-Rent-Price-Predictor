"""Rent price model: multiple linear regression on a log target.

Rent is generated multiplicatively - a district sets a price level, area
scales it, distance discounts it - so the model is fitted on log(rent_price)
and exponentiated back.  Three consequences make this the right frame:

  * log(rent_price) has skew 0.38 against 6.62 for the raw price, so least
    squares is fitting something close to symmetric noise;
  * exp(E[log y]) is the conditional *geometric mean*, which for symmetric
    log-residuals is the conditional **median** - and the median is exactly
    what minimises MAE, the competition metric;
  * coefficients become multipliers, so a prediction can be read as
    "base price x district x area x distance x floor".

Two solvers fit the identical objective, plus a gradient-boosting option:

    L(w) = 1/n * ||X w + b - log y||^2  +  alpha/n * ||w||^2

  * "ols" - closed-form normal equations
  * "gd"  - mini-batch gradient descent with momentum
  * "hgb" - HistGradientBoostingRegressor, as a non-linear reference
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.model_selection import KFold

from rent_data import (DISTRICTS, TARGET, build_features, feature_names,
                       load_train)

METHODS = {
    "ols": "Multiple linear regression - OLS (normal equations)",
    "gd": "Multiple linear regression - gradient descent",
    "hgb": "Gradient boosting (scikit-learn reference)",
}

CALIBRATION_GRID = np.linspace(0.80, 1.20, 401)


# --------------------------------------------------------------------------
# the linear regression
# --------------------------------------------------------------------------
class LinearRegressor:
    """Least squares with an L2 term; closed form or gradient descent."""

    def __init__(self, method="ols", alpha=1e-6, lr=0.08, epochs=300,
                 batch_size=256, momentum=0.9, seed=0):
        self.method = method
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.seed = seed
        self.loss_history_ = []

    def _standardise(self, X, fit=False):
        if fit:
            self.mu_ = X.mean(axis=0)
            self.sigma_ = X.std(axis=0)
            self.sigma_[self.sigma_ < 1e-12] = 1.0
        return (X - self.mu_) / self.sigma_

    def _loss(self, Z, y):
        r = Z @ self.w_ + self.b_ - y
        return float(r @ r / len(y) + self.alpha * (self.w_ @ self.w_) / len(y))

    def fit(self, X, y):
        Z = self._standardise(np.asarray(X, dtype=float), fit=True)
        y = np.asarray(y, dtype=float)

        if self.method == "ols":
            # normal equations on the centred design: (Z'Z + aI) w = Z'(y - ymean)
            n, d = Z.shape
            G = Z.T @ Z
            G[np.diag_indices_from(G)] += self.alpha
            self.w_ = np.linalg.solve(G, Z.T @ (y - y.mean()))
            self.b_ = float(y.mean())
            self.loss_history_ = [self._loss(Z, y)]
        elif self.method == "gd":
            rng = np.random.default_rng(self.seed)
            n, d = Z.shape
            self.w_ = np.zeros(d)
            self.b_ = float(y.mean())
            vw = np.zeros(d)
            vb = 0.0
            self.loss_history_ = []
            for epoch in range(self.epochs):
                order = rng.permutation(n)
                lr = self.lr * (0.5 ** (epoch / (self.epochs / 3)))   # decay
                for start in range(0, n, self.batch_size):
                    idx = order[start:start + self.batch_size]
                    r = Z[idx] @ self.w_ + self.b_ - y[idx]
                    m = len(idx)
                    gw = 2 * (Z[idx].T @ r) / m + 2 * self.alpha * self.w_ / n
                    gb = 2 * r.mean()
                    vw = self.momentum * vw - lr * gw
                    vb = self.momentum * vb - lr * gb
                    self.w_ += vw
                    self.b_ += vb
                if epoch % 10 == 0 or epoch == self.epochs - 1:
                    self.loss_history_.append(self._loss(Z, y))
        else:
            raise ValueError(f"unknown method {self.method!r}")
        return self

    def predict(self, X):
        return self._standardise(np.asarray(X, dtype=float)) @ self.w_ + self.b_

    def coefficients(self):
        """Coefficients on the original (un-standardised) feature scale."""
        w = self.w_ / self.sigma_
        b = self.b_ - float(self.mu_ @ w)
        return w, b


# --------------------------------------------------------------------------
# the full model
# --------------------------------------------------------------------------
class RentModel:
    """Features -> log target -> linear fit -> exp -> per-district calibration."""

    def __init__(self, method="ols", calibrate=True, **kwargs):
        self.method = method
        self.calibrate = calibrate
        self.kwargs = kwargs
        self.factors_ = {}

    def _new_estimator(self):
        if self.method == "hgb":
            return HistGradientBoostingRegressor(random_state=0)
        return LinearRegressor(method=self.method, **self.kwargs)

    # -- fitting ----------------------------------------------------------
    def fit(self, df):
        X = build_features(df)
        y = df[TARGET].to_numpy(dtype=float)
        self.model = self._new_estimator().fit(X, np.log(y))
        self.n_features_ = X.shape[1]
        self.factors_ = self._fit_calibration(df, X, y) if self.calibrate else {}
        return self

    def _fit_calibration(self, df, X, y):
        """exp(mean of log) is the geometric mean; MAE wants the median.  The
        gap differs by district, so one multiplier per district is fitted -
        always on out-of-fold predictions, never on the rows it corrects."""
        oof = np.zeros(len(y))
        for tr, va in KFold(5, shuffle=True, random_state=7).split(y):
            est = self._new_estimator().fit(X[tr], np.log(y[tr]))
            oof[va] = np.exp(est.predict(X[va]))

        district = df["district"].astype(str).to_numpy()
        factors = {}
        for name in DISTRICTS:
            mask = district == name
            if mask.sum() < 20:                       # too few rows to calibrate
                factors[name] = 1.0
                continue
            errors = [mean_absolute_error(y[mask], c * oof[mask])
                      for c in CALIBRATION_GRID]
            factors[name] = float(CALIBRATION_GRID[int(np.argmin(errors))])
        return factors

    # -- prediction -------------------------------------------------------
    def predict(self, df):
        price = np.exp(self.model.predict(build_features(df)))
        if self.factors_:
            district = df["district"].astype(str).to_numpy()
            price = price * np.array([self.factors_.get(d, 1.0) for d in district])
        return price

    def predict_one(self, area, district, floor, distance_to_center):
        row = pd.DataFrame([{"area": float(area), "district": str(district),
                             "floor": int(floor),
                             "distance_to_center": float(distance_to_center)}])
        return float(self.predict(row)[0])

    # -- explanation ------------------------------------------------------
    def explain(self, area, district, floor, distance_to_center, reference=None):
        """Why this apartment costs what it does, as multipliers.

        Because the model is linear in log space, moving one input from a
        reference value to its actual value multiplies the price by exactly
        exp(change in the fitted log price).  The steps below therefore
        multiply out to the prediction exactly.  The split between steps
        depends on the order they are applied in - the total does not.
        """
        ref = reference or self.reference_
        steps = []

        state = dict(ref)
        price = self.predict_one(**state)
        base = price

        for key, value, label in [
            ("district", str(district), "District"),
            ("area", float(area), "Area"),
            ("distance_to_center", float(distance_to_center), "Distance to centre"),
            ("floor", int(floor), "Floor"),
        ]:
            before = state[key]
            state[key] = value
            after = self.predict_one(**state)
            steps.append({
                "label": label,
                "from": before,
                "to": value,
                "multiplier": after / price if price else 1.0,
                "price": after,
            })
            price = after

        return {"base": base, "reference": ref, "steps": steps, "price": price}

    # -- persistence ------------------------------------------------------
    def save(self, path="model.pkl"):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path="model.pkl"):
        with open(path, "rb") as fh:
            return pickle.load(fh)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(df, method="ols", calibrate=True, n_splits=5, seed=42, **kwargs):
    """K-fold MAE.  The calibration is refitted inside every fold, so the
    figure reported here is not inflated by tuning on the rows being scored."""
    y = df[TARGET].to_numpy(dtype=float)
    scores, oof = [], np.zeros(len(y))

    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(y):
        model = RentModel(method=method, calibrate=calibrate, **kwargs)
        model.fit(df.iloc[tr])
        pred = model.predict(df.iloc[va])
        oof[va] = pred
        scores.append(mean_absolute_error(y[va], pred))

    scores = np.array(scores)
    residual = oof - y
    return {
        "mae": float(scores.mean()),
        "mae_std": float(scores.std()),
        "scores": scores,
        "points": max(0.0, 1 - float(scores.mean()) / 1500),
        "median_ae": float(median_absolute_error(y, oof)),
        "oof": oof,
        "residual": residual,
        "mae_by_district": {
            d: float(np.abs(residual[df["district"].to_numpy() == d]).mean())
            for d in DISTRICTS
        },
    }


def fit_full(df, method="ols", **kwargs):
    model = RentModel(method=method, **kwargs).fit(df)
    model.reference_ = {
        "area": float(df["area"].median()),
        "district": df["district"].mode()[0],
        "floor": int(df["floor"].median()),
        "distance_to_center": float(df["distance_to_center"].median()),
    }
    return model
