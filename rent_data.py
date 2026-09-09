"""Loading, feature engineering and a data report for the rent dataset."""

from __future__ import annotations

import csv
import io
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TARGET = "rent_price"
NUMERIC = ["area", "floor", "distance_to_center"]

# fixed order, so a model trained today still lines up with a file loaded later
DISTRICTS = ["business", "central", "green_zone", "industrial", "luxury_hills",
             "old_town", "residential", "riverside", "suburb", "university"]

DISTRICT_LABELS = {
    "business": "Business", "central": "Central", "green_zone": "Green zone",
    "industrial": "Industrial", "luxury_hills": "Luxury hills",
    "old_town": "Old town", "residential": "Residential", "riverside": "Riverside",
    "suburb": "Suburb", "university": "University",
}


def _unwrap(text: str) -> str:
    """Some of these files ship *doubly quoted* - every physical line is itself
    a single quoted CSV field:

        "7000,60.0,university,16,4.3"

    pandas reads that as one column, so it is unwrapped back into real columns
    here.  Normally-formatted files pass through untouched.
    """
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or max((len(r) for r in rows[:20]), default=0) != 1:
        return text

    inner = [next(csv.reader([row[0]])) for row in rows if row]
    width = len(inner[0])
    inner = [r for r in inner if len(r) == width]
    out = io.StringIO()
    csv.writer(out).writerows(inner)
    return out.getvalue()


def load_csv(source) -> pd.DataFrame:
    """Read a CSV from a path or a file-like object, handling both layouts."""
    if hasattr(source, "read"):
        text = source.read()
    else:
        with open(source, encoding="utf-8-sig") as fh:
            text = fh.read()

    df = pd.read_csv(io.StringIO(_unwrap(text)))
    df.columns = [c.strip() for c in df.columns]
    return df


def load_train(data_dir: str = DATA_DIR) -> pd.DataFrame:
    return load_csv(os.path.join(data_dir, "train.csv"))


def load_test(data_dir: str = DATA_DIR) -> pd.DataFrame:
    return load_csv(os.path.join(data_dir, "test.csv"))


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def feature_names() -> list[str]:
    names = ["area", "floor", "distance_to_center", "log_area"]
    for d in DISTRICTS[1:]:                      # first district is the reference
        names += [f"is_{d}", f"area×{d}", f"log_area×{d}",
                  f"distance×{d}", f"floor×{d}"]
    return names


def build_features(df: pd.DataFrame) -> np.ndarray:
    """District is categorical, so it becomes one-hot columns - and because the
    effect of area differs sharply by district (its log-log slope runs from
    ~0.0 in central to ~1.0 in luxury_hills), each district also gets its own
    slope for area, distance and floor."""
    area = df["area"].to_numpy(dtype=float)
    floor = df["floor"].to_numpy(dtype=float)
    dist = df["distance_to_center"].to_numpy(dtype=float)
    log_area = np.log(area)
    district = df["district"].astype(str).to_numpy()

    columns = [area, floor, dist, log_area]
    for d in DISTRICTS[1:]:
        mask = (district == d).astype(float)
        columns += [mask, area * mask, log_area * mask, dist * mask, floor * mask]
    return np.column_stack(columns)


def describe_dataset(df: pd.DataFrame) -> str:
    lines = [f"rows: {len(df)}"]
    if "id" in df.columns:
        lines.append(f"id range: {int(df['id'].min())} - {int(df['id'].max())}")

    missing = int(df.isna().sum().sum())
    lines.append(f"missing values: {missing}")
    lines.append(f"duplicate rows: {int(df.duplicated().sum())}")

    unseen = sorted(set(df["district"].astype(str)) - set(DISTRICTS))
    lines.append(f"districts: {df['district'].nunique()}"
                 + (f" (unrecognised: {unseen})" if unseen else ""))

    for col in NUMERIC:
        s = df[col]
        lines.append(f"{col:19s} min={s.min():8.2f} median={s.median():8.2f} "
                     f"max={s.max():9.2f}")

    if TARGET in df.columns:
        s = df[TARGET]
        lines.append(f"{TARGET:19s} min={s.min():8.2f} median={s.median():8.2f} "
                     f"max={s.max():9.2f}")
        lines.append(f"skew: {TARGET} {s.skew():.2f}, log({TARGET}) "
                     f"{np.log(s).skew():.2f}  -> the log scale is close to symmetric")
        lines.append(f"MAE of always predicting the median: "
                     f"{(s - s.median()).abs().mean():.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    train = load_train()
    print("== train.csv ==")
    print(describe_dataset(train))
    print(f"\nfeatures built: {build_features(train).shape[1]}")
