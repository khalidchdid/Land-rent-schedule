"""Use-based schedule for non-agricultural leases.

Rent per unit area falls as area grows, with a diminishing slope, and the
scatter grows with the rent. A log model matches both facts:

log(rent) = baseline(use) + b_area * log(1 + area) + b_duration * duration

which is the paper form

rent = K(use) * (1 + area) ** b_area * exp(b_duration * duration)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def fit_schedule(area, duration, use, rent_per_area):
    frame = pd.DataFrame({
        "log_area": np.log1p(pd.to_numeric(area, errors="coerce")),
        "duration": pd.to_numeric(duration, errors="coerce"),
        "use": pd.Series(use).astype(str),
    })
    target = pd.to_numeric(rent_per_area, errors="coerce")
    valid = (
        frame["log_area"].notna()
        & frame["duration"].notna()
        & target.notna()
        & (target > 0)
    )
    frame = frame.loc[valid]
    logged = np.log(target.loc[valid])

    prepare = ColumnTransformer([
        ("num", "passthrough", ["log_area", "duration"]),
        ("use", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["use"]),
    ])
    model = Pipeline([
        ("prepare", prepare),
        ("line", LinearRegression()),
    ])
    model.fit(frame, logged)

    line = model.named_steps["line"]
    names = list(model.named_steps["prepare"].get_feature_names_out())
    weights = dict(zip(names, line.coef_))
    encoder = model.named_steps["prepare"].named_transformers_["use"]
    reference = encoder.categories_[0][0]
    levels = {reference: float(np.exp(line.intercept_))}
    for group in encoder.categories_[0][1:]:
        shift = weights.get(f"use__use_{group}", 0.0)
        levels[group] = float(np.exp(line.intercept_ + shift))

    folds = KFold(n_splits=min(5, len(frame)), shuffle=True, random_state=0)
    held_out = float(cross_val_score(model, frame, logged, scoring="r2", cv=folds).mean())
    return {
        "level": levels,
        "area": float(weights["num__log_area"]),
        "duration": float(weights["num__duration"]),
        "held_out_r2": held_out,
    }


def rent_per_area(level, area_weight, duration_weight, area, duration):
    return level * (1.0 + area) ** area_weight * np.exp(duration_weight * duration)
