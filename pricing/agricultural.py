"""Agricultural schedule: a local baseline plus three block indices.

rent = baseline(place) + b_contract * contract + b_hydro * hydro + b_access * access

With a skewed rent, the same line is fit to log(1 + rent) and mapped back.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


INDEXES = ("contract", "hydro", "access")


def fit_schedule(frame, place, rent, log1p=False):
    work = frame[list(INDEXES) + [place]].copy()
    target = pd.to_numeric(frame[rent], errors="coerce")
    valid = target.notna() & np.isfinite(target)
    work = work.loc[valid]
    target = target.loc[valid]
    if log1p:
        fitted_target = np.log1p(target)
    else:
        fitted_target = target

    prepare = ColumnTransformer([
        ("index", "passthrough", list(INDEXES)),
        ("place", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), [place]),
    ])
    model = Pipeline([
        ("prepare", prepare),
        ("line", LinearRegression()),
    ])
    model.fit(work, fitted_target)

    line = model.named_steps["line"]
    names = list(model.named_steps["prepare"].get_feature_names_out())
    weights = dict(zip(names, line.coef_))
    encoder = model.named_steps["prepare"].named_transformers_["place"]
    reference = encoder.categories_[0][0]
    baselines = {reference: float(line.intercept_)}
    for group in encoder.categories_[0][1:]:
        baselines[group] = float(line.intercept_ + weights.get(f"place__{place}_{group}", 0.0))

    folds = KFold(n_splits=min(5, len(work)), shuffle=True, random_state=0)
    held_out = float(cross_val_score(model, work, fitted_target, scoring="r2", cv=folds).mean())
    return {
        "baselines": baselines,
        "contract": float(weights["index__contract"]),
        "hydro": float(weights["index__hydro"]),
        "access": float(weights["index__access"]),
        "log1p": log1p,
        "held_out_r2": held_out,
    }
