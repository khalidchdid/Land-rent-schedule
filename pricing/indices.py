"""One index per homogeneous block of columns.

A single principal component on mixed columns lets a few attributes
dominate. Contract terms, access, and soil and water are reduced separately.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def proximity(distance):
    """Turn a distance into a score in (0, 1]. A missing road scores 0."""
    distance = pd.to_numeric(distance, errors="coerce").clip(lower=0)
    return (1.0 / (1.0 + distance)).fillna(0.0)


def _encoder():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def block_index(frame, numeric=(), categorical=(), target=None, name="index"):
    """First principal component of one block, oriented with the rent if given."""
    numeric = [column for column in numeric if column in frame.columns]
    categorical = [column for column in categorical if column in frame.columns]
    if not numeric and not categorical:
        return pd.Series(0.0, index=frame.index, name=name)

    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", _encoder()),
            ]),
            categorical,
        ))

    model = Pipeline([
        ("prepare", ColumnTransformer(transformers)),
        ("pca", PCA(n_components=1, random_state=0)),
    ])
    scores = model.fit_transform(frame[numeric + categorical]).ravel()
    index = pd.Series(scores, index=frame.index, name=name)
    if target is not None:
        correlation = np.corrcoef(
            np.nan_to_num(index.to_numpy(float)),
            np.nan_to_num(np.asarray(target, float)),
        )[0, 1]
        if not np.isfinite(correlation) or correlation < 0:
            index = -index
            index.name = name
    return index
