# Simplifying a land-rent schedule

Observed leases do not make a usable price grid. Parcels differ in contract, access, soil, and water, and a cell for every combination is too large to read or maintain. This project replaces that grid with a short formula. The formula is a schedule: a local or use-specific starting level, plus a few corrections. No lease table is included.

There are two schedules, because the two kinds of land do not depend on the same things.

## Agricultural land

A single principal-component analysis on every column at once gives extreme weights to a few attributes and predicts poorly. The columns are not one homogeneous block. They are split first:

- **Contract:** start, end, and length. Missing numbers are filled with the column median, then standardized.
- **Access:** distance to each class of road, converted to proximity `1 / (1 + distance)`. A missing road scores 0.
- **Soil and water:** rainfall or a water-table indicator, plus soil type.

Each block is reduced to its first component. The sign is chosen so a higher index goes with a higher rent. The schedule is then ordinary least squares:

```text
rent = baseline(place) + b1 * contract + b2 * hydro + b3 * access
```

The baseline is the only term that changes from place to place. The slopes stay common, which keeps the formula short. When the rent is skewed, the same line is fit to `log(1 + rent)` and mapped back with `exp(x) - 1`. A robust regression is the alternative if a few leases would otherwise pull the fit. Quality is the coefficient of determination on the training rows and in k-fold cross-validation. Lease rents are noisy, so this schedule is meant to explain the stable part of the variation, not to appraise one parcel.

## Other leased land

Identifiers and repeated columns are dropped, and leases are split by use. What remains is area and contract length. Rent per unit area falls as area grows, and the fall flattens. The scatter also grows with the rent, so relative errors matter more than absolute ones. Both are handled by

```text
log(rent per area) = baseline(use) + b_area * log(1 + area) + b_duration * duration
```

On the original scale that is

```text
rent per area = K(use) * (1 + area) ** b_area * exp(b_duration * duration)
```

`b_area` is an elasticity. `K(use)` is the starting level of that use. Held-out fit on the log rent is the check. Once use, area, and duration are isolated, this formula carries most of the systematic variation.

## Run

The example invents both kinds of lease and recovers the shape of each formula. It does not read a data file.

```bash
python examples/synthetic.py
```

Dependencies are listed in `requirements.txt`.
