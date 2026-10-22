"""Run both schedules on invented leases. No observed grid is used."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pricing.activity import fit_schedule as fit_activity
from pricing.agricultural import fit_schedule as fit_agricultural
from pricing.indices import block_index, proximity


def agricultural_sample(rows=240, seed=0):
    rng = np.random.default_rng(seed)
    place = rng.choice(["zone-1", "zone-2", "zone-3"], rows)
    start = rng.normal(0, 1, rows)
    length = rng.uniform(1, 20, rows)
    water = rng.normal(0, 1, rows)
    soil = rng.choice(["soil-a", "soil-b", "soil-c"], rows)
    road = proximity(pd.Series(rng.uniform(0, 30, rows)))
    track = proximity(pd.Series(rng.uniform(0, 15, rows)))
    frame = pd.DataFrame({
        "place": place,
        "start": start,
        "length": length,
        "water": water,
        "soil": soil,
        "road": road,
        "track": track,
    })
    frame["contract"] = block_index(frame, numeric=("start", "length"), name="contract")
    frame["hydro"] = block_index(frame, numeric=("water",), categorical=("soil",), name="hydro")
    frame["access"] = block_index(frame, numeric=("road", "track"), name="access")
    baseline = pd.Series(place).map({"zone-1": 3.0, "zone-2": 4.0, "zone-3": 5.0}).to_numpy()
    frame["rent"] = (
        baseline
        + 0.8 * frame["contract"]
        + 0.5 * frame["hydro"]
        + 0.4 * frame["access"]
        + rng.normal(0, 0.4, rows)
    )
    return frame


def activity_sample(rows=240, seed=1):
    rng = np.random.default_rng(seed)
    use = rng.choice(["use-a", "use-b", "use-c"], rows)
    area = rng.uniform(0.2, 30, rows)
    duration = rng.uniform(1, 25, rows)
    level = pd.Series(use).map({"use-a": 80.0, "use-b": 140.0, "use-c": 40.0}).to_numpy()
    rent = level * (1.0 + area) ** -0.6 * np.exp(0.02 * duration)
    rent = rent * rng.lognormal(0, 0.08, rows)
    return pd.DataFrame({
        "use": use,
        "area": area,
        "duration": duration,
        "rent_per_area": rent,
    })


def main():
    fields = agricultural_sample()
    agricultural = fit_agricultural(fields, place="place", rent="rent")
    print("agricultural schedule")
    print(
        "  rent = local baseline"
        f" + {agricultural['contract']:.2f} contract"
        f" + {agricultural['hydro']:.2f} hydro"
        f" + {agricultural['access']:.2f} access"
    )
    print(f"  held-out R2 on the transformed rent: {agricultural['held_out_r2']:.2f}")

    uses = activity_sample()
    activity = fit_activity(uses["area"], uses["duration"], uses["use"], uses["rent_per_area"])
    print("activity schedule")
    print(
        "  log(rent per area) = use baseline"
        f" + {activity['area']:.2f} log(1 + area)"
        f" + {activity['duration']:.3f} duration"
    )
    print(f"  held-out R2 on the log rent: {activity['held_out_r2']:.2f}")


if __name__ == "__main__":
    main()
