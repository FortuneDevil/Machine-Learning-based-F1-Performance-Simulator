"""
fetch_circuits.py

Download real circuit outlines for the animated race view.

Source: the open "f1-circuits" dataset (github.com/bacinger/f1-circuits),
a GeoJSON file with one outline per circuit. Each outline is converted from
longitude/latitude to flat x/y coordinates and saved to data/circuits.json.

This step is optional. Without the file the animation falls back to a
generic loop.

Run
---
python fetch_circuits.py
"""

from __future__ import annotations

import json
import sys
import urllib.request

import numpy as np

from common import DATA_DIR

SOURCE_URL = (
    "https://raw.githubusercontent.com/bacinger/f1-circuits/master/"
    "f1-circuits.geojson"
)

CIRCUITS_PATH = DATA_DIR / "circuits.json"

# Circuit name used in this project -> id in the outline dataset.
CIRCUIT_IDS = {
    "Austin": "us-2012",
    "Baku": "az-2016",
    "Barcelona": "es-1991",
    "Budapest": "hu-1986",
    "Imola": "it-1953",
    "Jeddah": "sa-2021",
    "Las Vegas": "us-2023",
    "Le Castellet": "fr-1969",
    "Lusail": "qa-2004",
    "Marina Bay": "sg-2008",
    "Melbourne": "au-1953",
    "Mexico City": "mx-1962",
    "Miami": "us-2022",
    "Monaco": "mc-1929",
    "Montréal": "ca-1978",
    "Monza": "it-1922",
    "Sakhir": "bh-2002",
    "Shanghai": "cn-2004",
    "Silverstone": "gb-1948",
    "Spa-Francorchamps": "be-1925",
    "Spielberg": "at-1969",
    "Suzuka": "jp-1962",
    "São Paulo": "br-1940",
    "Yas Island": "ae-2009",
    "Zandvoort": "nl-1948",
}


def to_flat_xy(coordinates: list[list[float]]) -> list[list[float]]:
    """
    Longitude/latitude to x/y in a unit box, north up.

    A circuit spans a few kilometres, so scaling longitude by the cosine of
    the latitude is an accurate enough projection. The result is centred
    and scaled so the longer side runs from -1 to 1; y grows downwards to
    match screen coordinates.
    """
    points = np.asarray(coordinates, dtype=float)[:, :2]
    lon, lat = points[:, 0], points[:, 1]

    x = (lon - lon.mean()) * np.cos(np.radians(lat.mean()))
    y = -(lat - lat.mean())

    x -= (x.max() + x.min()) / 2.0
    y -= (y.max() + y.min()) / 2.0
    scale = max(x.max() - x.min(), y.max() - y.min()) / 2.0

    return np.round(np.column_stack([x, y]) / scale, 4).tolist()


def main() -> int:
    try:
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            geojson = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"ERROR: could not download circuit outlines: {exc}")
        return 1

    by_id = {
        feature["properties"]["id"]: feature["geometry"]["coordinates"]
        for feature in geojson["features"]
    }

    circuits = {}
    for circuit, circuit_id in CIRCUIT_IDS.items():
        if circuit_id not in by_id:
            print(f"missing outline for {circuit} ({circuit_id})")
            continue
        circuits[circuit] = to_flat_xy(by_id[circuit_id])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CIRCUITS_PATH.write_text(
        json.dumps(circuits, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved {len(circuits)} circuit outlines -> {CIRCUITS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
