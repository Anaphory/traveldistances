"""What we know about Earth."""

import json
import pickle
import typing as t

import cartopy.geodesic as geodesic
import cartopy.io.shapereader as shpreader
from h3.api import basic_int as h3
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.geometry import mapping

LonLat = t.Tuple[float, float]

# Define some constants
GEODESIC: geodesic.Geodesic = geodesic.Geodesic()


# Load LAND multipolygon from file
try:
    with open("./LAND.jar", "rb") as poly_file:
        LAND = pickle.load(poly_file)
except FileNotFoundError:
    LAND = unary_union(
        [
            record.geometry
            for record in shpreader.Reader(
                shpreader.natural_earth(
                    resolution="10m", category="physical", name="land"
                )
            ).records()
            if record.attributes.get("featurecla") != "Null island"
        ]
    ).difference(
        unary_union(
            list(
                shpreader.Reader(
                    shpreader.natural_earth(
                        resolution="10m", category="physical", name="lakes"
                    )
                ).geometries()
            )
        )
    )
    with open("./LAND.jar", "wb") as poly_file:
        pickle.dump(LAND, poly_file, pickle.HIGHEST_PROTOCOL)

PLAND = prep(LAND)
DEFINITELY_INLAND = prep(LAND.buffer(0.03))

H3Index = int

# ================
# Process hexagons
# ================
def find_coast_hexagons() -> t.Set[H3Index]:
    coastal = set()
    # Coast resolution is 10m and coasts are not known to be straight, so we
    # can expect that every coastal hexagon contains at least one of the
    # coastline polygon coordinates.
    for geom in LAND.boundary.geoms:
        for x, y in geom.coords:
            coastal.add(h3.geo_to_h3(y, x, 5))
    return coastal


try:
    COAST = set(json.load(open("COAST.json")))
except (FileNotFoundError, json.JSONDecodeError):
    COAST = find_coast_hexagons()
    json.dump(list(COAST), open("COAST.json", "w"))


def find_land_hexagons():
    hexagons = set()
    for poly in LAND.geoms:
        d = mapping(poly)
        hexagons |= h3.polyfill_geojson(d, 5)
    return hexagons | COAST


try:
    ALL = set(json.load(open("ALL.json")))
except (FileNotFoundError, json.JSONDecodeError):
    ALL = find_land_hexagons()
    json.dump(list(ALL), open("ALL.json", "w"))
