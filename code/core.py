#!/home/cluster/gkaipi/.pyenv/shims/python
import sys

import numpy
import rasterio
from database import db
from earth import ALL
from earth import COAST
from h3.api import basic_int as h3
from main import distances_from_focus
from main import load_distances
from raster_data import boundingbox_from_tile
from raster_data import fmt
from raster_data import H3Index
from raster_data import Tile
from raster_data import unfmt
from sqlalchemy.dialects.sqlite import insert
from tqdm import tqdm


def core_point(hexbin, distance_by_direction, wading_time, transform):
    """Compute the ‘core location’ of the h3 cell `hexbin`.

    Given a local hiking tistance rarster (`distance_by_direction`) and its
    wading times (`wading`) in addition to the affine transformation mapping
    col/row indices to latitudes/longitudes, compute the ‘core location’.

    The core location is the point with the best closeness centrality given the
    corners of the h3 hexagon cell and the center points of the hexagon edges,
    i.e. the point where the sum of distances to all these points is the
    shortest.

    """
    minlon, _ = transform * (0, 0)
    maxlon, _ = transform * distance_by_direction[(0, 1)].shape

    def rowcol(latlon):
        lat, lon = latlon
        if lon > maxlon:
            lon = lon - 360
        if lon < minlon:
            lon = lon + 360
        col, row = ~transform * (lon, lat)
        return round(row), round(col)

    points = [rowcol(latlon) for latlon in h3.h3_to_geo_boundary(hexbin)]
    rmin = min(r for r, c in points) - 1
    rmax = max(r for r, c in points) + 2
    cmin = min(c for r, c in points) - 1
    cmax = max(c for r, c in points) + 2
    points = [(r - rmin, c - cmin) for r, c in points]

    dist = {
        (n, e): d[rmin:rmax, cmin:cmax] for (n, e), d in distance_by_direction.items()
    }
    dist[0, -1] = dist[0, 1] = dist[0, 1] + dist[0, -1]
    dist[-1, -1] = dist[1, 1] = dist[1, 1] + dist[-1, -1]
    dist[1, -1] = dist[-1, 1] = dist[-1, 1] + dist[1, -1]
    dist[-1, 0] = dist[1, 0] = dist[1, 0] + dist[-1, 0]

    border = points[:]
    for i in range(-1, len(points) - 1):
        r0, c0 = points[i]
        r1, c1 = points[i + 1]
        border.append(
            (
                round(0.5 * (r0 + r1)),
                round(0.5 * (c0 + c1)),
            )
        )

    all_dist = wading_time[rmin:rmax, cmin:cmax] * 12
    # There are 12 boundary points, and we *really* want to punish core points
    # ending up on rivers. The wading penalty is naturally added once for each
    # boundary point, but this adds it a second time.
    for r0, c0 in border:
        all_dist = all_dist + distances_from_focus((r0, c0), None, dist)

    (r0, c0) = numpy.unravel_index(numpy.argmin(all_dist), all_dist.shape)
    return (r0 + rmin, c0 + cmin)


def tile_core_points(tile: Tile):
    """Add the actual core location for each hex in the tile to the database.

    Iterate through all h3 hexes in the given tile, and compute their core
    locations.

    """
    west, south, east, north = boundingbox_from_tile(tile)

    def tile_contains(hexagon: H3Index):
        lat, lon = h3.h3_to_geo(hexagon)
        return (west < lon < east) and (south < lat < north)

    hexes_by_tile = {hex for hex in ALL if tile_contains(hex)}

    distance_by_direction, transform = load_distances(tile)
    rivers = rasterio.open(f"x-rivers-{fmt(tile)}.tif")
    wading_time = rivers.read(1)
    for v in distance_by_direction.values():
        v += wading_time
    wading_time

    values = []
    for hexagon in tqdm(hexes_by_tile):
        hlat, hlon = h3.h3_to_geo(hexagon)
        row, col = core_point(hexagon, distance_by_direction, wading_time, transform)
        lon, lat = transform * (col, row)
        values.append(
            {
                "node_id": hexagon,
                "longitude": lon,
                "latitude": lat,
                "h3longitude": hlon,
                "h3latitude": hlat,
                "coastal": (hexagon in COAST),
            }
        )
    return values


if __name__ == "__main__":
    database, tables = db(sys.argv[1])
    ns, lat, ew, lon = unfmt(sys.argv[2])
    print("Finding core points in", fmt((ns, lat, ew, lon)), "…")
    values = tile_core_points((ns, lat, ew, lon))

    if values:
        database.execute(insert(tables["nodes"]).values(values))
