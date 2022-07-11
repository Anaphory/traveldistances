"""Aggregate distance inputs into a distance matrix."""
import typing as t
from itertools import count
from heapq import heappush as push, heappop as pop

import numpy
from tqdm import tqdm
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert

from h3.api import basic_int as h3

import rasterio
from raster_data import (
    boundingbox_from_tile,
    Tile,
    RowCol,
    fmt,
    unfmt
)
from database import db
from earth import LonLat

DATABASE, TABLES = db()
RESOLUTION = 5


def load_distances(tile: Tile):
    fname = f"distances-{fmt(tile)}.tif"

    distance_raster = rasterio.open(fname)
    return {
        (-1, 0): distance_raster.read(1),
        (-1, 1): distance_raster.read(2),
        (0, 1): distance_raster.read(3),
        (1, 1): distance_raster.read(4),
        (1, 0): distance_raster.read(5),
        (1, -1): distance_raster.read(6),
        (0, -1): distance_raster.read(7),
        (-1, -1): distance_raster.read(8),
    }, distance_raster.transform


def distances_from_focus(
    source: t.Tuple[int, int],
    destinations: t.Optional[t.Set[RowCol]],
    distance_by_direction: t.Dict[RowCol, numpy.ndarray],
    pred: t.Optional[t.Dict[RowCol, RowCol]] = None,
) -> numpy.ndarray:
    # Dijkstra's algorithm, adapted for our purposes from
    # networkx/algorithms/shortest_paths/weighted.html
    d = distance_by_direction[0, 1]
    dist: numpy.array = numpy.full(d.shape, numpy.inf, dtype=float)
    seen: t.Dict[t.Tuple[int, int], float] = {source: 0.0}
    c = count()
    # use heapq with (distance, label) tuples
    fringe: t.List[t.Tuple[float, int, t.Tuple[int, int]]] = []
    push(fringe, (0, next(c), source))

    def moore_neighbors(
        r0: int, c0: int
    ) -> t.Iterable[t.Tuple[t.Tuple[int, int], float]]:
        for (r, c), d in distance_by_direction.items():
            r1, c1 = r0 + r, c0 + c
            if 0 <= r1 < d.shape[0] and 0 <= c1 < d.shape[1]:
                yield (r1, c1), d[r0, c0]

    for u, cost in moore_neighbors(*source):
        if not numpy.isfinite(cost):
            r0, c0 = source[0] - u[0], source[1] - u[1]
            # We don't have the actual distance away from the river, so we
            # assume the terrain is somewhat homogenous around it.
            cost = max(
                distance_by_direction[r0, c0][u], distance_by_direction[-r0, -c0][u]
            )
            if numpy.isfinite(cost):
                push(fringe, (cost, next(c), u))

    while fringe:
        (d, _, spot) = pop(fringe)
        if numpy.isfinite(dist[spot]):
            continue  # already searched this node.
        dist[spot] = d
        if destinations is not None and spot in destinations:
            destinations.remove(spot)
            if not destinations:
                break

        for u, cost in moore_neighbors(*spot):
            vu_dist = dist[spot] + cost
            if numpy.isfinite(dist[u]):
                if vu_dist < dist[u]:
                    raise ValueError(
                        "Contradictory paths found. Do you have negative weights?"
                    )
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if pred is not None:
                    pred[u] = spot
    return dist


def distances_trimmed(source, points, distance_by_direction, transform):
    rmin = min(r for r, c in points) - 2
    rmax = max(r for r, c in points) + 3
    cmin = min(c for r, c in points) - 2
    cmax = max(c for r, c in points) + 3
    points = [(r - rmin, c - cmin) for r, c in points]

    r0, c0 = source[0] - rmin, source[1] - cmin

    dist = {
        (n, e): d[rmin:rmax, cmin:cmax] for (n, e), d in distance_by_direction.items()
    }

    return rmin, cmin, distances_from_focus((r0, c0), points, dist, pred=None)


def voronoi_and_neighbor_distances(tile, skip_existing=True):
    west, south, east, north = boundingbox_from_tile(tile)
    nodes = [
        (node, (lon, lat))
        for node, lon, lat in DATABASE.execute(
            select(
                TABLES["nodes"].c.node_id,
                TABLES["nodes"].c.longitude,
                TABLES["nodes"].c.latitude,
            ).where(
                TABLES["nodes"].c.longitude >= west,
                TABLES["nodes"].c.latitude >= south,
                TABLES["nodes"].c.longitude < east,
                TABLES["nodes"].c.latitude < north,
            )
        )
    ]

    distance_by_direction, transform = load_distances(tile)
    river_wading = rasterio.open(f"x-rivers-{fmt(tile)}.tif").read(1)
    for v in distance_by_direction.values():
        v += river_wading

    fname_v = f"voronoi-{fmt(tile)}.tif"
    fname_d = f"min_distances-{fmt(tile)}.tif"
    try:
        voronoi_allocation = rasterio.open(fname_v).read(1)
        min_distances = rasterio.open(fname_d).read(1)
    except rasterio.errors.RasterioIOError:
        height, width = distance_by_direction[1, 1].shape
        voronoi_allocation = numpy.zeros((height, width), dtype=numpy.uint64)
        min_distances = numpy.full((height, width), numpy.inf, dtype=numpy.float32)

    def rowcol(lonlat: LonLat):
        nlon, nlat = lonlat
        ncol, nrow = ~transform * (nlon, nlat)
        if ncol > 8202:
            ncol, nrow = ~transform * (nlon - 360, nlat)
        if ncol < -2:
            ncol, nrow = ~transform * (nlon + 360, nlat)
        return (round(nrow), round(ncol))

    try:
        for node, (lon, lat) in tqdm(nodes):
            if node > 100000000:
                # Node is an h3 index, not a river reach index
                environment = h3.k_ring(node, 2)
            else:
                # Node is a river reach end point
                environment = h3.k_ring(h3.geo_to_h3(lat, lon, 5), 2)
                environment.add(node)

            extremities = DATABASE.execute(
                select(
                    TABLES["nodes"].c.longitude,
                    TABLES["nodes"].c.latitude,
                    TABLES["nodes"].c.h3longitude,
                    TABLES["nodes"].c.h3latitude,
                ).where(
                    TABLES["nodes"].c.longitude != None,
                    TABLES["nodes"].c.latitude != None,
                    TABLES["nodes"].c.node_id.in_(environment),
                )
            ).fetchall()

            x, y, hx, hy = zip(*extremities)

            source = rowcol((lon, lat))
            nnodes = []
            nrowcol = []
            neighbors = DATABASE.execute(
                select(
                    TABLES["nodes"].c.node_id,
                    TABLES["nodes"].c.longitude,
                    TABLES["nodes"].c.latitude,
                ).where(
                    min(x) <= TABLES["nodes"].c.longitude,
                    TABLES["nodes"].c.longitude <= max(x),
                    min(y) <= TABLES["nodes"].c.latitude,
                    TABLES["nodes"].c.latitude <= max(y),
                )
            ).fetchall() + [
                # Add the hexagon neighbors as virtual targets, to have a
                # uniform breadth
                (None, hlon, hlat)
                for hlon, hlat in zip(hx, hy)
                if hlon is not None and hlat is not None
            ]

            if skip_existing:
                already_known = {
                    n
                    for n, in DATABASE.execute(
                        select(TABLES["edges"].c.node2).where(
                            TABLES["edges"].c.node1 == node,
                            TABLES["edges"].c.source == "grid",
                        )
                    )
                }
                to_be_known = {n for n, _, _ in neighbors} - {None}
                if already_known >= to_be_known and (
                    (node < 100000000)
                    or (
                        min_distances[source] == 0.0
                        and voronoi_allocation[source] == node
                    )
                ):
                    continue
                # print(already_known - to_be_known, to_be_known - already_known)

            for nnode, nlon, nlat in neighbors:
                row, col = rowcol((nlon, nlat))
                if (
                    0 <= row < distance_by_direction[1, 1].shape[0]
                    and 0 <= col < distance_by_direction[1, 1].shape[1]
                ):
                    nrowcol.append((row, col))
                    nnodes.append(nnode)

            if not nnodes:
                continue
            rmin, cmin, array = distances_trimmed(
                source, nrowcol, distance_by_direction, transform
            )

            if node < 100000000:
                array -= river_wading[source]

            values = [
                {
                    "node1": node,
                    "node2": nnode,
                    "source": "grid",
                    "travel_time": array[nr - rmin, nc - cmin],
                }
                for nnode, (nr, nc) in zip(nnodes, nrowcol)
                if nnode is not None
            ]
            DATABASE.execute(
                insert(TABLES["edges"]).values(values).on_conflict_do_nothing()
            )

            if node > 100000000:
                # Node is an h3 index, not a river reach index, so update the voronoi shapes around it
                rows, cols = array.shape
                voronoi_allocation[rmin : rmin + rows, cmin : cmin + cols][
                    min_distances[rmin : rmin + rows, cmin : cmin + cols] >= array
                ] = node
                min_distances[rmin : rmin + rows, cmin : cmin + cols] = numpy.fmin(
                    min_distances[rmin : rmin + rows, cmin : cmin + cols], array
                )
    finally:
        profile = rasterio.profiles.DefaultGTiffProfile()
        profile["height"] = voronoi_allocation.shape[0]
        profile["width"] = voronoi_allocation.shape[1]
        profile["transform"] = transform
        del profile["dtype"]
        profile["count"] = 1

        with rasterio.open(
            fname_v,
            "w",
            dtype=numpy.uint32,
            **profile,
        ) as dst:
            dst.write(voronoi_allocation, 1)
        with rasterio.open(
            fname_d,
            "w",
            dtype=numpy.float32,
            **profile,
        ) as dst:
            dst.write(min_distances, 1)


if __name__ == "__main__":
    import sys

    DATABASE, TABLES = db(sys.argv[1])
    ns, lat, ew, lon = unfmt(sys.argv[2])
    print("Calculating local distances for", (ns, lat, ew, lon))

    voronoi_and_neighbor_distances((ns, lat, ew, lon), skip_existing=False)
