"""Generate pairwise distance training data."""
import sys
import typing as t
from heapq import heappop as pop
from heapq import heappush as push
from itertools import count

import cartopy.geodesic as geodesic
import numpy
from database import db
from numpy import inf
from sqlalchemy import select
from tqdm import tqdm

GEODESIC: geodesic.Geodesic = geodesic.Geodesic()


def distances_from_focus(
    source,
    database,
    tables,
    filter_sources: t.Optional[t.Set[str]] = None,
    maximum_dist: float = 5_000_000.0,
    destinations: t.Set[int] = None,
    filter_nodes: bool = False,
    pred: t.Optional = None,
) -> numpy.array:
    """Compute shortest distances on a database."""
    # Dijkstra's algorithm, adapted for our purposes from
    # networkx/algorithms/shortest_paths/weighted.html,
    # extended to be an A* implementation
    seen: t.Dict[t.Tuple[int, int], float] = {source: 0.0}
    c = count()
    # use heapq with (distance, label) tuples
    fringe: t.List[t.Tuple[float, int, t.Tuple[int, int]]] = []
    # Speeds faster than 3m/s are really rare, even on rivers, so use geodesic
    # distance with a speed of 3m/s as heuristic
    push(fringe, (0, next(c), source))

    if filter_sources:
        filter = tables["edges"].c.source.in_(filter_sources)
        if filter_nodes:
            filter &= tables["nodes"].c.node_id > 100_000_000
    else:
        if filter_nodes:
            filter = tables["nodes"].c.node_id > 100_000_000
        else:
            filter = True

    dist = t.DefaultDict(lambda: inf)

    while fringe:
        (d, _, spot) = pop(fringe)
        if dist.get(spot, numpy.inf) < d:
            continue  # already found this node, faster, through another path
        dist[spot] = d
        if d > maximum_dist:
            break
        if destinations is not None and spot in destinations:
            destinations.remove(spot)
            if not destinations:
                break

        for u, cost, data_source, lon, lat in database.execute(
            select(
                [
                    tables["edges"].c.node2,
                    tables["edges"].c.travel_time,
                    tables["edges"].c.source,
                    tables["nodes"].c.longitude,
                    tables["nodes"].c.latitude,
                ]
            )
            .select_from(
                tables["edges"].join(
                    tables["nodes"],
                    onclause=tables["edges"].c.node2 == tables["nodes"].c.node_id,
                )
            )
            .where(
                (
                    (
                        (tables["edges"].c.node1 == spot)
                        & (tables["edges"].c.node2 != spot)
                        & (tables["edges"].c.travel_time >= 0)
                    )
                    if filter is True
                    else filter
                )
                & (tables["edges"].c.node1 == spot)
                & (tables["edges"].c.node2 != spot)
            )
        ):
            if u < 100_000_000 and spot > 100_000_000:
                # Moving from land to kayak takes 3 hours
                cost += 3 * 3600
            vu_dist = dist[spot] + cost
            if u in dist and vu_dist < dist[u]:
                if pred:
                    via = f"via {pred[u]} [{dist[pred[u]]}]"
                else:
                    via = ""
                print(
                    f"Contradictory paths found: Already reached u={u} at distance {dist[u]}{via} from {data_source}, but now finding a shorter connection {vu_dist} via {spot} [{dist[spot]}]. Do you have negative weights, or a bad heuristic?"
                )
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if pred is not None:
                    pred[u] = spot, data_source
    return dist


def more_y(n):
    more_y = []
    try:
        for i in tqdm(range(n)):
            i_start = numpy.random.randint(len(nodes))
            start = node_indices[i_start]
            lonlat_start = nodes[start]
            dist = distances_from_focus(
                start, maximum_dist=500_000.0, database=DATABASE, tables=TABLES
            )
            for mid, distance in dist.items():
                if mid not in nodes:
                    # This can happen if the nodes are filtered down. They are not
                    # filtered down in the Dijkstra.
                    continue
                more_y.append(
                    [
                        distance,
                        lonlat_start[0],
                        lonlat_start[1],
                        nodes[mid][0],
                        nodes[mid][1],
                    ]
                )
    finally:
        more_y = numpy.array(more_y)
        try:
            y = numpy.load(open("distances.npy", "rb"), allow_pickle=False)
            y = numpy.vstack((y, more_y))
        except FileNotFoundError:
            y = more_y
        numpy.save(open("distances.npy", "wb"), y, allow_pickle=False)
    return len(more_y)


if __name__ == "__main__":
    DATABASE, TABLES = db(sys.argv[1])
    nodes = {
        node: (lon, lat)
        for node, lon, lat in DATABASE.execute(
            select(
                TABLES["nodes"].c.node_id,
                TABLES["nodes"].c.longitude,
                TABLES["nodes"].c.latitude,
            )
        )
    }
    node_indices = list(nodes)
    print(more_y(2000))
