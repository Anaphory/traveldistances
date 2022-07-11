from tqdm import tqdm
import numpy

import sqlalchemy
from sqlalchemy.dialects.sqlite import insert

import shapely.geometry as sgeom

from earth import GEODESIC
from database import db
from by_river import KAYAK_SPEED  # in m/s

from earth import DEFINITELY_INLAND

DATABASE, TABLES = db()


def distance_by_sea(definitely_inland, skip: bool = True) -> None:
    query = sqlalchemy.select(
        [
            TABLES["nodes"].c.node_id,
            TABLES["nodes"].c.longitude,
            TABLES["nodes"].c.latitude,
        ]
    ).where(TABLES["nodes"].c.coastal)
    for node0, lon0, lat0 in tqdm(DATABASE.execute(query).fetchall()):
        if (
            skip
            and DATABASE.execute(
                sqlalchemy.select([TABLES["edges"].c.node1]).where(
                    TABLES["edges"].c.node1 == node0,
                    TABLES["edges"].c.travel_time != None,
                    TABLES["edges"].c.source == "sea",
                )
            ).fetchall()
        ):
            continue
        values = []
        for node1, lon1, lat1 in DATABASE.execute(
            query.where(
                (
                    (lat0 - TABLES["nodes"].c.latitude)
                    * (lat0 - TABLES["nodes"].c.latitude)
                )
                + (
                    (lon0 - TABLES["nodes"].c.longitude)
                    * (lon0 - TABLES["nodes"].c.longitude)
                )
                * numpy.cos(lat0 * numpy.pi / 180) ** 2
                < 9
            )
        ).fetchall():
            d = GEODESIC.inverse((lon0, lat0), (lon1, lat1))[0, 0]
            if d > 300_000:  # meters
                continue
            if definitely_inland.intersects(
                sgeom.LineString([(lon0, lat0), (lon1, lat1)])
            ):
                continue
            t = d / KAYAK_SPEED
            values.append(
                {
                    "node1": node0,
                    "node2": node1,
                    "source": "sea",
                    "travel_time": t,
                    "flat_distance": d,
                }
            )
        if values:
            DATABASE.execute(insert(TABLES["edges"]).values(values))

if __name__ == "__main__":
    import sys
    DATABASE, TABLES = db(
    file=sys.argv[1]
)
    distance_by_sea(DEFINITELY_INLAND)
