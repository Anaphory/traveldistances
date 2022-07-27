import json
import typing as t
import zipfile
from pathlib import Path

import numpy
import rasterio.errors
import shapefile
import shapely.geometry as sgeom
from database import db
from h3.api import basic_int as h3
from more_itertools import windowed
from raster_data import fmt
from raster_data import gmted_tile
from raster_data import unfmt
from sqlalchemy.dialects.sqlite import insert
from tqdm import tqdm


class RiverNetwork:
    cache = None

    @classmethod
    def reaches(cls):
        """
        >>> rivers = RiverNetwork.reaches()
        >>> rivers.fields
        [('DeletionFlag', 'C', 1, 0), ['Reach_ID', 'N', 10, 0], ['Next_down', 'N', 10, 0], ['Length_km', 'F', 13, 11], ['Log_Q_avg', 'F', 13, 11], ['Log_Q_var', 'F', 13, 11], ['Class_hydr', 'N', 10, 0], ['Temp_min', 'F', 13, 11], ['CMI_indx', 'F', 13, 11], ['Log_elev', 'F', 13, 11], ['Class_phys', 'N', 10, 0], ['Lake_wet', 'N', 10, 0], ['Stream_pow', 'F', 13, 11], ['Class_geom', 'N', 10, 0], ['Reach_type', 'N', 10, 0], ['Kmeans_30', 'N', 10, 0]]
        >>> rivers.numRecords
        ...

        """
        if cls.cache is not None:
            return cls.cache
        zipshape = zipfile.ZipFile(
            (Path(__file__).parent / "../rivers/GloRiC_v10_shapefile.zip").open("rb")
        )
        shape = shapefile.Reader(
            shp=zipshape.open("GloRiC_v10_shapefile/GloRiC_v10.shp"),
            shx=zipshape.open("GloRiC_v10_shapefile/GloRiC_v10.shx"),
            dbf=zipshape.open("GloRiC_v10_shapefile/GloRiC_v10.dbf"),
            encoding="utf-8",
        )
        cls.cache = shape
        return shape

    def __init__(self, mask: t.Optional[sgeom.Polygon] = None):
        """

        >>> eco = RiverNetwork()

        """
        self.shp = self.reaches()


RESOLUTION = 5


def neighbors(hex) -> t.Set:
    this, neighbors = h3.k_ring_distances(hex, 1)
    return neighbors


def intersection(xa, ya, xb, yb, xl, yl, xr, yr):
    """

    >>> intersection(0, 1, 3, 1, 2, 3, 2, 0)
    (0.6666666666666666, 0.3333333333333333)
    """

    if xl != xr:
        t = (xl * (ya - yb) + xa * (yb - yl) + xb * (yl - ya)) / (
            (xl - xr) * (ya - yb) + xa * (yr - yl) + xb * (yl - yr)
        )
    else:
        t = (yl * (xa - xb) + ya * (xb - xl) + yb * (xl - xa)) / (
            (yl - yr) * (xa - xb) + ya * (xr - xl) + yb * (xl - xr)
        )

    if xb != xa:
        u = (xb * (yl - yr) + xl * (yr - yb) + xr * (yb - yl)) / (
            (xb - xa) * (yl - yr) + xl * (ya - yb) + xr * (yb - ya)
        )
    else:
        u = (yb * (xl - xr) + yl * (xr - xb) + yr * (xb - xl)) / (
            (yb - ya) * (xl - xr) + yl * (xa - xb) + yr * (xb - xa)
        )

    return t, u


def find_hexes_crossed(
    x0: float, y0: float, x1: float, y1: float, length_start: float, length_end: float
) -> t.List[t.Tuple]:
    """Find all hexes that a line segment crosses.

    By bisecting the line segment from (x0, y0) to (x1, y1), find all h3 hexes
    that the line crosses.

    >>> find_hexes_crossed(-10, -10, -9, -9)
    ...

    """
    start_hex = h3.geo_to_h3(y0, x0, RESOLUTION)
    end_hex = h3.geo_to_h3(y1, x1, RESOLUTION)
    if start_hex == end_hex:
        # Hexes are convex, so the line segment must run entirely within.
        return [(start_hex, length_start, length_end)]
    elif end_hex in neighbors(start_hex):
        # Check whether the line segment runs through the left neighbor,
        # through the right neighbor, or through the shared boundary of start
        # and end.
        nr, nl = neighbors(end_hex) & neighbors(start_hex)
        ya, xa = h3.h3_to_geo(start_hex)
        yb, xb = h3.h3_to_geo(end_hex)
        yr, xr = h3.h3_to_geo(nr)
        yl, xl = h3.h3_to_geo(nl)

        # strange things can happen when wrapping the date line.
        t, u = intersection(xa, ya, xb, yb, xl, yl, xr, yr)
        assert 1.0 / 6.0 < t < 5.0 / 6.0

        if u < 1e-10:
            # The line practially runs completely on the side of the end hex
            return [(end_hex, length_start, length_end)]
        if u > 1 - 1e-10:
            # The line practially runs completely on the side of the start hex
            return [(start_hex, length_start, length_end)]

        crossing_point = length_start + u * (length_end - length_start)

        if t < 1.0 / 3.0:
            x_m, y_m = xa + u * (xb - xa), yb + u * (yb - ya)
            return find_hexes_crossed(
                x0, y0, x_m, y_m, length_start, crossing_point
            ) + find_hexes_crossed(x_m, y_m, x1, y1, crossing_point, length_end)
        elif t > 2.0 / 3.0:
            x_m, y_m = xa + u * (xb - xa), yb + u * (yb - ya)
            return find_hexes_crossed(
                x0, y0, x_m, y_m, length_start, crossing_point
            ) + find_hexes_crossed(x_m, y_m, x1, y1, crossing_point, length_end)
        else:
            return [
                (start_hex, length_start, crossing_point),
                (end_hex, crossing_point, length_end),
            ]
    else:
        xmid = (x0 + x1) / 2.0
        ymid = (y0 + y1) / 2.0
        length_mid = length_start + 0.5 * (length_end - length_start)
        return find_hexes_crossed(
            x0, y0, xmid, ymid, length_start, length_mid
        ) + find_hexes_crossed(xmid, ymid, x1, y1, length_mid, length_end)


# 3 knots is about 1.5433333 m/s
KAYAK_SPEED = 1.5433333


def estimate_flow_speed(width, depth, slope, discharge):
    """Estimate the flow speed, in m/s from discharge (m³/s) and slope (m/m)

    This is a very rough estimate, following [@schulze2005simulating]. They
    suggest to at least estimate the widely varying river roughness n, but we
    cannot do that, so we use their reported mean value of 0.044.

    W = 2.71 · Q^0.557
    D = 0.349 · Q^0.341
    R = D · W / (2 D + W)
    n = 0.044

    # Manning-Strickler formula
    v = 1/n · R^2/3 · S^1/2

    """
    n = 0.044
    r = depth * width / (2 * depth + width)
    # Manning-Strickler formula
    v = 1 / n * r ** (2 / 3) * slope**0.5
    return max(v, discharge / (width * depth))


RIVERS = RiverNetwork()


def draw_line(x1, y1, x2, y2):
    """Draw a line from (x1, y1) to (x2, y2).

    Give all the integer coordinate that a line from (x1, y1) to (x2, y2)
    passes through, using a modification of Bresenham's line drawing algorithm,
    where no diagonal connections are allowed, so diagonal steps cannot skip
    the drawn line.

    """
    # Swap parameters such that x1<x2, y1<y2, and the slope m<1.
    if y2 < y1:
        for x, y in draw_line(x1, -y1, x2, -y2):
            yield x, -y
    elif x2 < x1:
        for x, y in draw_line(-x1, y1, -x2, y2):
            yield -x, y
    elif (y2 - y1) > (x2 - x1):
        for y, x in draw_line(y1, x1, y2, x2):
            yield x, y
    else:
        m_adj = 2 * (y2 - y1)
        slope_error_adj = m_adj - (x2 - x1)

        y = y1
        for x in range(x1, x2 + 1):
            yield x, y
            slope_error_adj = slope_error_adj + m_adj
            if slope_error_adj >= 0:
                if y <= y2:
                    yield x, y + 1
                y = y + 1
                slope_error_adj = slope_error_adj - 2 * (x2 - x1)


# For a river that is impossible to cross, set wading time to 20 years. That's
# several orders of magnitude above what we expect for a normal river crossing.
PRACTICALLY_INFINITY = 20 * 365.2421 * 24 * 3600


def wading_time(depth, width, velocity):
    """Calculate the time to wade/swim across a river of

    depth D [m], width W [m], velocity V [m/s]

    Judging from [@davenport2017wading], analysed in a supplementary
    spreadsheet (rough data extracted using WebPlotDigitizer), it seems that
    swim speed S = 0.5788666326641 [m/s] is roughly a lower bound for wading
    speed X [m/s], which otherwise depends on the depth of the body of water D
    [m] as roughly

    X = 1/(0.63 D + 1/X0)

    where X0 is walking speed, so X0 = 1.4454683853418 for the beach walking
    speed of men in their measurements. These data were however measured on a
    beach, where even running through deep water was possible.

    For treacherous terrain, such as inside a river, both stride length and
    stride frequency will be much lower. Waders are advised to shuffle
    carefully and not cross their legs, and measure each step carefully. Our
    base navigation speed is already around 1km/h instead of the more than
    5km/h of [@davenport2017wading]. With the additional safety measures for
    river crossings, an X0 = 0.14454683853418 seems appropriate.

    General advice is to wade across a deeper river roughly at an angle, ending
    up further downstream than the start, while a shallow river can be crossed
    orthogonally. By linear interpolation, we take the effective distance
    across as

    A = √(W² + (D W)²) = W × √(1+D²)

    [@jonkmann2008human] cite three studies reporting the critical product DV
    (hv_c in their notation) at which human wading becomes unstable, and
    compute their own estimate. For a 1.7 m tall person with a weight of 68.25
    kg, these numbers are 1.32 m²/s, 1.27 m²/s, 0.664 m²/s, and 0.5 m²/s. We
    pick 1.27 m²/s as a compromise, because the model is for an experienced and
    prepared explorer (and helpful tools are not out of scope). Above this
    critical product, swimming is necessary.

    """
    X0 = 1.4454683853418 / 10
    if depth * velocity > 1.27:
        # Wading is unstable
        return PRACTICALLY_INFINITY
    if depth > 1.5:
        # Too deep for wading
        return PRACTICALLY_INFINITY
    X = 1 / (0.63 * depth + 1 / X0)
    # TODO: What about swim speed? This already starts out much lower than swim
    # speed. Divide swim speed by 10 also?

    A = width * numpy.sqrt(1 + depth**2)
    return A / X  # Can be infinity


try:
    maxest = json.load(Path("maxest_river.json").open())
except FileNotFoundError:
    maxest = {
        "width": 0,
        "depth": 0,
        "discharge": 0,
        "wading": 0,
        "v": 0,
        "slope": 0,
    }


ENGINE, TABLES = db()


def process_rivers(tile):
    t_node = TABLES["nodes"]
    t_dist = TABLES["edges"]

    template = gmted_tile(tile)
    raster = template.read(1)
    transform = template.transform

    west, north = transform * (-0.5, -0.5)
    east, south = transform * (
        raster.shape[1] + 0.5,
        raster.shape[0] + 0.5,
    )

    print(f"Working between boundaries S{south}W{west}N{north}E{east}")

    # For checking afterwards whether a core location is on the river
    is_river = numpy.zeros(raster.shape, dtype=rasterio.float64)

    for r, reach in tqdm(enumerate(RIVERS.shp.iterShapeRecords())):
        data = reach.record
        reach_id = int(data[0])

        points: t.List[t.Tuple[float, float]] = reach.shape.points

        for (lon0, lat0), (lon1, lat1) in windowed(points, 2):
            if not (
                (west < lon0 < east or west < lon1 < east)
                and (south < lat0 < north or south < lat1 < north)
            ):
                continue
            break
        else:
            continue

        discharge = 10 ** data[3]

        # This is a very rough estimate, following [@allen1994downstream].
        # [@schulze2005simulating] give the numbers used here, which are the
        # translation from Allen's formulas measeured in feet into meters.
        width = 2.71 * discharge**0.557
        depth = 0.349 * discharge**0.341

        # Slope is not directly available in the data, but the stream power is
        # directly proportional to the product of discharge and gradient, so we
        # can reverse-engineer it: Stream power [kg m/s³] = water density
        # [kg/m³] * gravity [m/s²] * discharge [m³/s] * slope [m/m]; so Slope =
        # stream power / (discharge * water density * gravity). GloRiC bases
        # its slope (used for calculating stream power) on maximum elevation
        # minus mean elevation of the reach, so this may still be off by a
        # factor of 2.
        slope = data[11] / (10 ** data[3] * 9810)

        v = estimate_flow_speed(
            width=width, depth=depth, slope=slope, discharge=discharge
        )
        if v > 40.0:
            raise RuntimeError(
                "Found a reach flowing faster than 40 m/s, something is clearly wrong."
            )

        wading = wading_time(width=width, depth=depth, velocity=v)

        if wading < 2:
            continue

        if data[3] < 0.4520448942601702:
            livingood = 3.0
        elif data[3] < 1.4520448942601702:
            livingood = 300.0
        elif data[3] < 2.4520448942601702:
            livingood = 600.0
        else:
            livingood = 1800.0

        xest = ""
        if width > maxest["width"]:
            maxest["width"] = width
            xest += "widest "
        if depth > maxest["depth"]:
            maxest["depth"] = depth
            xest += "deepest "
        if discharge > maxest["discharge"]:
            maxest["discharge"] = discharge
            xest += "biggest "
        if wading > maxest["wading"]:
            maxest["wading"] = wading
            xest += "most difficult to cross "
        if v > maxest["v"]:
            maxest["v"] = v
            xest += "fastest "
        if slope > maxest["slope"]:
            maxest["slope"] = slope
            xest += "steepest "
        if xest:
            print(
                f"Found new {xest}river reach:",
                f" Reach starting at {points[0]}",
                f"Width: {width}",
                f"Depth: {depth}",
                f"Discharge: {discharge}",
                f"Wading time: {wading}",
                f"Flow speed: {v}",
                "Effective flow: {}".format(
                    discharge / (depth * width * v) if v > 0 else "inf"
                ),
                f"Slope: {slope}",
                f"Livingood penalty: {livingood}",
                sep="\n  ",
            )

        for (lon0, lat0), (lon1, lat1) in windowed(points, 2):
            if not (
                (west < lon0 < east or west < lon1 < east)
                and (south < lat0 < north or south < lat1 < north)
            ):
                continue

            col0, row0 = ~transform * (lon0, lat0)
            col1, row1 = ~transform * (lon1, lat1)
            # The river network shows artefacts of being derived from a GEM
            # with compatible resolution (I think 15" instead of 30"),
            # which show up as NE-SW running river reaches crossing exactly
            # through the pixel corners. Shifting them a tiny bit to SE –
            # taking care that it won't be precisely diagonal, to avoid
            # introducing other artefact – should help with that.

            cells = draw_line(round(row0), round(col0), round(row1), round(col1))
            # Filter down to the cells that are part of this tile
            cells = [
                c
                for c in cells
                if 0 <= c[0] < is_river.shape[0]
                if 0 <= c[1] < is_river.shape[1]
            ]
            # Leaving a river pixel costs time.
            for cell in cells:
                is_river[cell] = max(is_river[cell], wading)

        # Is this reach navigable by Kayak/Canoe? From
        # [@rood2006instream,@zinke2018comparing] it seems that reaches with a
        # flow lower than 5m³/s are not navigable even by professional extreme
        # sport athletes, and generally even that number seems to be an outlier
        # with opinions starting at 8m³/s, so we take that as the cutoff.
        #
        # [@zinke2018comparing] further plots wild water kayaking run slopes
        # vs. difficulty. All of these are below 10%, so we assume that reaches
        # above 10% are not navigable.
        if data[3] < 0.9030899869919434 or slope > 0.1:
            continue

        with ENGINE.begin() as conn:
            # slope = stream power / discharge / (1000 * 9.81) > 10% = 0.1
            print(data[0], KAYAK_SPEED + v, KAYAK_SPEED + v)

            downstream = data[1]
            conn.execute(
                insert(t_node)
                .values(
                    {
                        "node_id": reach_id,
                        "longitude": points[0][0],
                        "latitude": points[0][1],
                        "coastal": False,
                    },
                )
                .on_conflict_do_nothing()
            )
            if downstream == 0:
                downstream = -reach_id
                coastal = True
            else:
                coastal = False
            conn.execute(
                insert(t_node)
                .values(
                    {
                        "node_id": downstream,
                        "longitude": points[-1][0],
                        "latitude": points[-1][1],
                        "coastal": coastal,
                    },
                )
                .on_conflict_do_nothing()
            )

            conn.execute(
                insert(t_dist)
                .values(
                    {
                        "node1": reach_id,
                        "node2": downstream,
                        "source": "river",
                        "flat_distance": data[2] * 1000,
                        "travel_time": data[2] * 1000 / (KAYAK_SPEED + v),
                    },
                )
                .on_conflict_do_nothing()
            )
            if KAYAK_SPEED > v:
                conn.execute(
                    insert(t_dist)
                    .values(
                        {
                            "node1": downstream,
                            "node2": reach_id,
                            "source": "river",
                            "flat_distance": data[2] * 1000,
                            "travel_time": data[2] * 1000 / (KAYAK_SPEED - v),
                        },
                    )
                    .on_conflict_do_nothing()
                )

    profile = rasterio.profiles.DefaultGTiffProfile()
    profile["height"] = is_river.shape[0]
    profile["width"] = is_river.shape[1]
    profile["transform"] = transform
    profile["dtype"] = numpy.float64
    profile["count"] = 1

    fname = "rivers-{0:s}.tif".format(fmt(tile))

    with rasterio.open(
        fname,
        "w",
        **profile,
    ) as dst:
        dst.write(is_river.astype(numpy.float64), 1)


if __name__ == "__main__":
    import sys

    ENGINE, TABLES = db(sys.argv[1])
    ns, lat, ew, lon = unfmt(sys.argv[2])
    print((ns, lat, ew, lon))
    process_rivers((ns, lat, ew, lon))
    try:
        maxest_now = json.load(Path("maxest_river.json").open())
    except FileNotFoundError:
        maxest_now = {
            "width": 0,
            "depth": 0,
            "discharge": 0,
            "wading": 0,
            "v": 0,
            "slope": 0,
        }
    maxest = {k: max(maxest[k], maxest_now[k]) for k in maxest}
    json.dump(maxest, Path("maxest_river.json").open("w"))
