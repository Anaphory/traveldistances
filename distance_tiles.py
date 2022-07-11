import typing as t

import numpy

import rasterio

from earth import GEODESIC
from ecoregions import TC
from raster_data import (
    gmted_tile,
    Tile,
    ecoregion_tile,
    boundingbox_from_tile,
    tile_from_geocoordinates,
    RowCol,
)


def extend_tile(
    tile: Tile, buffer=500
) -> t.Tuple[numpy.ndarray, numpy.ndarray, rasterio.Affine]:
    """Load ecology data.

    Load the ecology metadata (elevations and ecoregions, in that order) of the
    tile (30°×20°) surrounding `containing`, plus a 500 pixel overlapping
    boundary region on each side, giving a (8200×5800) numpy array for the
    elevations (in m) and the ecoregions (integer categories)

    Returns
    =======
    Elevation: numpy.array
    Ecoregions: numpy.array
    transform: rasterio.Affine

    """
    print(f"Preparing rasters around for tile {tile}:")
    elevation_file = gmted_tile(tile)
    m_trafo = elevation_file.transform
    height, width = elevation_file.shape
    elevation = numpy.full((height + 2 * buffer, width + 2 * buffer), -100, int)
    ecoregions = numpy.full((height + 2 * buffer, width + 2 * buffer), 999, int)
    elevation[buffer:-buffer, buffer:-buffer] = elevation_file.read(1)
    ecoregions[buffer:-buffer, buffer:-buffer] = ecoregion_tile(tile).read(1)
    print("Loading margins of adjacent tiles…")
    west, south, east, north = boundingbox_from_tile(tile)
    try:
        nw = tile_from_geocoordinates(west - 15, north + 10)
        elevation[:buffer, :buffer] = gmted_tile(nw).read(1)[-buffer:, -buffer:]
        ecoregions[:buffer, :buffer] = ecoregion_tile(nw).read(1)[-buffer:, -buffer:]
    except rasterio.RasterioIOError:
        pass
    try:
        n = tile_from_geocoordinates(west + 15, north + 10)
        elevation[:buffer, buffer:-buffer] = gmted_tile(n).read(1)[-buffer:, :]
        ecoregions[:buffer, buffer:-buffer] = ecoregion_tile(n).read(1)[-buffer:, :]
    except rasterio.RasterioIOError:
        pass
    try:
        ne = tile_from_geocoordinates(east + 15, north + 10)
        elevation[:buffer, -buffer:] = gmted_tile(ne).read(1)[-buffer:, :buffer]
        ecoregions[:buffer, -buffer:] = ecoregion_tile(ne).read(1)[-buffer:, :buffer]
    except rasterio.RasterioIOError:
        pass
    try:
        w = tile_from_geocoordinates(west - 15, south + 10)
        elevation[buffer:-buffer, :buffer] = gmted_tile(w).read(1)[:, -buffer:]
        ecoregions[buffer:-buffer, :buffer] = ecoregion_tile(w).read(1)[:, -buffer:]
    except rasterio.RasterioIOError:
        pass
    try:
        e = tile_from_geocoordinates(east + 15, south + 10)
        elevation[buffer:-buffer, -buffer:] = gmted_tile(e).read(1)[:, :buffer]
        ecoregions[buffer:-buffer, -buffer:] = ecoregion_tile(e).read(1)[:, :buffer]
    except rasterio.RasterioIOError:
        pass
    try:
        sw = tile_from_geocoordinates(west - 15, south - 10)
        elevation[-buffer:, :buffer] = gmted_tile(sw).read(1)[:buffer, -buffer:]
        ecoregions[-buffer:, :buffer] = ecoregion_tile(sw).read(1)[:buffer, -buffer:]
    except rasterio.RasterioIOError:
        pass
    try:
        s = tile_from_geocoordinates(west + 15, south - 10)
        elevation[-buffer:, buffer:-buffer] = gmted_tile(s).read(1)[:buffer, :]
        ecoregions[-buffer:, buffer:-buffer] = ecoregion_tile(s).read(1)[:buffer, :]
    except rasterio.RasterioIOError:
        pass
    try:
        se = tile_from_geocoordinates(east + 15, south - 10)
        elevation[-buffer:, -buffer:] = gmted_tile(se).read(1)[:buffer, :buffer]
        ecoregions[-buffer:, -buffer:] = ecoregion_tile(se).read(1)[:buffer, :buffer]
    except rasterio.RasterioIOError:
        pass

    transform = rasterio.Affine(
        m_trafo.a,
        0,
        m_trafo.c - buffer * m_trafo.a,
        0,
        m_trafo.e,
        m_trafo.f - buffer * m_trafo.e,
    )
    print("Data loaded")
    return elevation, ecoregions, transform


# =========================
# Travelling time functions
# =========================
def navigation_speed(slope: float) -> float:
    """Using the slope in %, calculate the navigation speed in m/s

    This function calculates the off-road navigation speed (for male cadets in
    forested areas with navigational targets every 20 minutes) following
    [@irmischer2018measuring]. Like their formula, slope is in % and speed in
    m/s.

    > [T]he fastest off-road navigation speed was 0.78 m/s for males […] with a
    > peak at −2%.

    >>> navigation_speed(-2.)
    0.78
    >>> navigation_speed(-2.) > navigation_speed(-1.5)
    True
    >>> navigation_speed(-2.) > navigation_speed(-2.5)
    True

    """
    return 0.11 + 0.67 * numpy.exp(-((slope + 2.0) ** 2) / 1800.0)


def all_moore_neighbor_distances(
    elevation: numpy.array,
    transform: rasterio.Affine,
    terrain_coefficients: numpy.array,
) -> t.Dict[RowCol, numpy.array]:
    """Calculate the arrays of distances in all 8 neighbor directions.

    From an elevation raster and a terrain coefficient raster (higher
    coefficient = higher walking speed), localized using an affine
    transformation, compute the walking times 1 pixel into each direction (N,
    NE, E, …, NW).

    Return the resulting arrays inside a dictionary with index manipulations:

        {
         (-1, 0): distance to north,
         (-1, 1): distance to north east,
         (0, 1): distance to east,
         ...
         (-1, -1): distance to north west,
        }

    """
    # Compute the geodesic distances. They are constant for each row, which
    # corresponds to a constant latitude.
    height, width = elevation.shape
    d_n, d_e, d_ne = [], [], []
    for y in range(len(elevation)):
        (lon0, lat0) = transform * (0, y + 1)
        (lon1, lat1) = transform * (1, y)

        d = GEODESIC.inverse((lon0, lat0), [(lon0, lat1), (lon1, lat0), (lon1, lat1)])
        d_n.append(d[0, 0])
        d_e.append(d[1, 0])
        d_ne.append(d[2, 0])
    distance_to_north = numpy.array(d_n[:-1])
    slope_to_north = (
        100 * (elevation[1:, :] - elevation[:-1, :]) / distance_to_north[:, None]
    )
    tc_to_north = (terrain_coefficients[1:, :] + terrain_coefficients[:-1, :]) / 2
    north = numpy.vstack(
        (
            numpy.full((1, width), numpy.inf),
            distance_to_north[:, None]
            / (navigation_speed(slope_to_north) * tc_to_north),
        )
    )
    south = numpy.vstack(
        (
            distance_to_north[:, None]
            / (navigation_speed(-slope_to_north) * tc_to_north),
            numpy.full((1, width), numpy.inf),
        )
    )
    del distance_to_north, slope_to_north, tc_to_north

    distance_to_east = numpy.array(d_e)
    slope_to_east = (
        100 * (elevation[:, 1:] - elevation[:, :-1]) / distance_to_east[:, None]
    )
    tc_to_east = (terrain_coefficients[:, 1:] + terrain_coefficients[:, :-1]) / 2
    east = numpy.hstack(
        (
            distance_to_east[:, None] / (navigation_speed(slope_to_east) * tc_to_east),
            numpy.full((height, 1), numpy.inf),
        )
    )
    west = numpy.hstack(
        (
            numpy.full((height, 1), numpy.inf),
            distance_to_east[:, None] / (navigation_speed(-slope_to_east) * tc_to_east),
        )
    )
    del distance_to_east, slope_to_east, tc_to_east

    distance_to_northeast = numpy.array(d_ne[:-1])
    slope_to_northeast = (
        100 * (elevation[1:, 1:] - elevation[:-1, :-1]) / distance_to_northeast[:, None]
    )
    tc_to_northeast = (
        terrain_coefficients[1:, 1:] + terrain_coefficients[:-1, :-1]
    ) / 2
    northeast = numpy.vstack(
        (
            numpy.full((1, width), numpy.inf),
            numpy.hstack(
                (
                    distance_to_northeast[:, None]
                    / (navigation_speed(slope_to_northeast) * tc_to_northeast),
                    numpy.full((height - 1, 1), numpy.inf),
                )
            ),
        )
    )
    southwest = numpy.vstack(
        (
            numpy.hstack(
                (
                    numpy.full((height - 1, 1), numpy.inf),
                    distance_to_northeast[:, None]
                    / (navigation_speed(-slope_to_northeast) * tc_to_northeast),
                )
            ),
            numpy.full((1, width), numpy.inf),
        )
    )
    del distance_to_northeast, slope_to_northeast, tc_to_northeast
    distance_to_northwest = numpy.array(d_ne)[:-1]
    slope_to_northwest = (
        100 * (elevation[1:, :-1] - elevation[:-1, 1:]) / distance_to_northwest[:, None]
    )

    tc_to_northwest = (
        terrain_coefficients[1:, :-1] + terrain_coefficients[:-1, 1:]
    ) / 2

    southeast = numpy.vstack(
        (
            numpy.hstack(
                (
                    distance_to_northwest[:, None]
                    / (navigation_speed(-slope_to_northwest) * tc_to_northwest),
                    numpy.full((height - 1, 1), numpy.inf),
                )
            ),
            numpy.full((1, width), numpy.inf),
        )
    )

    northwest = numpy.vstack(
        (
            numpy.full((1, width), numpy.inf),
            numpy.hstack(
                (
                    numpy.full((height - 1, 1), numpy.inf),
                    distance_to_northwest[:, None]
                    / (navigation_speed(slope_to_northwest) * tc_to_northwest),
                )
            ),
        )
    )

    del distance_to_northwest, slope_to_northwest, tc_to_northwest

    return {
        (-1, 0): north,
        (-1, 1): northeast,
        (0, 1): east,
        (1, 1): southeast,
        (1, 0): south,
        (1, -1): southwest,
        (0, -1): west,
        (-1, -1): northwest,
    }


def moore_distances(tile: Tile):
    elevation, ecoregions, transform = extend_tile(tile)

    terrain_coefficient_raster = TC[ecoregions]
    distance_by_direction = all_moore_neighbor_distances(
        elevation, transform, terrain_coefficient_raster
    )
    profile = rasterio.profiles.DefaultGTiffProfile()
    profile["height"] = elevation.shape[0]
    profile["width"] = elevation.shape[1]
    profile["transform"] = transform
    profile["dtype"] = rasterio.float64
    profile["count"] = 8
    del elevation, ecoregions, terrain_coefficient_raster

    fname = "distances-{1:02d}{0:s}{3:03d}{2:s}.tif".format(*tile)
    with rasterio.open(
        fname,
        "w",
        **profile,
    ) as dst:
        for i, band in enumerate(distance_by_direction.values(), 1):
            dst.write(band.astype(rasterio.float64), i)
    return distance_by_direction, transform


if __name__ == "__main__":
    import sys
    tile = sys.argv[1]
    ns, lat, ew, lon = tile[2], int(tile[0:2]), tile[6], int(tile[3:6])
    print((ns, lat, ew, lon))
    moore_distances((ns, lat, ew, lon))
