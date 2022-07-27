import sys

import numpy
import rasterio
from by_river import PRACTICALLY_INFINITY
from raster_data import fmt
from raster_data import tile_from_geocoordinates
from raster_data import unfmt


def merge_river_tile(lon: float, lat: float):
    tile = tile_from_geocoordinates(lon, lat)
    fname = f"rivers-{fmt(tile)}.tif"
    print(f"Merging river rasters around {fmt(tile)}…")
    rivers_r = rasterio.open(fname)
    transform = rivers_r.transform
    rivers_center = rivers_r.read(1)
    rivers = numpy.full(
        (rivers_center.shape[0] + 1000, rivers_center.shape[1] + 1000),
        PRACTICALLY_INFINITY,
    )
    rivers[500:-500, 500:-500] = rivers_center
    del rivers_center

    print("Loading adjacent data…")

    # NORTH
    try:
        n_tile = tile_from_geocoordinates(lon, lat + 20)
        n_fname = f"rivers-{fmt(n_tile)}.tif"
        n_rivers = rasterio.open(n_fname).read(
            1, window=((4800 - 500, 4800), (0, 7200))
        )
        rivers[:500, 500:-500] = n_rivers
    except rasterio.RasterioIOError:
        pass

    # NORTH-EAST
    try:
        n_tile = tile_from_geocoordinates(lon + 30, lat + 20)
        n_fname = f"rivers-{fmt(n_tile)}.tif"
        n_rivers = rasterio.open(n_fname).read(1, window=((4800 - 500, 4800), (0, 500)))
        rivers[:500, -500:] = n_rivers
    except rasterio.RasterioIOError:
        pass

    # EAST
    try:
        n_tile = tile_from_geocoordinates(lon + 30, lat)
        n_fname = f"rivers-{fmt(n_tile)}.tif"
        n_rivers = rasterio.open(n_fname).read(1, window=((0, 4800), (0, 500)))
        rivers[500:-500, -500:] = n_rivers
    except rasterio.RasterioIOError:
        pass

    # SOUTH-EAST
    try:
        n_tile = tile_from_geocoordinates(lon + 30, lat - 20)
        n_fname = f"rivers-{fmt(n_tile)}.tif"
        n_rivers = rasterio.open(n_fname).read(1, window=((0, 500), (0, 500)))
        rivers[-500:, -500:] = n_rivers
    except rasterio.RasterioIOError:
        pass

    # SOUTH
    try:
        n_tile = tile_from_geocoordinates(lon, lat - 20)
        n_fname = f"rivers-{fmt(n_tile)}.tif"
        n_rivers = rasterio.open(n_fname).read(1, window=((0, 500), (0, 7200)))
        rivers[-500:, 500:-500] = n_rivers
    except rasterio.RasterioIOError:
        pass

    # SOUTH-WEST
    try:
        n_tile = tile_from_geocoordinates(lon - 30, lat - 20)
        n_fname = f"rivers-{fmt(n_tile)}.tif"
        n_rivers = rasterio.open(n_fname).read(1, window=((0, 500), (7200 - 500, 7200)))
        rivers[-500:, :500] = n_rivers
    except rasterio.RasterioIOError:
        pass

    # WEST
    try:
        n_tile = tile_from_geocoordinates(lon - 30, lat)
        n_fname = f"rivers-{fmt(n_tile)}.tif"
        n_rivers = rasterio.open(n_fname).read(
            1, window=((0, 4800), (7200 - 500, 7200))
        )
        rivers[500:-500, :500] = n_rivers
    except rasterio.RasterioIOError:
        pass

    # NORTH-WEST
    try:
        n_tile = tile_from_geocoordinates(lon - 30, lat + 20)
        n_fname = f"rivers-{fmt(n_tile)}.tif"
        n_rivers = rasterio.open(n_fname).read(
            1, window=((4800 - 500, 4800), (7200 - 500, 7200))
        )
        rivers[:500, :500] = n_rivers
    except rasterio.RasterioIOError:
        pass

    fname_x = f"x-rivers-{fmt(tile)}.tif"
    profile = rasterio.profiles.DefaultGTiffProfile()
    profile["height"] = 5800
    profile["width"] = 8200
    profile["transform"] = transform
    profile["dtype"] = rasterio.float64
    profile["count"] = 1

    with rasterio.open(
        fname_x,
        "w",
        **profile,
    ) as dst:
        dst.write(rivers, 1)


if __name__ == "__main__":
    ns, lat, ew, lon = unfmt(sys.argv[1])
    print("Stitching river tiles", fmt((ns, lat, ew, lon)), "…")
    if ns == "S":
        lat *= -1
    if ew == "W":
        lon *= -1
    lat += 10
    lon += 15
    merge_river_tile(lon, lat)
