import sys
import numpy
import rasterio

from raster_data import tile_from_geocoordinates, fmt, unfmt


def merge_voronoi_tile(lon: float, lat: float):
    tile = tile_from_geocoordinates(lon, lat)
    fname_v = f"voronoi-{fmt(tile)}.tif"
    fname_d = f"min_distances-{fmt(tile)}.tif"
    print("Merging Voronoi rasters around ({:}, {:}) ({:}):".format(lon, lat, fname_v))
    voronoi_allocation_f = rasterio.open(fname_v)
    transform = voronoi_allocation_f.transform
    voronoi_allocation = voronoi_allocation_f.read(1, window=((0, 5800), (0, 8200)))
    min_distances = rasterio.open(fname_d).read(1, window=((0, 5800), (0, 8200)))

    print("Loading adjacent data…")

    try:
        n_tile = tile_from_geocoordinates(lon, lat + 20)
        n_fname_v = f"voronoi-{fmt(n_tile)}.tif"
        n_fname_d = f"min_distances-{fmt(n_tile)}.tif"
        n_voronoi_allocation = rasterio.open(n_fname_v).read(
            1, window=((5800 - 1000, 5800), (1000, 8200 - 1000))
        )
        n_min_distances = rasterio.open(n_fname_d).read(
            1, window=((5800 - 1000, 5800), (1000, 8200 - 1000))
        )

        other_is_better = n_min_distances < min_distances[:1000, 1000:-1000]
        if other_is_better.any():
            print("Updating from the north ({})".format(n_fname_v))
        voronoi_allocation[:1000, 1000:-1000][other_is_better] = n_voronoi_allocation[
            other_is_better
        ]
    except rasterio.RasterioIOError:
        pass

    try:
        n_tile = tile_from_geocoordinates(lon + 30, lat + 20)
        n_fname_v = f"voronoi-{fmt(n_tile)}.tif"
        n_fname_d = f"min_distances-{fmt(n_tile)}.tif"
        n_voronoi_allocation = rasterio.open(n_fname_v).read(
            1, window=((5800 - 1000, 5800), (0, 1000))
        )
        n_min_distances = rasterio.open(n_fname_d).read(
            1, window=((5800 - 1000, 5800), (0, 1000))
        )

        other_is_better = n_min_distances < min_distances[:1000, -1000:]
        if other_is_better.any():
            print("Updating from the north-east ({})".format(n_fname_v))
        voronoi_allocation[:1000, -1000:][other_is_better] = n_voronoi_allocation[
            other_is_better
        ]
    except rasterio.RasterioIOError:
        pass

    try:
        n_tile = tile_from_geocoordinates(lon + 30, lat)
        n_fname_v = f"voronoi-{fmt(n_tile)}.tif"
        n_fname_d = f"min_distances-{fmt(n_tile)}.tif"
        n_voronoi_allocation = rasterio.open(n_fname_v).read(
            1, window=((1000, 5800 - 1000), (0, 1000))
        )
        n_min_distances = rasterio.open(n_fname_d).read(
            1, window=((1000, 5800 - 1000), (0, 1000))
        )

        other_is_better = n_min_distances < min_distances[1000:-1000, -1000:]
        if other_is_better.any():
            print("Updating from the east ({})".format(n_fname_v))
        voronoi_allocation[1000:-1000, -1000:][other_is_better] = n_voronoi_allocation[
            other_is_better
        ]
    except rasterio.RasterioIOError:
        pass

    try:
        n_tile = tile_from_geocoordinates(lon + 30, lat - 20)
        n_fname_v = f"voronoi-{fmt(n_tile)}.tif"
        n_fname_d = f"min_distances-{fmt(n_tile)}.tif"
        n_voronoi_allocation = rasterio.open(n_fname_v).read(
            1, window=((0, 1000), (0, 1000))
        )
        n_min_distances = rasterio.open(n_fname_d).read(
            1, window=((0, 1000), (0, 1000))
        )

        other_is_better = n_min_distances < min_distances[-1000:, -1000:]
        if other_is_better.any():
            print("Updating from the south-east ({})".format(n_fname_v))
        voronoi_allocation[-1000:, -1000:][other_is_better] = n_voronoi_allocation[
            other_is_better
        ]
    except rasterio.RasterioIOError:
        pass

    try:
        n_tile = tile_from_geocoordinates(lon, lat - 20)
        n_fname_v = f"voronoi-{fmt(n_tile)}.tif"
        n_fname_d = f"min_distances-{fmt(n_tile)}.tif"
        n_voronoi_allocation = rasterio.open(n_fname_v).read(
            1, window=((0, 1000), (1000, 8200 - 1000))
        )
        n_min_distances = rasterio.open(n_fname_d).read(
            1, window=((0, 1000), (1000, 8200 - 1000))
        )

        other_is_better = n_min_distances < min_distances[-1000:, 1000:-1000]
        if other_is_better.any():
            print("Updating from the south ({})".format(n_fname_v))
        voronoi_allocation[-1000:, 1000:-1000][other_is_better] = n_voronoi_allocation[
            other_is_better
        ]
    except rasterio.RasterioIOError:
        pass

    try:
        n_tile = tile_from_geocoordinates(lon - 30, lat - 20)
        n_fname_v = f"voronoi-{fmt(n_tile)}.tif"
        n_fname_d = f"min_distances-{fmt(n_tile)}.tif"
        n_voronoi_allocation = rasterio.open(n_fname_v).read(
            1, window=((0, 1000), (8200 - 1000, 8200))
        )
        n_min_distances = rasterio.open(n_fname_d).read(
            1, window=((0, 1000), (8200 - 1000, 8200))
        )

        other_is_better = n_min_distances < min_distances[-1000:, :1000]
        if other_is_better.any():
            print("Updating from the south-west ({})".format(n_fname_v))
        voronoi_allocation[-1000:, :1000][other_is_better] = n_voronoi_allocation[
            other_is_better
        ]
    except rasterio.RasterioIOError:
        pass

    try:
        n_tile = tile_from_geocoordinates(lon - 30, lat)
        n_fname_v = f"voronoi-{fmt(n_tile)}.tif"
        n_fname_d = f"min_distances-{fmt(n_tile)}.tif"
        n_voronoi_allocation = rasterio.open(n_fname_v).read(
            1, window=((1000, 5800 - 1000), (8200 - 1000, 8200))
        )
        n_min_distances = rasterio.open(n_fname_d).read(
            1, window=((1000, 5800 - 1000), (8200 - 1000, 8200))
        )

        other_is_better = n_min_distances < min_distances[1000:-1000, :1000]
        if other_is_better.any():
            print("Updating from the west ({})".format(n_fname_v))
        voronoi_allocation[1000:-1000, :1000][other_is_better] = 0
    except rasterio.RasterioIOError:
        pass

    try:
        n_tile = tile_from_geocoordinates(lon - 30, lat + 20)
        n_fname_v = f"voronoi-{fmt(n_tile)}.tif"
        n_fname_d = f"min_distances-{fmt(n_tile)}.tif"
        n_voronoi_allocation = rasterio.open(n_fname_v).read(
            1, window=((5800 - 1000, 5800), (8200 - 1000, 8200))
        )
        n_min_distances = rasterio.open(n_fname_d).read(
            1, window=((5800 - 1000, 5800), (8200 - 1000, 8200))
        )

        other_is_better = n_min_distances < min_distances[:1000, :1000]
        if other_is_better.any():
            print("Updating from the north-west ({})".format(n_fname_v))
        voronoi_allocation[:1000, :1000][other_is_better] = n_voronoi_allocation[
            other_is_better
        ]
    except rasterio.RasterioIOError:
        pass

    profile = rasterio.profiles.DefaultGTiffProfile()
    profile["height"] = 4800
    profile["width"] = 7200
    profile["transform"] = transform
    profile["dtype"] = rasterio.uint32
    profile["count"] = 1

    with rasterio.open(
        f"x-voronoi-{fmt(tile)}.tif",
        "w",
        **profile,
    ) as dst:
        dst.write(voronoi_allocation[500:-500, 500:-500], 1)


if __name__ == "__main__":
    ns, lat, ew, lon = unfmt(sys.argv[1])
    print("Stitching Voronoi polygons", fmt((ns, lat, ew, lon)), "…")
    if ns == "S":
        lat *= -1
    if ew == "W":
        lon *= -1
    lat += 10
    lon += 15
    merge_voronoi_tile(lon, lat)
