"""Plot embedding model results."""
import argparse

import cartopy.crs as ccrs
import cartopy.geodesic as geodesic
import keras.backend as K
import keras.models
import matplotlib.cm as cm
import numpy
import tensorflow as tf
from database import db
from generate_training_data import distances_from_focus
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from sqlalchemy import select
from tqdm import tqdm


DATABASE, TABLES = db("sqlite:///all-distances.sqlite")
GEODESIC: geodesic.Geodesic = geodesic.Geodesic()

lon, lat = 34.08, 45.00
# distances_from_focus(nearby_node(lon, lat), database=DATABASE, tables=TABLES)


def nearby_node(longitude, latitude):
    lon_scaling_factor = numpy.cos(latitude / 180 * numpy.pi)
    d_min = numpy.inf
    for node, lon, lat in DATABASE.execute(
        select(
            [
                TABLES["nodes"].c.node_id,
                TABLES["nodes"].c.longitude,
                TABLES["nodes"].c.latitude,
            ]
        )
        .select_from(TABLES["nodes"])
        .where(
            ((TABLES["nodes"].c.longitude - longitude) * lon_scaling_factor < 2)
            & ((TABLES["nodes"].c.longitude - longitude) * lon_scaling_factor > -2)
            & ((TABLES["nodes"].c.latitude - latitude) < 2)
            & ((TABLES["nodes"].c.latitude - latitude) > -2)
        )
        .order_by(
            (
                (TABLES["nodes"].c.longitude - longitude)
                * (TABLES["nodes"].c.longitude - longitude)
            )
            * lon_scaling_factor**2
            + (TABLES["nodes"].c.latitude - latitude)
            * (TABLES["nodes"].c.latitude - latitude)
        )
    ):
        d = GEODESIC.inverse((lon, lat), (longitude, latitude))[0, 0]
        if d < d_min:
            d_min = d
            node_min = node
        if d > 2 * d_min:
            break
    else:
        raise ValueError("No habitable place nearby.")
    return node_min


def distance_raster(start, ends):
    distances = {}
    for node, dist in distances_from_focus(
        start,
        database=DATABASE,
        tables=TABLES,
        maximum_dist=numpy.inf,
        destinations=ends,
    ).items():
        lon, lat = DATABASE.execute(
            select(
                [
                    TABLES["nodes"].c.longitude,
                    TABLES["nodes"].c.latitude,
                ]
            )
            .select_from(TABLES["nodes"])
            .where(TABLES["nodes"].c.node_id == node)
        ).one()
        distances[lon, lat] = dist
    return distances


def lonlat_to_3d(lonlat):
    lon = lonlat[..., 0] * numpy.pi / 180
    lat = lonlat[..., 1] * numpy.pi / 180
    return tf.stack(
        (K.sin(lon) * K.cos(lat), K.cos(lon) * K.cos(lat), K.sin(lat)),
        -1,
        name="coords_3d",
    )


examples = [
    {
        "Alb": (19 + 27 / 60, 41 + 19 / 60),
        "Arm": (47.057, 39.68),
        "Ana": (34.617, 40.020),
        "BSl": (31 + 32 / 60 + 57 / 3600, 49 + 42 / 60 + 10 / 3600),
        "Cel": (8 + 42 / 60, 45 + 43 / 60),
        "Ger": (10 + 18 / 60 + 38 / 3600, 55 + 26 / 60 + 22 / 3600),
        "Grk": (21 + 43.4 / 3600, 36 + 59.7 / 60),
        "Lat": (12 + 54 / 60, 41 + 50 / 60),
        "IAr": (76.182, 27.43),
        "Ira": (47 + 26 / 60 + 9 / 3600, 34 + 23 / 60 + 26 / 3600),
        "Toc": (82 + 58 / 60, 41 + 43 / 60),
    },
    {
        "crh": (34.08, 45.00),
        "nog": (43.17, 44.90),
        "kum": (47.00, 43.00),
        "azj": (46.47, 40.98),
        "tuk": (59.18, 37.09),
        "kaa": (63.32, 39.98),
    },
]

X = 300
Y = 300
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stored_model")
    args = parser.parse_args()

    full_model = keras.models.load_model(args.stored_model)

    embedding_model = keras.Model(
        inputs=[full_model.input[0]],
        outputs=[full_model.get_layer("euclidean_distance_in_embedding").input[0]],
    )

    mesh_coords = (
        numpy.stack(
            numpy.meshgrid(numpy.linspace(-180, 180, 720), numpy.linspace(-90, 90, 360))
        )
        .transpose((1, 2, 0))
        .reshape(-1, 2)
    )
    results = embedding_model.predict(mesh_coords)
    if results.shape[1] > 25:
        results = results[:, :25]

    layout = int(results.shape[1] ** 0.5)
    fig = plt.figure()
    ref = None
    for i, component in enumerate(results.T):
        ax = fig.add_subplot(
            (results.shape[1] - 1) // layout + 1,
            layout,
            i + 1,
            projection=ccrs.Mollweide(),
            sharex=ref,
            sharey=ref,
        )
        if ref is None:
            ref = ax
        ax.coastlines()
        plt.imshow(
            component.reshape((360, 720)),
            extent=(-180, 180, -90, 90),
            transform=ccrs.PlateCarree(),
            origin="lower",
        )
    plt.show()

    for positions in examples:
        lons, lats = zip(*positions.values())
        min_lon, max_lon, min_lat, max_lat = min(lons), max(lons), min(lats), max(lats)
        mesh_coords = numpy.stack(
            numpy.meshgrid(
                numpy.linspace(min_lon, max_lon, X), numpy.linspace(min_lat, max_lat, Y)
            )
        ).transpose((1, 2, 0))
        n = len(positions)
        coords = numpy.array(list(positions.values()))
        nodes = [nearby_node(lon, lat) for lon, lat in coords]
        stack = numpy.stack(
            numpy.broadcast_arrays(mesh_coords[None, :, :, :], coords[:, None, None, :])
        )
        starts, ends = numpy.broadcast_arrays(
            mesh_coords[None, :, :, :], coords[:, None, None, :]
        )
        results = full_model.predict([starts.reshape(-1, 2), ends.reshape(-1, 2)])

        max_dist = []
        distances = []
        common_nodes = None
        for i, node in enumerate(tqdm(nodes)):
            d = distance_raster(node, nodes[:i] + nodes[i + 1 :])
            if common_nodes is None:
                common_nodes = set(d)
            else:
                common_nodes &= set(d)
            distances.append(d)
            max_dist.append(max(d.values()))

        median_max = numpy.median(max_dist)

        cmap = cm.get_cmap("viridis")
        normalizer = Normalize(0, median_max)
        im = cm.ScalarMappable(norm=normalizer)
        fig = plt.figure()
        all_axes = []
        for i, (lon, lat) in enumerate(tqdm(positions.values())):
            ax = fig.add_subplot(
                int(n**0.5),
                (n - 1) // int(n**0.5) + 1,
                i + 1,
                projection=ccrs.Mollweide(),
            )
            all_axes.append(ax)
            ax.coastlines()
            mesh = ax.pcolormesh(
                mesh_coords[..., 0],
                mesh_coords[..., 1],
                results[0].reshape((n, X, Y))[i],
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=normalizer,
            )
            x = [x for x, y in common_nodes]
            y = [y for x, y in common_nodes]
            c = [distances[i][n] for n in common_nodes]
            plt.scatter(
                x, y, 1, c=c, transform=ccrs.PlateCarree(), cmap=cmap, norm=normalizer
            )
            ax.scatter([lon], [lat], transform=ccrs.PlateCarree())
            ax.set_title(list(positions)[i])
        plt.show()

        common_distances = {
            point: sum(d[point] for d in distances) for point in tqdm(common_nodes)
        }
        x = [x for x, y in common_distances]
        y = [y for x, y in common_distances]
        c = list(common_distances.values())
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
        ax.coastlines()
        ax.set_global()
        cmap = cm.get_cmap("viridis")
        normalizer = Normalize(0, median_max * len(positions))
        plt.pcolormesh(
            mesh_coords[..., 0],
            mesh_coords[..., 1],
            results[0].reshape((n, X, Y)).sum(0),
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=normalizer,
        )
        plt.scatter(
            x, y, 1, c=c, transform=ccrs.PlateCarree(), cmap=cmap, norm=normalizer
        )
        plt.show()
