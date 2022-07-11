import typing as t
import sys
from tqdm import tqdm
import numpy
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, PReLU, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import keras as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from database import db
from sqlalchemy import select
import itertools
from itertools import count
import typing as t
from heapq import heappush as push, heappop as pop

from tqdm import tqdm
from sqlalchemy import func
import numpy
from numpy import pi, cos, inf
from sqlalchemy import select

import rasterio
import cartopy.io.shapereader as shpreader
import shapely.wkb

import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

from database import db
from earth import GEODESIC



def train_model(X, y, **kwargs):
    # unpack parameters. default values match our model architecture.
    width = kwargs["width"] if "width" in kwargs else 128
    depth = kwargs["depth"] if "depth" in kwargs else 3
    dropout = kwargs["dropout"] if "dropout" in kwargs else 0.05
    early_stopping = kwargs["early_stopping"] if "early_stopping" in kwargs else None
    train_epochs = kwargs["train_epochs"] if "train_epochs" in kwargs else 200
    plot = kwargs["plot"] if "plot" in kwargs else False
    verbose = kwargs["verbose"] if "verbose" in kwargs else False

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=9
    )

    latlon_inputs_1 = keras.Input(shape=(2))
    latlon_inputs_2 = keras.Input(shape=(2))

    embedding_1 = latlon_inputs_1
    embedding_2 = latlon_inputs_2
    for i in range(depth):
        dense = Dense(width)
        prelu = PReLU()
        dout = Dropout(dropout)
        embedding_1 = dout(prelu(dense(embedding_1)))
        embedding_2 = dout(prelu(dense(embedding_2)))

    def euclidean_distance(x1, x2):
        return K.sum((x1 - x2) * (x1 - x2)) ** 0.5

    euclidean = Lambda(euclidean_distance)(embedding_1, embedding_2)

    exhuming_1 = embedding_1
    exhuming_2 = embedding_2
    for i in range(depth - 1):
        dense = Dense(width)
        prelu = PReLU()
        dout = Dropout(dropout)
        exhuming_1 = dout(prelu(dense(exhuming_1)))
        exhuming_2 = dout(prelu(dense(exhuming_2)))
    dense = Dense(2)
    prelu = PReLU()
    dout = Dropout(dropout)
    exhuming_1 = dout(prelu(dense(exhuming_1)))
    exhuming_2 = dout(prelu(dense(exhuming_2)))

    full_model = keras.Model(
        inputs=[latlon_inputs_1, latlon_inputs_2],
        outputs=[euclidean, exhuming_1, exhuming_2],
    )
    full_model.compile(
        loss=[
            tf.keras.losses.MeanAbsoluteError(),
            tf.keras.losses.MeanSquaredError(),
            tf.keras.losses.MeanSquaredError(),
        ]
    )
    es = EarlyStopping(patience=early_stopping)
    history = full_model.fit(
        X_train,
        y_train,
        epochs=train_epochs,
        validation_split=0.1,
        callbacks=[checkpointer, es],
    )

    val_losses = history.history["val_loss"]
    best_epoch = val_losses.index(min(val_losses)) + 1

    y_pred = model.predict(X)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    if verbose:
        print("TRAIN MSE: %f" % train_mse)
        print("TEST MSE: %f" % test_mse)
        print("BEST EPOCH: %i" % best_epoch)

    if plot:
        plt.scatter(y_pred_train, y_train)
        plt.title("Train")
        plt.show()

        plt.scatter(y_pred_test, y_test)
        plt.title("Test")
        plt.show()

        plt.scatter(y_pred, y)
        plt.title("All")
        plt.show()

    return best_epoch, train_mse, test_mse


def distances_from_focus(
    source,
    destination,
    filter_sources: t.Optional[t.Set[str]] = None,
    filter_nodes: bool = False,
    pred: t.Optional = None,
) -> numpy.array:
    # Dijkstra's algorithm, adapted for our purposes from
    # networkx/algorithms/shortest_paths/weighted.html,
    # extended to be an A* implementation
    seen: t.Dict[t.Tuple[int, int], float] = {source: 0.0}
    c = count()
    # use heapq with (distance, label) tuples
    fringe: t.List[t.Tuple[float, int, t.Tuple[int, int]]] = []
    source_lonlat = DATABASE.execute(
        select(
            [
                TABLES["nodes"].c.longitude,
                TABLES["nodes"].c.latitude,
            ]
        ).where(TABLES["nodes"].c.node_id == source)
    ).fetchone()
    dest_lonlat = DATABASE.execute(
        select(
            [
                TABLES["nodes"].c.longitude,
                TABLES["nodes"].c.latitude,
            ]
        ).where(TABLES["nodes"].c.node_id == destination)
    ).fetchone()
    # Speeds faster than 3m/s are really rare, even on rivers, so use geodesic
    # distance with a speed of 3m/s as heuristic
    heuristic = GEODESIC.inverse(source_lonlat, dest_lonlat)[0, 0] / 3
    push(fringe, (heuristic, 0, next(c), source))

    if filter_sources:
        filter = TABLES["edges"].c.source.in_(filter_sources)
        if filter_nodes:
            filter &= TABLES["nodes"].c.node_id > 100000000
    else:
        if filter_nodes:
            filter = TABLES["nodes"].c.node_id > 100000000
        else:
            filter = True

    dist = t.DefaultDict(lambda: inf)

    while fringe:
        (_, d, _, spot) = pop(fringe)
        if dist.get(spot, numpy.inf) < d:
            continue  # already found this node, faster, through another path
        dist[spot] = d
        if spot == destination:
            break

        for u, cost, source, lon, lat in DATABASE.execute(
            select(
                [
                    TABLES["edges"].c.node2,
                    TABLES["edges"].c.travel_time,
                    TABLES["edges"].c.source,
                    TABLES["nodes"].c.longitude,
                    TABLES["nodes"].c.latitude,
                ]
            )
            .select_from(
                TABLES["edges"].join(
                    TABLES["nodes"],
                    onclause=TABLES["edges"].c.node2 == TABLES["nodes"].c.node_id,
                )
            )
            .where(
                (
                    (
                        (TABLES["edges"].c.node1 == spot)
                        & (TABLES["edges"].c.node2 != spot)
                    )
                    if filter is True
                    else filter
                )
                & (TABLES["edges"].c.node1 == spot)
                & (TABLES["edges"].c.node2 != spot)
            )
        ):
            if u < 100000000:
                cost += 3 * 3600
            vu_dist = dist[spot] + cost
            if u in dist and vu_dist < dist[u]:
                if pred:
                    via = f"via {pred[u]} [{dist[pred[u]]}]"
                else:
                    via = ""
                print(
                    f"Contradictory paths found: Already reached u={u} at distance {dist[u]}{via} from {start}, but now finding a shorter connection {vu_dist} via {spot} [{dist[spot]}]. Do you have negative weights, or a bad heuristic?"
                )
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                heuristic = GEODESIC.inverse((lon, lat), dest_lonlat)[0, 0] / 3
                push(fringe, (vu_dist + heuristic, vu_dist, next(c), u))
                if pred is not None:
                    pred[u] = spot, source
    return dist


if __name__ == "__main__":
    DATABASE, TABLES = db(sys.argv[1])
    nodes = [
        (node, (lon, lat))
        for node, lon, lat in DATABASE.execute(
            select(
                TABLES["nodes"].c.node_id,
                TABLES["nodes"].c.longitude,
                TABLES["nodes"].c.latitude,
            )
        )
    ]

    X, y = [], []
    for i in tqdm(range(2000)):
        i_start = numpy.random.randint(len(nodes))
        i_end = numpy.random.randint(len(nodes))
        start, lonlat_start = nodes[i_start]
        end, lonlat_end = nodes[i_end]
        dist = distances_from_focus(start, end)
        X.append([lonlat_start, lonlat_end])
        y.append([dist, lonlat_start, lonlat_end])
    train_model(X, y, verbose=True, plot=True, model_name="alpha_model")
