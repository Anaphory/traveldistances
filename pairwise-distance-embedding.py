import sys
import typing as t
from heapq import heappop as pop
from heapq import heappush as push
from itertools import count

import cartopy.geodesic as geodesic
import keras
import keras.backend as K
import numpy
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import PReLU
from matplotlib import pyplot as plt
from numpy import inf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sqlalchemy import select
from tqdm import tqdm

from database import db

GEODESIC: geodesic.Geodesic = geodesic.Geodesic()


def train_model(y, **kwargs):
    # unpack parameters. default values match our model architecture.
    width = kwargs["width"] if "width" in kwargs else 128
    depth = kwargs["depth"] if "depth" in kwargs else 3
    dropout = kwargs["dropout"] if "dropout" in kwargs else 0.05
    early_stopping = kwargs["early_stopping"] if "early_stopping" in kwargs else None
    train_epochs = kwargs["train_epochs"] if "train_epochs" in kwargs else 200
    plot = kwargs["plot"] if "plot" in kwargs else False
    verbose = kwargs["verbose"] if "verbose" in kwargs else False

    y_train, y_test, _, _ = train_test_split(y, y, test_size=0.1, random_state=9)

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

    def euclidean_distance(x1x2):
        x1, x2 = x1x2
        return K.sum((x1 - x2) * (x1 - x2), axis=1) ** 0.5

    euclidean = Lambda(euclidean_distance)([embedding_1, embedding_2])

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
        ],
        optimizer="adam",
    )
    tf.keras.utils.plot_model(full_model, show_shapes=True)

    X_train = y_train[:, 1:]
    model_fp = "%s/%s.hdf5" % ("models", "embed")
    callbacks = [ModelCheckpoint(model_fp, save_best_only=True)]
    if early_stopping:
        callbacks.append(EarlyStopping(early_stopping))
    history = full_model.fit(
        [X_train[:, 0:2], X_train[:, 2:4]],
        [y_train[:, 0], y_train[:, 1:3], y_train[:, 3:5]],
        epochs=train_epochs,
        validation_split=0.1,
        callbacks=callbacks,
    )

    val_losses = history.history["val_loss"]
    best_epoch = val_losses.index(min(val_losses)) + 1

    y_pred = full_model.predict((y[:, 1:3], y[:, 3:5]))
    y_pred_train = full_model.predict((y_train[:, 1:3], y_train[:, 3:5]))
    y_pred_test = full_model.predict((y_test[:, 1:3], y_train[:, 3:5]))
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
                        & (TABLES["edges"].c.travel_time >= 0)
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
    y = [
        (
            1636622.1629435765,
            -55.78125000000057,
            -7.25625000000074,
            -41.343750000000796,
            -9.697916666667346,
        ),
        (
            980861.3498341956,
            -59.331250000000466,
            -4.993750000000787,
            -53.531250000000625,
            -3.597916666667473,
        ),
        (
            1539035.1210719983,
            -40.72513888888889,
            -5.82097222222222,
            -51.86041666666728,
            -0.10625000000088392,
        ),
        (
            2685168.1103357156,
            -35.23958333333425,
            -5.781250000000789,
            -56.906250000000455,
            -6.014583333334123,
        ),
        (
            909709.7219902322,
            -59.2937500000005,
            -5.797916666667383,
            -53.90430555555556,
            -5.200138888888887,
        ),
        (
            1391485.1291209944,
            -49.981250000000614,
            -2.6562500000007745,
            -54.7020833333338,
            4.539583333332416,
        ),
        (
            725329.9977462677,
            -52.89791666666724,
            -9.989583333334068,
            -52.47291666666729,
            -5.443750000000826,
        ),
        (
            816665.7067046541,
            -50.793750000000614,
            -0.7520833333341628,
            -54.493750000000546,
            2.7270833333324163,
        ),
        (
            914174.6743733712,
            -56.1895833333339,
            -7.264583333334102,
            -59.96458333333385,
            -3.585416666667541,
        ),
        (
            762551.2793549526,
            -49.34375000000057,
            -0.452083333334123,
            -46.835416666667385,
            -1.239583333334167,
        ),
        (
            762551.2793549526,
            -49.34375000000057,
            -0.452083333334123,
            -46.835416666667385,
            -1.239583333334167,
        ),
        (
            1451798.5695275445,
            -59.858472222222225,
            -3.337638888888888,
            -51.256250000000534,
            -1.5187500000008711,
        ),
        (
            2347127.421323842,
            -58.9520833333338,
            3.6354166666657264,
            -49.306250000000716,
            -8.602083333334072,
        ),
        (
            2130689.4799715267,
            -50.11041666666728,
            -0.8729166666674217,
            -59.89597222222223,
            8.191527777777779,
        ),
    ]

    train_model(numpy.array(y), verbose=True, plot=True, model_name="alpha_model")


if __name__ == "__new_main__":
    DATABASE, TABLES = db(sys.argv[1])
    nodes = [
        (node, (lon, lat))
        for node, lon, lat in DATABASE.execute(
            select(
                TABLES["nodes"].c.node_id,
                TABLES["nodes"].c.longitude,
                TABLES["nodes"].c.latitude,
            ).where(
                (-60 < TABLES["nodes"].c.longitude)
                & (TABLES["nodes"].c.longitude < -30)
                & (-10 < TABLES["nodes"].c.latitude)
                & (TABLES["nodes"].c.latitude < 10)
            )
        )
    ]

    y = [], []
    for i in tqdm(range(20)):
        i_start = numpy.random.randint(len(nodes))
        i_end = numpy.random.randint(len(nodes))
        start, lonlat_start = nodes[i_start]
        end, lonlat_end = nodes[i_end]
        dist = distances_from_focus(start, end)
        y.append([dist, lonlat_start, lonlat_end])
        print(dist[end], lonlat_start, lonlat_end)
