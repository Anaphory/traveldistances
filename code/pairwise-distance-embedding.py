"""Train an embedding representing travel times in euclidean space."""
import datetime

import keras
import keras.backend as K
import numpy
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import PReLU
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def lonlat_to_3d(lonlat):
    lon = lonlat[..., 0] * numpy.pi / 180
    lat = lonlat[..., 1] * numpy.pi / 180
    return tf.stack(
        (K.sin(lon) * K.cos(lat), K.cos(lon) * K.cos(lat), K.sin(lat)),
        -1,
        name="coords_3d",
    )


def euclidean_distance(x1x2):
    x1, x2 = x1x2
    return K.sum((x1 - x2) * (x1 - x2), axis=1) ** 0.5


def identity(x):
    return x


class ReducingDropout(Dropout):
    def call(self, inputs, training=None):
        if self.rate > 1e-13:
            self.rate *= 0.95
            return super().call(inputs, training)
        return inputs


def train_model(
    train,
    test,
    example,
    widths=[128, 128],
    embedding=8,
    plot=False,
    from_3d=False,
    dropout=0.05,
    **kwargs,
):
    # unpack parameters. default values match our model architecture.
    early_stopping = kwargs["early_stopping"] if "early_stopping" in kwargs else None
    train_epochs = kwargs["train_epochs"] if "train_epochs" in kwargs else 2000
    verbose = kwargs["verbose"] if "verbose" in kwargs else False

    model_string = "embedding-{}-{}".format("-".join(str(w) for w in widths), embedding)
    if from_3d:
        model_string += "-from3d"
    if dropout < -0.5:
        model_string += "-batchnorm"
    elif dropout:
        model_string += "-dout-{:0.5f}".format(dropout)

    # Do we need the h5 format? Because this isn't generating an h5 model store.
    model_fp = f"models/{model_string:s}/"

    try:
        full_model = keras.models.load_model(model_fp)
    except OSError:
        latlon_inputs_1 = keras.Input(shape=(2))
        latlon_inputs_2 = keras.Input(shape=(2))

        if from_3d:
            embedding_1 = lonlat_to_3d(latlon_inputs_1)
            embedding_2 = lonlat_to_3d(latlon_inputs_2)
        else:
            embedding_1 = latlon_inputs_1
            embedding_2 = latlon_inputs_2
        for width in widths:
            dense = Dense(width)
            prelu = PReLU()
            if dropout < 0.5:
                dout = BatchNormalization()
            elif dropout:
                dout = ReducingDropout(dropout)
            else:
                dout = identity
            embedding_1 = dout(prelu(dense(embedding_1)))
            embedding_2 = dout(prelu(dense(embedding_2)))
        dense = Dense(embedding, name="embedding")
        embedding_1 = dense(embedding_1)
        embedding_2 = dense(embedding_2)

        euclidean = Lambda(euclidean_distance, name="euclidean_distance_in_embedding")(
            [embedding_1, embedding_2]
        )

        exhuming_1 = embedding_1
        exhuming_2 = embedding_2
        for width in widths:
            dense = Dense(width)
            prelu = PReLU()
            if dropout < 0.5:
                dout = BatchNormalization()
            elif dropout:
                dout = ReducingDropout(dropout)
            else:
                dout = identity
            exhuming_1 = dout(prelu(dense(exhuming_1)))
            exhuming_2 = dout(prelu(dense(exhuming_2)))
        dense = Dense(2, name="exhuming")
        exhuming_1 = dense(exhuming_1)
        exhuming_2 = dense(exhuming_2)
        to3d = Lambda(lonlat_to_3d, name="spherical_3d_coordinates")
        output_3d_1 = to3d(exhuming_1)
        output_3d_2 = to3d(exhuming_2)

        full_model = keras.Model(
            inputs=[latlon_inputs_1, latlon_inputs_2],
            outputs=[euclidean, output_3d_1, output_3d_2, exhuming_1, exhuming_2],
        )
        full_model.compile(
            loss=[
                tf.keras.losses.MeanSquaredLogarithmicError(),
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError(),
            ],
            loss_weights=[1.0, 1.0, 1.0, 1e-5, 1e-5],
            optimizer=Adam(),
        )

    if plot:
        tf.keras.utils.plot_model(
            full_model, show_shapes=True, to_file=f"models/{model_string:}.png"
        )

    def print_stats(epoch, logs):
        print()
        for inputs, outputs in example.as_numpy_iterator():
            for lonlat1, lonlat2, true_dist, _, _, _, _ in zip(*inputs, *outputs):
                dist, xyz1, xyz2, _, _ = full_model.predict(
                    [lonlat1[None], lonlat2[None]], verbose=0
                )
                lon1, lat1 = (
                    numpy.arctan2(xyz1[0, 0], xyz1[0, 1]) / numpy.pi * 180,
                    numpy.arcsin(xyz1[0, 2]) / numpy.pi * 180,
                )
                lon2, lat2 = (
                    numpy.arctan2(xyz2[0, 0], xyz2[0, 1]) / numpy.pi * 180,
                    numpy.arcsin(xyz2[0, 2]) / numpy.pi * 180,
                )
                print(
                    "The distance from",
                    "{:=+06.1f}/{:=+05.1f}".format(*lonlat1),
                    "to",
                    "{:=+06.1f}/{:=+05.1f}".format(*lonlat2),
                    "is",
                    numpy.format_float_scientific(true_dist, 3),
                )
                print(
                    "Predicted as:    ",
                    "{:=+06.1f}/{:=+05.1f}".format(lon1, lat1),
                    "to",
                    "{:=+06.1f}/{:=+05.1f}".format(lon2, lat2),
                    "is",
                    numpy.format_float_scientific(dist, 3),
                )
        print()
        print()

    log_dir = "logs/fit/{:s}_{:s}".format(
        model_string, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    callbacks = [
        ModelCheckpoint(model_fp, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        LambdaCallback(on_epoch_end=print_stats),
    ]
    if early_stopping:
        callbacks.append(EarlyStopping(early_stopping))

    history = full_model.fit(
        train,
        validation_data=test,
        shuffle=True,
        epochs=train_epochs,
        callbacks=callbacks,
    )

    val_losses = history.history["val_loss"]
    best_epoch = val_losses.index(min(val_losses)) + 1

    d_pred_train, _, _ = full_model.predict(train)
    d_pred_test, _, _ = full_model.predict(test)

    train_mse = mean_squared_error(train, d_pred_train)
    test_mse = mean_squared_error(test, d_pred_test)

    if verbose:
        print("TRAIN MSE: %f" % train_mse)
        print("TEST MSE: %f" % test_mse)
        print("BEST EPOCH: %i" % best_epoch)

    return best_epoch, train_mse, test_mse


BATCH = 1024


def generator(mmap, indices):
    order = list(range(0, len(indices), BATCH))
    numpy.random.shuffle(order)
    for i in order:
        ind = indices[i : i + BATCH]
        data = mmap[ind].copy()
        selector = numpy.random.random(size=len(data)) < 0.5
        data[selector][:, [0, 1, 2, 3, 4]] = data[selector][:, [0, 3, 4, 1, 2]]
        yield data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", type=int, default=512)
    parser.add_argument(
        "--widths", nargs="+", type=int, default=[1024, 512, 512, 512, 512]
    )
    parser.add_argument("--from-3d", action="store_true", default=False)
    parser.add_argument(
        "--dropout",
        type=float,
        default=-1.0,
        help="Dropout rate. For negative values, use Batch Normalization instead.",
    )
    parser.add_argument("--plot", action="store_true", default=False)
    args = parser.parse_args()

    # Overrides for interactive use
    args.from_3d = True

    y_longdistance = numpy.load(
        "distances_bigstars.npy", allow_pickle=False, mmap_mode="r"
    )
    y_local = numpy.load("distances.npy", allow_pickle=False, mmap_mode="r")
    # Make sure the training/testig data contains about 1/5 long distance data
    p_longdistance = len(y_longdistance) / len(y_longdistance) * 0.25
    y = numpy.vstack(
        (
            y_local,
            y_longdistance[
                numpy.random.random(size=len(y_longdistance)) < p_longdistance
            ],
        )
    )
    print(f"Starting MC with a total of {len(y)} data points.")
    # Split into training, testing, and validation sets
    rest, meta_test_ix = train_test_split(
        range(len(y)), test_size=5, random_state=9, shuffle=True
    )
    train_ix, test_ix = train_test_split(
        range(len(y)), test_size=0.1, random_state=9, shuffle=True
    )

    train = tf.data.Dataset.from_generator(
        lambda: generator(y, train_ix),
        output_signature=tf.TensorSpec(
            shape=(
                None,
                5,
            ),
            dtype=tf.float64,
        ),
    ).map(
        lambda row: (
            (row[..., 1:3], row[..., 3:5]),
            (
                row[..., 0],
                lonlat_to_3d(row[..., 1:3]),
                lonlat_to_3d(row[..., 3:5]),
                row[..., 1:3],
                row[..., 3:5],
            ),
        )
    )
    test = tf.data.Dataset.from_generator(
        lambda: generator(y, test_ix),
        output_signature=tf.TensorSpec(
            shape=(
                None,
                5,
            ),
            dtype=tf.float64,
        ),
    ).map(
        lambda row: (
            (row[..., 1:3], row[..., 3:5]),
            (
                row[..., 0],
                lonlat_to_3d(row[..., 1:3]),
                lonlat_to_3d(row[..., 3:5]),
                row[..., 1:3],
                row[..., 3:5],
            ),
        )
    )
    example = tf.data.Dataset.from_generator(
        lambda: generator(y, meta_test_ix),
        output_signature=tf.TensorSpec(
            shape=(
                None,
                5,
            ),
            dtype=tf.float64,
        ),
    ).map(
        lambda row: (
            (row[..., 1:3], row[..., 3:5]),
            (
                row[..., 0],
                lonlat_to_3d(row[..., 1:3]),
                lonlat_to_3d(row[..., 3:5]),
                row[..., 1:3],
                row[..., 3:5],
            ),
        )
    )

    train_model(
        train,
        test,
        example,
        verbose=True,
        plot=args.plot,
        embedding=args.embedding,
        widths=args.widths,
        from_3d=args.from_3d,
        dropout=args.dropout,
    )
