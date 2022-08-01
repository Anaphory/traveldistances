"""Plot embedding model results."""
import argparse

import keras.models
import numpy
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

X = 800
Y = 400
if True:  # __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stored_model",
        nargs="?",
        default="models/embedding-1024-512-512-512-512-512-from3d-batchnorm/",
    )
    args = parser.parse_args([])

    full_model = keras.models.load_model(args.stored_model)

    embedding_model = keras.Model(
        inputs=[full_model.input[0]],
        outputs=[full_model.get_layer("euclidean_distance_in_embedding").input[0]],
    )
    hidden = [
        embedding_model.get_layer(index=d).output
        for d, dense in enumerate(embedding_model.layers)
        if type(dense) == keras.layers.core.dense.Dense
    ]
    embedding_inspector_model = keras.Model(
        inputs=[full_model.input[0]],
        outputs=hidden,
    )

    mesh_coords = (
        numpy.stack(
            numpy.meshgrid(numpy.linspace(-180, 180, X), numpy.linspace(-90, 90, Y))
        )
        .transpose((1, 2, 0))
        .reshape(-1, 2)
    )
    results = embedding_model.predict(mesh_coords)
    pca = PCA()
    pca.fit(results)

    cumulative_explained_variance = numpy.cumsum(pca.explained_variance_)
    all_variance = cumulative_explained_variance[-1]
    until = numpy.argmax(cumulative_explained_variance / all_variance > 0.97)

    per_layer_results = embedding_inspector_model.predict(mesh_coords)
    for il, layer_state in enumerate(tqdm(per_layer_results)):
        pca = PCA()
        pca.fit(layer_state)

        # Our best model has 0.06 variance, so cut off at half that – 0.03
        cumulative_explained_variance = numpy.cumsum(pca.explained_variance_)
        all_variance = cumulative_explained_variance[-1]
        until = numpy.argmax(cumulative_explained_variance / all_variance > 0.97)
        plt.plot(
            cumulative_explained_variance[: until + 1] / all_variance,
            label=f"Hidden layer {il:d}" if il else "Input",
        )
    plt.plot(
        cumulative_explained_variance[: until + 1] / all_variance, label="Embedding"
    )

    plt.legend()
    plt.show()
