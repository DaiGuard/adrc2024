import argparse

from xy_dataset import XYDataset
import numpy as np
import cv2
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def plot_histogram(dataset):

    value = []
    for image, xy, _, _ in dataset:
        value.append(xy[1])
    value = np.array(value)

    plt.hist(value, bins=3, range=(-1.0, 1.0))
    plt.show()

def plot_umap(dataset):

    images = []
    classes = []
    for image, xy, _, _ in dataset:
        if xy[1] < -0.3:
            classes.append(0)
        elif xy[1] > 0.3:
            classes.append(1)
        else:
            classes.append(2)

        image = cv2.resize(image, (224, 224))
        images.append(image.flatten())
    images = np.array(images)
    classes = np.array(classes)

    scaler = StandardScaler()
    images_scaled = scaler.fit_transform(images)

    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    # reducer = umap.UMAP(n_neighbors=5, min_dist=10.0, metric="euclidean")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
    )
    embedding = reducer.fit_transform(images_scaled)
    labels = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=50,
    ).fit_predict(embedding)
    clustered = (labels >= 0)

    fig = plt.figure()
    sub1 = fig.add_subplot(1, 2, 1)
    sub2 = fig.add_subplot(1, 2, 2)

    sub1.scatter(embedding[~clustered, 0],
            embedding[~clustered, 1],
            color=(0.5, 0.5, 0.5),
            s=1,
            alpha=0.5)
    sub1.scatter(embedding[clustered, 0],
                embedding[clustered, 1],
                c=labels[clustered],
                s=1,
                cmap='Spectral')

    left_class = (classes == 0)
    right_class = (classes == 1)
    center_class = (classes == 2)

    sub2.scatter(embedding[left_class, 0], 
                 embedding[left_class, 1], 
                 color="blue", s=1,  label="left")
    sub2.scatter(embedding[right_class, 0], 
                 embedding[right_class, 1], 
                 color="red", s=1,  label="right")
    sub2.scatter(embedding[center_class, 0], 
                 embedding[center_class, 1], 
                 color="green", s=1, label="center")

    sub1.legend(loc="upper right")
    sub2.legend(loc="upper right")

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="dataset analysis")
    parser.add_argument("--data", "-d", action="append", required=True, type=str)
    args = parser.parse_args()

    # データセットの読込み
    dataset = XYDataset(args.data)

    # plot_umap(dataset)
    plot_histogram(dataset)

         
