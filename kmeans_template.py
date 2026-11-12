"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
import numpy as np
from pathlib import Path

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.
    """

    # convert to np array
    data = np.asarray(data)
    centroids = np.asarray(centroids)

    # create (n,k,d) np array with difference between point n and centroid k
    diff = data[:, None, :] - centroids[None, :, :]
    # turn diff into squared distance np array
    sq_dist = np.sum(diff**2, axis=2)  # (n,k)

    closest_centroids_idx = np.argmin(sq_dist, axis=1)  # (n,)

    # np.argmin returns the first minimum it finds if there is a tie (lowest centroid index)

    return closest_centroids_idx


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """

    data = np.asarray(data)
    assignments = np.asarray(assignments)

    centroids_copy = np.array(centroids.copy())

    k = centroids_copy.shape[0]
    n = data.shape[0]

    # iterate over centroids
    for i in range(k):
        # get the data points that are assigned to the current centroid
        data_in_centroid = []
        for j in range(n):
            if assignments[j] == i:
                data_in_centroid.append(data[j])

        if len(data_in_centroid) == 0:
            # do nothing and continue, old centroid is already in centroids_copy
            continue

        # update the centroid
        data_in_centroid = np.array(data_in_centroid)
        centroids_copy[i] = data_in_centroid.mean(axis=0)

    return centroids_copy


def kmeans(data, centroids, max_iterations=100):
    """Assignment 4's provided alternating-update loop, with a final assignment."""
    for _ in range(max_iterations):
        assignments = update_assignments(data, centroids)
        centroids = update_centroids(data, centroids, assignments)
    assignments = update_assignments(data, centroids)
    return centroids, assignments


def make_toy_data():
    """Generate the exact make_blobs configuration from Assignment 4."""
    from sklearn.datasets import make_blobs

    return make_blobs(n_samples=4000, centers=4, cluster_std=0.60, random_state=0)


def get_initial_clusters(k, data, rng):
    """Sample k rows with replacement, as in the source's randint initializer.

    This adaptation accepts a seeded NumPy generator for reproducibility.
    Repeated initial rows are possible; your empty-cluster policy must cope.
    """
    return data[rng.integers(0, data.shape[0], size=k)].copy()


def run_experiments(output_dir, seed=0, max_iterations=100):
    """Q2(b), 15 points: run all five k values and save cluster plots.

    Use the four-component truth labels only for a reference plot. K-means
    must cluster the features without access to those labels.
    """
    import matplotlib.pyplot as plt

    # Load toy data, create a seeded RNG, and make the output folder.
    data, true_labels = make_toy_data()
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reference plot with the true labels.
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, c=true_labels, s=5)
    plt.title("True labels")
    plt.savefig(output_dir / "true_labels.png")
    plt.close()

    # Run kmeans for every k and save a plot.
    for k in K_VALUES:
        initial_centroids = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, initial_centroids, max_iterations)

        plt.scatter(x, y, c=assignments, s=5)
        plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x", s=100)
        plt.title("k = " + str(k))
        plt.savefig(output_dir / ("kmeans_k" + str(k) + ".png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "kmeans",
    )
    args = parser.parse_args()
    run_experiments(args.output_dir, args.seed, args.max_iterations)


if __name__ == "__main__":
    main()
