"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.
    """
    # number of points
    n = data.shape[0]
    # number of clusters
    k = centroids.shape[0]

    assignments = np.zeros(n, dtype=int)

    # for each point assign it to the closest cluster
    for i in range(n):
        min_dist = float("inf")
        cluster_idx = 0
        # find the cluster that has minimal distance
        for j in range(k):
            # euclidean distance
            dist = np.sum((data[i] - centroids[j]) ** 2)
            # update if there's a closer cluster

            # Ties: if it's is equally close to multiple clusters,
            # this would choose the cluster with the smallest index
            if dist < min_dist:
                min_dist = dist
                cluster_idx = j
        assignments[i] = cluster_idx
    return assignments


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """

    # number of clusters
    k = centroids.shape[0]

    # to not modify inplace
    new_centroids = centroids.copy()

    for i in range(k):

        # filter points assigned to this cluster
        points = data[assignments == i]

        # update nonempty clusters (retain the previous if it's empty)
        if len(points) > 0:
            # average all the points down each column
            new_centroids[i] = np.mean(points, axis=0)

    return new_centroids


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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load data
    data, _ = make_toy_data()

    rng = np.random.default_rng(seed)

    results = []

    # run for each k
    for k in K_VALUES:
        centroids = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, centroids, max_iterations)
        results.append((k, centroids, assignments))

    # plot the results
    for k, centroids, assignments in results:
        plt.figure()
        # Plot data points
        plt.scatter(
            data[:, 0], data[:, 1],
            c=assignments, cmap="tab10", s=5
        )
        # Plot final centroids
        plt.scatter(
            centroids[:, 0], centroids[:, 1],
            c="black", marker="X", s=150
        )
        plt.title(f"K-Means (k={k})")
        plt.savefig(f"{output_dir}/kmeans_{k}.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "kmeans")
    args = parser.parse_args()
    run_experiments(args.output_dir, args.seed, args.max_iterations)


if __name__ == "__main__":
    main()
