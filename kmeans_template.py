"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
import matplotlib.pyplot as plt
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
    assert isinstance(data, np.ndarray) and data.ndim == 2
    assert (isinstance(centroids, np.ndarray) and centroids.ndim == 2
            and centroids.shape[1] == data.shape[1])
    # Compute distances and assign each point to a centroid.
    N, K = data.shape[0], centroids.shape[0]
    distances = np.zeros((N, K))
    for j in range(K):
        distances[:, j] = np.sum((data - centroids[j]) ** 2, axis=1)
    # argmin returns the first minimum, so ties go to the lowest cluster index
    return np.argmin(distances, axis=1)
    # raise NotImplementedError("Implement the assignment step from scratch.")


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    An empty cluster retains its previous centroid.
    """
    assert isinstance(data, np.ndarray) and data.ndim == 2
    assert (isinstance(centroids, np.ndarray) and centroids.ndim == 2
            and centroids.shape[1] == data.shape[1])
    assert isinstance(assignments, np.ndarray) and assignments.shape == (len(data),)
    # Update the centroid of each nonempty cluster.
    new_centroids = np.array(centroids, dtype=float, copy=True)
    for k in range(new_centroids.shape[0]):
        cluster_members = data[assignments == k]
        # a cluster with no members keeps its previous centroid
        if cluster_members.shape[0] > 0:
            new_centroids[k] = cluster_members.mean(axis=0)
    return new_centroids
    # raise NotImplementedError("Implement the centroid step from scratch.")


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


def plot_clusters(K, output_path, data, assignments, centroids=None, title=None):
    assert isinstance(data, np.ndarray) and data.ndim == 2
    assert isinstance(assignments, np.ndarray) and assignments.shape == (len(data),)
    if centroids is not None:
        assert (isinstance(centroids, np.ndarray) and centroids.ndim == 2
                and centroids.shape[1] == data.shape[1])

    for k in range(K):
        cluster_members = data[assignments == k]
        plt.scatter(cluster_members[:, 0], cluster_members[:, 1],
                    s=10, alpha=0.35,
                    label=f"cluster {k}")
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    color="black", edgecolors="white", marker="X", s=100,
                    label="centroids")
    if title:
        plt.title(title)
    plt.legend(fontsize="small")
    plt.axis('equal')
    plt.savefig(output_path)
    plt.close()


def run_experiments(output_dir, seed=0, max_iterations=100):
    """Q2(b), 15 points: run all five k values and save cluster plots.

    Use the four-component truth labels only for a reference plot. K-means
    must cluster the features without access to those labels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load toy data and create a seeded RNG.
    data, true_labels = make_toy_data()
    rng = np.random.default_rng(seed)
    plot_clusters(4, output_dir / "true_clusters.png",
                  data, true_labels, title="True clusters (generating groups)")

    # Run kmeans for every value in K_VALUES.
    for K in K_VALUES:
        initial_clusters = get_initial_clusters(K, data, rng)
        centroids, assignments = kmeans(data, initial_clusters, max_iterations)
        # Plot each set of cluster assignments and its final centroids.
        # Save plots under output_dir and discuss the effect of k.
        plot_clusters(K, output_dir / f"kmeans_k{K}.png",
                      data, assignments, centroids, title=f"K-means K={K}")

        total = 0.0
        for k in range(centroids.shape[0]):
            cluster_members = data[assignments == k]
            total += np.sum((cluster_members - centroids[k]) ** 2)
        print(f"K={K} variance:", total / (data.shape[0] * data.shape[1]))

    # raise NotImplementedError("Complete the K-means experiment and plots.")


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
