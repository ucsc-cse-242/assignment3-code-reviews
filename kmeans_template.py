"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path

from numpy import ndarray
# from sklearn.metrics.pairwise import euclidean_distances

from utils import _euclidean_distance

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data: ndarray, centroids: ndarray)-> ndarray:
    """Return one cluster index per row, using nearest-centroid distance.

    inputs:
        @param data: array of shape (n, d) data observations
        @param centroids: array of shape (k, d) centroids
    output:
        integer array with shape (n,) representing assignment

    Equal-Distance Tie: in case of multiple occurrences of the minimum value,
    the indices corresponding to the first occurrence are returned.
    """
    # initalize matrix with distances from centroids
    # distances[i, k] = distance from point i to centroid k, shape (n, k)
    distances = ndarray(shape=(data.shape[0], centroids.shape[0]))

    # calculate the euclidean distance to each centroid
    # one column at a time: all n points vs. centroid k (broadcast in numpy)
    for k in range(centroids.shape[0]):
        distances[:, k] = _euclidean_distance(data, centroids[k])

    # assign each point to a cluster based on minimum distance to centroid
    # tie rule: return the index of the first minimum (lowest cluster index)
    # ref: https://numpy.org/doc/2.1/reference/generated/numpy.argmin.html
    assignments = distances.argmin(axis=1)
    return assignments


def update_centroids(
    data: ndarray, centroids: ndarray, assignments: ndarray
) -> ndarray:
    """Return an updated (k, d) centroid array based on cluster assignments.

    inputs:
        @param data: array of shape (n, d) data observations
        @param centroids: array of shape (k, d) centroids
        @param assignments: array with shape (n,) assigning a datapoint to a cluster


    Empty cluster policy: if no points are assigned to cluster i, retain its previous
    centroid. This can happen when the initializer samples the same row twice and
    has duplicate centroids. The tie rule sends shared points to the lower index cluster
    """
    # copy so the input centroid array is not modified in place
    new_centroids = centroids.astype(float).copy()
    for i in range(centroids.shape[0]):
        # filter data where assignment is centroid
        members = data[assignments == i]
        # empty cluster policy: if no points assigned to cluster, retain previous centroid
        if members.shape[0] == 0:
            continue
        # find mean of that data group and update centroid
        new_centroids[i] = members.mean(axis=0)
    return new_centroids


def kmeans(data: ndarray, centroids: ndarray, max_iterations:int = 100) -> tuple:
    """Assignment 4's provided alternating-update loop, with a final assignment."""
    for _ in range(max_iterations):
        assignments = update_assignments(data, centroids)
        centroids = update_centroids(data, centroids, assignments)
    assignments = update_assignments(data, centroids)
    return centroids, assignments


def make_toy_data()-> tuple:
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
    from numpy.random import default_rng

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # make_blobs returns (features, true_labels). Only the features are
    # clustered; the true labels are used only for the reference plot.
    toy_data, true_labels = make_toy_data()
    rng = default_rng(seed)

    def save_plot(labels, centroids, title, filename):
        """draw and save scatter plots"""
        fig, ax = plt.subplots(figsize=(6, 6))
        # tab10 has 10 distinct colors, which helps with display for k=10
        ax.scatter(
            toy_data[:, 0], toy_data[:, 1], c=labels, cmap="tab10",
            vmin=0, vmax=9, s=6, alpha=0.6
        )
        if centroids is not None:
            ax.scatter(
                centroids[:, 0], centroids[:, 1], c="black",
                marker="x", label="final centroids"
            )
            ax.legend()
        ax.set(xlabel="x1", ylabel="x2", title=title)
        ax.grid(alpha=0.3)
        fig.savefig(output_dir/filename)
        plt.close(fig)

    # reference plot of the four generating groups
    save_plot(
        true_labels, None, "Reference: true blob labels (4 groups)",
        "reference_true_labels.png"
    )

    results = {}
    for k in K_VALUES:
        clusters = get_initial_clusters(k, toy_data, rng)
        centroids, assignments = kmeans(toy_data, clusters, max_iterations)
        results[k] = (centroids, assignments)
        save_plot(assignments, centroids, f"K-means, k = {k}", f"kmeans_k{k}.png")
    return results


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
