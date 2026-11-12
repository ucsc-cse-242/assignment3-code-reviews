"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.
    """
    # TODO: Compute distances and assign each point to a centroid.
    retValue = []

    for point in data:
        curCentroid = float("inf")
        centroidIndex = None
        for i in range(len(centroids)):
            distance = np.linalg.norm(point - centroids[i])
            if distance < curCentroid:
                curCentroid = distance
                centroidIndex = i

        retValue.append(centroidIndex)

    return np.array(retValue, dtype=int)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # TODO: Update the centroid of each nonempty cluster.

    # Created a new array with a copy of the current centroids array
    # Since numpy array is fixed, we can create a copy to ensure that the returned array is the same size as previous centroids
    new_centroids = centroids.astype(float)

    for i in range(len(centroids)):
        # Obtain the cluster point coordinates that are connected to the current centroid
        cluster_points = data[assignments == i]

        # If the cluster is not empty, obtain the mean of the XY coordinates that is associated with the centroid and apply it to the aray
        if len(cluster_points) > 0:
            new_centroids[i] = cluster_points.mean(axis=0)

    return new_centroids

    # TODO: Handle empty clusters using your documented policy.


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
    # TODO: Load toy data, create a seeded RNG, and initialize centroids.
    # Call make_toy_data()
    # Call get_inital_clusters(k, data, seed)

    # Create directory within current file location
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data, and obtaining the coordinates from toy_data tuple
    toy_data = make_toy_data()
    data = toy_data[0]

    # Need np rng data type for get_initial_clusters
    rng = np.random.default_rng(seed)

    for current_k in K_VALUES:
        # Each current k value is used for clustering and kmeans
        initial_centroids = get_initial_clusters(current_k, data, rng)
        current_centroids, current_assignments = kmeans(data, initial_centroids, 100)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            data[:, 0],
            data[:, 1],
            c=current_assignments,
            cmap="viridis",
            s=10,
            alpha=0.6,
        )

        plt.scatter(
            current_centroids[:, 0],
            current_centroids[:, 1],
            c="red",
            marker=MarkerStyle("X"),
            s=150,
            edgecolors="black",
            label="Centroids",
        )

        plt.title(f"K-means clustering with k={current_k}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"kmeans_k{current_k}.png")
        plt.close()

    # TODO: Run kmeans for every value in K_VALUES.
    # TODO: Plot each set of cluster assignments and its final centroids.
    # TODO: Save plots under output_dir and discuss the effect of k.

    """
    From obtaining the different results based on $k$, we can see that the more centroids were created, the more diverse the data was. 
    $k_4$ was able to split the data equally into its respective sections. However, when $k > 4$, it begins to over-cluster the data, causing splits between already classified clusters. 
    Depending on whether we need to identify a subgroup within the clusters, the over-clustering could lead to misinformation.
    """


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
