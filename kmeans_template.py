"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.
    """
    import numpy as np

    # Compare each point to every centroid using squared distance.
    # We can skip the square root since it won't change which one is closest.
    distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)

    # If two distances are equal, argmin uses the first centroid.
    assignments = np.argmin(distances, axis=1)
    return assignments


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    import numpy as np

    # Make a copy so the original centroid array stays unchanged.
    new_centroids = centroids.copy()
    for cluster_index in range(len(centroids)):
        cluster_points = data[assignments == cluster_index]

        # The new center is just the average of the points in this cluster.
        if len(cluster_points) > 0:
            new_centroids[cluster_index] = np.mean(cluster_points, axis=0)
        # If this cluster has no points, we keep its previous center.

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
    import numpy as np
    import matplotlib.pyplot as plt

    # We don't use the actual labels when clustering, just the data points.
    data, _ = make_toy_data()
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cluster_count in K_VALUES:
        # Start with random centers and let our K-means functions update them.
        initial_centroids = get_initial_clusters(cluster_count, data, rng)
        centroids, assignments = kmeans(data, initial_centroids, max_iterations)

        # Plot the points using their assigned clusters, then mark the centers.
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(data[:, 0], data[:, 1], c=assignments, cmap="tab10", s=7)
        ax.scatter(centroids[:, 0], centroids[:, 1], color="black",
                   marker="X", s=150, edgecolors="white", label="Centroids")
        ax.set_title(f"K-means with k = {cluster_count}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"kmeans_k{cluster_count}.png", dpi=170)
        plt.close(fig)

        # The loss lets us compare how the clustering changes with k.
        squared_loss = np.sum((data - centroids[assignments]) ** 2)
        print(f"k = {cluster_count}, squared-distance loss = {squared_loss:.4f}")


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
