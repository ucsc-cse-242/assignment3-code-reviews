"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.
    """
    # Compute distances and assign each point to a centroid.
    distances = ((data[:, None] - centroids[None, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # Update the centroid of each nonempty cluster.
    updated_centroids = centroids.copy()
    for k in range(centroids.shape[0]):
        mask = assignments == k
        if mask.any():
            updated_centroids[k] = data[mask].mean(axis=0)
    # Handle empty clusters using your documented policy.
    # An empty cluster retains its previous centroid.
    return updated_centroids


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
    # Load toy data, create a seeded RNG, and initialize centroids.
    X, y = make_toy_data()
    rng = np.random.default_rng(seed)
    for k in K_VALUES:
        initial_centroids = get_initial_clusters(k, X, rng)
        # Run kmeans for every value in K_VALUES.
        final_centroids, assignments = kmeans(X, initial_centroids, max_iterations)
        # Plot each set of cluster assignments and its final centroids.
        plt.figure(figsize=(6, 6))
        plt.scatter(X[:, 0], X[:, 1], c=assignments, s=10, cmap="tab10")
        plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c="black", s=100, marker="X")
        plt.title(f"K-means clustering with k={k}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        plt.grid(alpha=0.3)
        # Save plots under output_dir and discuss the effect of k.
        output_path = Path(output_dir) / f"kmeans_k{k}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=180)
        plt.close()
    # Make truth reference plot for comparison
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
    plt.title("Ground truth clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    plt.grid(alpha=0.3)
    output_path = Path(output_dir) / "kmeans_truth.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
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
