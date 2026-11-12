"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path

import numpy as np

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.
    """
    data = np.asarray(data, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    if data.ndim != 2 or centroids.ndim != 2:
        raise ValueError("data and centroids must both be two-dimensional.")
    if data.shape[1] != centroids.shape[1] or centroids.shape[0] == 0:
        raise ValueError("data and centroids must have compatible nonempty shapes.")

    # Squared distances are sufficient, and argmin selects the lowest-index
    # centroid consistently when distances are equal.
    squared_distances = np.sum(
        (data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(squared_distances, axis=1)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    data = np.asarray(data, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    assignments = np.asarray(assignments)
    if data.ndim != 2 or centroids.ndim != 2:
        raise ValueError("data and centroids must both be two-dimensional.")
    if data.shape[1] != centroids.shape[1]:
        raise ValueError("data and centroids must have the same number of features.")
    if assignments.shape != (data.shape[0],):
        raise ValueError("assignments must contain one cluster index per row.")
    if not np.issubdtype(assignments.dtype, np.integer):
        raise ValueError("assignments must contain integer cluster indices.")
    if np.any(assignments < 0) or np.any(assignments >= centroids.shape[0]):
        raise ValueError("assignments contain an invalid cluster index.")

    updated = centroids.copy()
    for cluster_index in range(centroids.shape[0]):
        members = data[assignments == cluster_index]
        if members.shape[0] > 0:
            updated[cluster_index] = np.mean(members, axis=0)
        # Empty clusters retain their previous centroid.
    return updated


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

    data, _ = make_toy_data()
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for k in K_VALUES:
        initial_centroids = get_initial_clusters(k, data, rng)
        final_centroids, assignments = kmeans(
            data, initial_centroids, max_iterations=max_iterations
        )
        results[k] = (final_centroids, assignments)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            data[:, 0],
            data[:, 1],
            c=assignments,
            cmap="tab20",
            s=8,
            alpha=0.55,
        )
        ax.scatter(
            final_centroids[:, 0],
            final_centroids[:, 1],
            c=np.arange(k),
            cmap="tab20",
            marker="X",
            edgecolors="black",
            linewidths=1.0,
            s=140,
            label="Final centroids",
        )
        ax.set(
            xlabel="Feature 1",
            ylabel="Feature 2",
            title=f"K-means clustering (k={k})",
        )
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"kmeans_k{k}.png", dpi=180)
        plt.close(fig)

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
