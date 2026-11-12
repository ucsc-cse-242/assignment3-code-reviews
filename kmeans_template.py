"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Name: Astra Tsai
Student ID: 2000926

Implement the two update functions from scratch, then complete the experiment.
AI assistance: OpenAI Codex helped implement and check this solution.
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
    # Squared Euclidean distance has the same minimizer as distance.
    # argmin chooses the smallest centroid index in an exact distance tie.
    squared_distances = np.sum(
        (np.asarray(data)[:, None, :] - np.asarray(centroids)[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(squared_distances, axis=1)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    data = np.asarray(data)
    assignments = np.asarray(assignments)
    # Floating output avoids truncating means when initial rows are integers.
    updated = np.array(centroids, dtype=float, copy=True)
    for cluster in range(len(updated)):
        members = data[assignments == cluster]
        if len(members):
            updated[cluster] = members.mean(axis=0)
        # Otherwise retain the previous centroid, including duplicate starts.
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

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, _ = make_toy_data()
    rng = np.random.default_rng(seed)
    results = {}
    for k in K_VALUES:
        initial = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, initial, max_iterations)
        inertia = float(np.sum((data - centroids[assignments]) ** 2))
        fig, ax = plt.subplots(figsize=(5.6, 4.3))
        ax.scatter(data[:, 0], data[:, 1], c=assignments, cmap="tab10",
                   vmin=-0.5, vmax=9.5, s=5, alpha=0.65, linewidths=0)
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=150,
                   c="black", edgecolors="white", linewidths=1,
                   label="Final centroids", zorder=3)
        ax.set(xlabel="x1", ylabel="x2", title=f"K-means: k = {k}",
               xlim=(data[:, 0].min() - 0.3, data[:, 0].max() + 0.3),
               ylim=(data[:, 1].min() - 0.3, data[:, 1].max() + 0.3))
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        path = output_dir / f"kmeans_k{k}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        results[k] = (centroids, assignments)
        print(f"k={k}: inertia={inertia:.6f}; saved {path}")
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
