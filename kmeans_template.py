"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
import json
import numpy as np
from pathlib import Path

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Equal-distance ties select the smallest centroid index.
    """
    # Squared Euclidean distance has the same minimizer as Euclidean distance.
    squared_distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(squared_distances, axis=1)  # Ties: smallest centroid index.


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Empty clusters retain their previous centroid. The input centroid array
    is not modified in place.
    """
    updated = np.array(centroids, dtype=float, copy=True)
    for cluster in range(len(centroids)):
        members = data[assignments == cluster]
        if len(members):
            updated[cluster] = members.mean(axis=0)
        # Empty cluster: retain its previous centroid.
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
    data, _ = make_toy_data()  # Generating labels are not used by clustering.
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    lower, upper = data.min(axis=0) - 0.6, data.max(axis=0) + 0.6
    for k in K_VALUES:
        initial = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, initial, max_iterations)
        inertia = float(np.sum((data - centroids[assignments]) ** 2))
        counts = np.bincount(assignments, minlength=k)
        fig, ax = plt.subplots(figsize=(6.8, 5.0), layout="constrained")
        ax.scatter(data[:, 0], data[:, 1], c=assignments, cmap="tab10",
                   vmin=-0.5, vmax=9.5, s=7, alpha=0.55, linewidths=0)
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X",
                   s=140, edgecolors="white", linewidths=0.8, label="Final centroids")
        ax.set(xlabel="Feature 1", ylabel="Feature 2", title=f"K-means: k = {k}",
               xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.2)
        fig.savefig(output_dir / f"kmeans_k{k}.png", dpi=180)
        plt.close(fig)
        summaries.append({"k": k, "inertia": inertia, "counts": counts.tolist(),
                          "centroids": centroids.tolist()})
        print(f"k={k}: inertia={inertia:.6f}; cluster sizes={counts.tolist()}")
    summary = {"seed": seed, "max_iterations": max_iterations, "runs": summaries}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summaries


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
