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
    
    diffs = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.sum(diffs ** 2, axis=2)
    
    return np.argmin(distances, axis=1).astype(int)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    new_centroids = np.array(centroids, dtype=float, copy=True)
    for j in range(new_centroids.shape[0]):
        members = data[assignments == j]
        if len(members) > 0:
            new_centroids[j] = members.mean(axis=0)

    # empty cluster keeps the old centroid from the copy 
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, true_labels = make_toy_data()
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional reference plot of the generating groups (not used to cluster).
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data[:, 0], data[:, 1], c=true_labels, cmap="tab10", s=5)
    ax.set(title="Reference: true make_blobs groups", xlabel="x1", ylabel="x2")
    fig.tight_layout()
    fig.savefig(output_dir / "reference_true_groups.png", dpi=150)
    plt.close(fig)

    results = {}
    for k in K_VALUES:
        initial = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, initial, max_iterations)
        results[k] = (centroids, assignments)
        sizes = np.bincount(assignments, minlength=k)
        inertia = float(np.sum((data - centroids[assignments]) ** 2))
        print(f"k={k:2d}: cluster sizes {sizes.tolist()}, "
              f"within-cluster SSE {inertia:.1f}")

        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_clusters(ax, data, centroids, assignments, k)
        fig.tight_layout()
        path = output_dir / f"kmeans_k{k}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  saved {path}")

    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(5 * len(K_VALUES), 4.5))
    for ax, k in zip(axes, K_VALUES):
        _plot_clusters(ax, data, *results[k], k)
    fig.tight_layout()
    path = output_dir / "kmeans_all.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved combined plot to {path}")
    return results


def _plot_clusters(ax, data, centroids, assignments, k):
    """Scatter points colored by cluster, with final centroids as black Xs."""
    ax.scatter(data[:, 0], data[:, 1], c=assignments, cmap="tab10",
               vmin=0, vmax=9, s=5)
    ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X",
               s=160, edgecolors="white", linewidths=1.5, label="centroids")
    ax.set(title=f"K-means, k = {k}", xlabel="x1", ylabel="x2")
    ax.legend(loc="upper right")


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
