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
    # distance from every point to every centroid -> shape (n, k)
    dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    # argmin picks the first (lowest index) centroid if there's a tie
    return np.argmin(dists, axis=1)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    import numpy as np
    new_centroids = centroids.copy()
    for j in range(len(centroids)):
        members = data[assignments == j]
        if len(members) > 0:
            new_centroids[j] = members.mean(axis=0)
        # if a cluster ends up empty it just keeps its old centroid
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, true_labels = make_toy_data()  # true labels only used for the reference plot
    rng = np.random.default_rng(seed)

    def plot(ax, labels, title, centroids=None):
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", vmin=0, vmax=9, s=4, alpha=0.6)
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=160, c="black", edgecolors="white")
        ax.set_title(title)
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")

    fig, ax = plt.subplots(figsize=(6, 5))
    plot(ax, true_labels, "True blobs (reference only)")
    fig.savefig(output_dir / "kmeans_reference_truth.png", dpi=150)
    plt.close(fig)

    results = {}
    for k in K_VALUES:
        centroids, assignments = kmeans(data, get_initial_clusters(k, data, rng), max_iterations)
        results[k] = (centroids, assignments)
        sse = ((data - centroids[assignments]) ** 2).sum()
        print(f"k={k}: SSE={sse:.1f}, cluster sizes={np.bincount(assignments, minlength=k).tolist()}")

        fig, ax = plt.subplots(figsize=(6, 5))
        plot(ax, assignments, f"K-means, k = {k}", centroids)
        fig.savefig(output_dir / f"kmeans_k{k}.png", dpi=150)
        plt.close(fig)

    # all in one figure for the writeup
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    plot(axes[0, 0], true_labels, "True blobs")
    for ax, k in zip(axes.flat[1:], K_VALUES):
        plot(ax, results[k][1], f"k = {k}", results[k][0])
    fig.tight_layout()
    fig.savefig(output_dir / "kmeans_all.png", dpi=130)
    plt.close(fig)
    print(f"Plots saved to {output_dir}")


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
