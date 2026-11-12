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
    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    new_centroids = centroids.copy()
    for j in range(centroids.shape[0]):
        members = data[assignments == j]
        if len(members) > 0:
            new_centroids[j] = members.mean(axis=0)
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


def plot_clusters(data, centroids, assignments, title, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(data[:, 0], data[:, 1], c=assignments, cmap="tab10", s=6, alpha=0.6)
    ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", s=160)
    ax.set(title=title, xlabel="x1", ylabel="x2")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_experiments(output_dir, seed=0, max_iterations=100):
    """Q2(b), 15 points: run all five k values and save cluster plots.

    Use the four-component truth labels only for a reference plot. K-means
    must cluster the features without access to those labels.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, truth = make_toy_data()
    rng = np.random.default_rng(seed)

    for k in K_VALUES:
        initial = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, initial, max_iterations)
        sizes = np.bincount(assignments, minlength=k)
        inertia = ((data - centroids[assignments]) ** 2).sum()
        print(f"k={k:>2}  cluster sizes={sizes.tolist()}  inertia={inertia:.1f}")
        plot_clusters(data, centroids, assignments, f"K-means, k = {k}",
                      output_dir / f"kmeans_k{k}.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(data[:, 0], data[:, 1], c=truth, cmap="tab10", s=6, alpha=0.6)
    ax.set(title="Reference: true generating groups", xlabel="x1", ylabel="x2")
    fig.savefig(output_dir / "true_groups.png", dpi=150)
    plt.close(fig)
    print(f"Saved plots to {output_dir}")


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
