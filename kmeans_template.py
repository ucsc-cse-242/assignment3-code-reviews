"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.

Student: Kangan Qian (SID 2346765)

Documented design choices:
- Distance ties (assignment step): a point at exactly equal distance from two
  centroids is assigned to the lowest centroid index (``numpy.argmin`` returns
  the first minimum), so the rule is deterministic.
- Empty clusters (centroid step): a cluster with no members keeps its previous
  centroid. Duplicated initial centroids (possible with the with-replacement
  initializer) leave one copy empty; keeping the old position lets it win
  points back in later iterations instead of crashing the run.
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
    # Pairwise (squared Euclidean) distances via the identity
    # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2, shaped (n, k). Squared
    # distances give the same argmin as plain distances and stay nonnegative.
    sq_data = np.sum(data ** 2, axis=1)[:, None]            # (n, 1)
    sq_centroids = np.sum(centroids ** 2, axis=1)[None, :]  # (1, k)
    cross = data @ centroids.T                              # (n, k)
    sq_distances = sq_data - 2.0 * cross + sq_centroids
    # Tie rule: argmin returns the lowest centroid index among exact ties.
    return np.argmin(sq_distances, axis=1).astype(int)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    data = np.asarray(data, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    k = centroids.shape[0]
    new_centroids = centroids.copy()  # empty clusters keep their old position
    for cluster in range(k):
        members = data[assignments == cluster]
        if members.shape[0] > 0:
            new_centroids[cluster] = members.mean(axis=0)
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
    import matplotlib.pyplot as plt

    data, true_labels = make_toy_data()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Reference plot with the generating labels (never seen by K-means).
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(data[:, 0], data[:, 1], c=true_labels, cmap="tab10", s=5, alpha=0.6)
    ax.set(title="Reference: make_blobs ground-truth groups (k = 4)",
           xlabel="x1", ylabel="x2")
    fig.tight_layout()
    fig.savefig(output_dir / "reference_true_labels.png", dpi=180)
    plt.close(fig)

    for k in K_VALUES:
        centroids = get_initial_clusters(k, data, rng)
        final_centroids, assignments = kmeans(data, centroids, max_iterations)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(data[:, 0], data[:, 1], c=assignments, cmap="tab10",
                   s=5, alpha=0.6)
        ax.scatter(final_centroids[:, 0], final_centroids[:, 1],
                   marker="X", s=220, color="black", edgecolors="white",
                   linewidths=1.5, label="final centroids", zorder=3)
        ax.set(title=f"K-means result, k = {k} (seed = {seed}, "
                     f"max_iterations = {max_iterations})",
               xlabel="x1", ylabel="x2")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"kmeans_k{k}.png", dpi=180)
        plt.close(fig)

        sizes = np.bincount(assignments, minlength=k)
        print(f"k = {k:2d}: final centroids saved, cluster sizes = {sizes.tolist()}")

    print(f"Saved reference plot and {len(K_VALUES)} cluster plots to {output_dir}")


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
