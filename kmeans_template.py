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

    Tie rule: if a point is equally close to two or more centroids, it is
    assigned to the centroid with the LOWEST index (np.argmin returns the
    first minimum). This also covers duplicate centroids from the
    with-replacement initializer: only the lower-index copy receives points.
    """
    import numpy as np
    data = np.asarray(data, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    # Squared Euclidean distance from every point to every centroid, shape (n, k).
    # Squaring does not change which centroid is nearest.
    diffs = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    sq_distances = np.sum(diffs ** 2, axis=2)
    return np.argmin(sq_distances, axis=1).astype(int)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.

    Each nonempty cluster's centroid becomes the mean of the points assigned
    to it. Empty-cluster rule: a cluster with no assigned points KEEPS its
    previous centroid unchanged. The input array is copied, never modified.
    """
    import numpy as np
    data = np.asarray(data, dtype=float)
    new_centroids = np.array(centroids, dtype=float, copy=True)
    for j in range(new_centroids.shape[0]):
        members = data[assignments == j]
        if members.shape[0] > 0:
            new_centroids[j] = members.mean(axis=0)
        # else: empty cluster, keep new_centroids[j] = previous centroid.
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

    # true_labels are used ONLY for the optional reference plot below.
    data, true_labels = make_toy_data()
    rng = np.random.default_rng(seed)

    def save_scatter(labels, centers, title, filename):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10",
                   vmin=0, vmax=9, s=6, alpha=0.6)
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=260,
                       c="black", edgecolors="white", linewidths=1.5,
                       label="final centroids")
            ax.legend(loc="best")
        ax.set(xlabel="x1", ylabel="x2", title=title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    for k in K_VALUES:
        initial = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, initial, max_iterations)
        sizes = np.bincount(assignments, minlength=k)
        inertia = float(np.sum((data - centroids[assignments]) ** 2))
        path = save_scatter(assignments, centroids,
                            f"K-means, k = {k} (seed {seed})", f"kmeans_k{k}.png")
        print(f"k={k:2d}  inertia={inertia:10.2f}  cluster sizes={sizes.tolist()}  "
              f"saved {path}")

    path = save_scatter(true_labels, None,
                        "Reference: true make_blobs labels (not used by K-means)",
                        "true_labels_reference.png")
    print(f"Saved reference plot {path}")


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
