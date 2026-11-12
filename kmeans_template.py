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

    Distance is squared Euclidean, which ranks centroids the same way as
    Euclidean distance. If a point is equally close to several centroids,
    it is assigned to the smallest centroid index (NumPy argmin).
    """
    import numpy as np

    data = np.asarray(data, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    differences = data[:, None, :] - centroids[None, :, :]
    squared_distances = np.sum(differences * differences, axis=2)
    return np.argmin(squared_distances, axis=1)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Each nonempty cluster is replaced by the mean of its assigned rows.
    An empty cluster keeps its previous centroid. That covers duplicate
    initial centers, which the with-replacement initializer can produce.
    The input centroid array is copied and is not modified in place.
    """
    import numpy as np

    data = np.asarray(data, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    assignments = np.asarray(assignments)
    updated = centroids.copy()
    for cluster in range(centroids.shape[0]):
        members = data[assignments == cluster]
        if len(members) > 0:
            updated[cluster] = members.mean(axis=0)
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
    import numpy as np

    data, true_labels = make_toy_data()
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    panels = [("Generating groups (not used by K-means)", data, true_labels, None)]
    for k in K_VALUES:
        initial = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, initial, max_iterations)
        path = output_dir / f"kmeans_k{k}.png"
        _save_cluster_plot(
            data, assignments, centroids, f"K-means assignments, k = {k}", path,
        )
        sizes = np.bincount(assignments, minlength=k)
        print(f"k={k}: cluster sizes {sizes.tolist()}, saved {path}")
        saved.append(path)
        panels.append((f"k = {k}", data, assignments, centroids))

    reference_path = output_dir / "reference_labels.png"
    _save_cluster_plot(
        data, true_labels, None, "Reference: four generating groups", reference_path,
    )
    overview_path = output_dir / "kmeans_overview.png"
    _save_overview(panels, overview_path)
    print(f"Saved reference plot to {reference_path}")
    print(f"Saved overview to {overview_path}")
    return saved


def _save_cluster_plot(data, labels, centroids, title, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=8, alpha=0.75, linewidths=0)
    if centroids is not None:
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            c="black", marker="X", s=140, linewidths=1.2,
            edgecolors="white", label="centroids", zorder=3,
        )
        ax.legend(loc="best")
    ax.set(xlabel="x1", ylabel="x2", title=title)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_overview(panels, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (title, data, labels, centroids) in zip(axes.ravel(), panels):
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=4, alpha=0.7, linewidths=0)
        if centroids is not None:
            ax.scatter(
                centroids[:, 0], centroids[:, 1],
                c="black", marker="X", s=70, linewidths=0.8,
                edgecolors="white", zorder=3,
            )
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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
