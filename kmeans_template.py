"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

# Outside-help acknowledgment: this solution was written with the help of
# Claude, a generative AI assistant made by Anthropic.

import argparse
import csv
from pathlib import Path

import numpy as np

K_VALUES = (2, 3, 4, 6, 10)

# Ten distinct named colours (Matplotlib's "tab10" palette); enough for k <= 10.
CLUSTER_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red",
                  "tab:purple", "tab:brown", "tab:pink", "tab:gray",
                  "tab:olive", "tab:cyan")


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.

    Method: build the (n, k) matrix of squared Euclidean distances with
    broadcasting, then take the argmin of each row. Squaring does not change
    which centroid is nearest, so the square root is skipped.

    Tie rule: if a point is equally close to several centroids, it goes to the
    one with the smallest index. np.argmin returns the first minimum, so this
    rule is deterministic. The differences are computed directly (no
    ||x||^2 - 2 x.c + ||c||^2 expansion), so exact ties, such as those from
    duplicate centroids, stay exact.
    """
    data = np.asarray(data, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    # (n, 1, d) - (1, k, d) -> (n, k, d), then sum over d -> (n, k)
    differences = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    squared_distances = np.sum(differences * differences, axis=2)
    return np.argmin(squared_distances, axis=1).astype(int)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.

    Method: each non-empty cluster's centroid becomes the mean of the points
    assigned to it.

    Empty-cluster policy: a cluster that received no points keeps its previous
    centroid. This can happen when the initializer samples the same row twice.
    The tie rule then gives all the shared points to the lower index, and the
    duplicate is left empty. The retained centroid stays in place and can win
    points again once the other centroids move. The number of clusters stays
    at k and no randomness is added.

    The result is written to a new array, so the caller's centroids are never
    modified in place.
    """
    data = np.asarray(data, dtype=float)
    assignments = np.asarray(assignments)
    new_centroids = np.array(centroids, dtype=float, copy=True)
    for cluster in range(new_centroids.shape[0]):
        members = data[assignments == cluster]
        if members.shape[0] > 0:
            new_centroids[cluster] = members.mean(axis=0)
        # else: empty cluster -> keep the previous centroid (already copied).
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


def within_cluster_sse(data, centroids, assignments):
    """Sum of squared distances from each point to its assigned centroid."""
    residuals = np.asarray(data, dtype=float) - np.asarray(centroids)[assignments]
    return float(np.sum(residuals * residuals))


def _import_pyplot():
    """Import Matplotlib with a non-interactive backend (works without a display)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib could not be imported. Use the README's clean virtual "
            "environment and install requirements.txt."
        ) from exc
    return plt


def _draw_clusters(ax, data, assignments, centroids, title):
    """Scatter the points coloured by cluster and mark each final centroid."""
    k = centroids.shape[0]
    for cluster in range(k):
        members = data[assignments == cluster]
        ax.scatter(members[:, 0], members[:, 1], s=5, alpha=0.55,
                   color=CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)],
                   linewidths=0, rasterized=True)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=230,
               c=[CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in range(k)],
               edgecolors="black", linewidths=1.6, zorder=3)
    # Neutral legend entry: every centroid is drawn in its cluster's colour.
    ax.scatter([], [], marker="X", s=160, color="white", edgecolors="black",
               linewidths=1.6, label="final centroid (one per cluster)")
    ax.set(title=title, xlabel="feature 1", ylabel="feature 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def run_experiments(output_dir, seed=0, max_iterations=100):
    """Q2(b), 15 points: run all five k values and save cluster plots.

    Use the four-component truth labels only for a reference plot. K-means
    must cluster the features without access to those labels.

    Outputs written to output_dir:
      kmeans_k02.png ... kmeans_k10.png   one plot per k (assignments + centroids)
      kmeans_overview.png                 all five k values next to the reference
      reference_true_blobs.png            optional plot of the generating labels
      kmeans_summary.csv                  k, SSE, cluster sizes, empty clusters
    """
    plt = _import_pyplot()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The true labels go into a separate variable that K-means never receives.
    data, true_labels = make_toy_data()
    rng = np.random.default_rng(seed)  # one seeded generator, used in K_VALUES order

    results = {}
    for k in K_VALUES:
        initial_centroids = get_initial_clusters(k, data, rng)
        n_duplicate_starts = k - np.unique(initial_centroids, axis=0).shape[0]
        centroids, assignments = kmeans(data, initial_centroids, max_iterations)
        sizes = np.bincount(assignments, minlength=k)
        sse = within_cluster_sse(data, centroids, assignments)
        results[k] = {"centroids": centroids, "assignments": assignments,
                      "sse": sse, "sizes": sizes,
                      "duplicate_starts": n_duplicate_starts}

        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        _draw_clusters(ax, data, assignments, centroids,
                       f"K-means, k = {k}  (SSE = {sse:,.0f})")
        ax.legend(loc="upper right", fontsize=8)
        path = output_dir / f"kmeans_k{k:02d}.png"
        _save(fig, path)
        plt.close(fig)
        print(f"k = {k:2d}: SSE = {sse:10.2f}; cluster sizes = {sizes.tolist()}; "
              f"duplicate initial centroids = {n_duplicate_starts}; "
              f"empty clusters = {int(np.sum(sizes == 0))}; saved {path}")

    # Optional reference plot: the only place the true blob labels are used.
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for group in np.unique(true_labels):
        members = data[true_labels == group]
        ax.scatter(members[:, 0], members[:, 1], s=5, alpha=0.55, linewidths=0,
                   color=CLUSTER_COLORS[group], label=f"blob {group}",
                   rasterized=True)
    ax.set(title="Reference: true generating blobs (not used by K-means)",
           xlabel="feature 1", ylabel="feature 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, markerscale=3)
    reference_path = output_dir / "reference_true_blobs.png"
    _save(fig, reference_path)
    plt.close(fig)

    # Overview figure for the report: reference plus every k value.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
    axes = axes.ravel()
    for group in np.unique(true_labels):
        members = data[true_labels == group]
        axes[0].scatter(members[:, 0], members[:, 1], s=4, alpha=0.55,
                        linewidths=0, color=CLUSTER_COLORS[group],
                        rasterized=True)
    axes[0].set(title="Reference: 4 generating blobs (labels not used)",
                xlabel="feature 1", ylabel="feature 2")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(alpha=0.25)
    for ax, k in zip(axes[1:], K_VALUES):
        _draw_clusters(ax, data, results[k]["assignments"],
                       results[k]["centroids"],
                       f"k = {k}  (SSE = {results[k]['sse']:,.0f})")
    overview_path = output_dir / "kmeans_overview.png"
    _save(fig, overview_path)
    plt.close(fig)

    summary_path = output_dir / "kmeans_summary.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "within_cluster_sse", "cluster_sizes",
                         "duplicate_initial_centroids", "empty_clusters"])
        for k in K_VALUES:
            sizes = results[k]["sizes"]
            writer.writerow([k, f"{results[k]['sse']:.4f}",
                             " ".join(str(s) for s in sizes),
                             results[k]["duplicate_starts"],
                             int(np.sum(sizes == 0))])
    print(f"Saved reference plot to {reference_path}")
    print(f"Saved overview plot to {overview_path}")
    print(f"Saved summary table to {summary_path}")
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
