"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from pathlib import Path

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,), with values in 0..k-1
    (the index of the centroid).
    Resolve equal-distance ties consistently and document your rule.

    Distance is squared Euclidean distance (same nearest centroid as
    Euclidean distance, since the square root is monotonic).

    Tie rule: if a point is equally close to several centroids, it is
    assigned to the one with the lowest index. This is the behavior of
    np.argmin, which returns the first occurrence of the minimum.
    """

    # (n, 1, d) - (1, k, d) -> (n, k, d): every point vs. every centroid
    diffs = data[:, None, :] - centroids[None, :, :]

    # Square, then sum over the coordinate axis -> (n, k) squared distances
    sq_dists = (diffs ** 2).sum(axis=2)

    # Nearest centroid per point -> (n,)
    return np.argmin(sq_dists, axis=1)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.

    Empty-cluster policy: a cluster with no assigned points keeps its previous centroid.
    """

    temp_centroids = centroids.astype(float)

    # for each centroid
    for j in range(centroids.shape[0]):
        # points in j
        mask = assignments == j
        # if j has any points
        if mask.any():
            points_in_j = data[mask]  # shape (m_j, d)
            # update centroid j to the mean of its points
            temp_centroids[j] = points_in_j.mean(axis=0)  # shape (d,)
    return temp_centroids


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
    """
    Explain what changes as k increases and how the results relate to the four generating groups:
    
    k below 4 merges true blobs together, k of 4 matches the truth plot, and k above 4 splits blobs apart.
    """

    X, y = make_toy_data()
    rng = np.random.default_rng(seed)

    # Reference plot: true labels, for comparison only
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=5, alpha=0.6)
    ax.set_title("Ground-truth labels (not used by k-means)")
    fig.savefig(os.path.join(output_dir, "truth.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    for k in K_VALUES:
        init = get_initial_clusters(k, X, rng)
        centroids, assignments = kmeans(X, init, max_iterations)

        plot_clustering(X, centroids, assignments,
                        title=f"k-means, k={k}",
                        path=os.path.join(output_dir, f"kmeans_k{k}.png"))


def plot_clustering(X, centroids, assignments, title, path):
    k = centroids.shape[0]
    vmax = max(k - 1, 1)                  # same color scale for points and centroids
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], c=assignments, cmap="tab10",
               vmin=0, vmax=vmax, s=5, alpha=0.6)
    ax.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(k), cmap="tab10",
               vmin=0, vmax=vmax, marker="X", s=200, edgecolors="k")
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# def kmeans_history(data, centroids, max_iterations=100):
#     """
#     Record history of kmeans assignments and centroids
#     """
#     centroid_history = [centroids.astype(float)]
#     assignment_history = []
#     for _ in range(max_iterations):
#         assignments = update_assignments(data, centroids)
#         centroids = update_centroids(data, centroids, assignments)
#         assignment_history.append(assignments)
#         centroid_history.append(centroids)
#     assignment_history.append(update_assignments(data, centroids))
#     return centroid_history, assignment_history
#
#
# def plot_steps(data, centroid_history, assignment_history, steps):
#     """
#     Plot kmeans history
#     """
#     path = np.stack(centroid_history)              # (T, k, d)
#     k = path.shape[1]
#     cols = 3
#     rows = int(np.ceil(len(steps) / cols))
#     fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
#                              sharex=True, sharey=True)
#     axes = np.atleast_1d(axes).ravel()
#
#     for ax, t in zip(axes, steps):
#         # points colored by current assignment
#         ax.scatter(data[:, 0], data[:, 1], c=assignment_history[t],
#                    cmap="tab10", vmin=0, vmax=9, s=5, alpha=0.6)
#         # trail showing where each centroid has been
#         for j in range(k):
#             ax.plot(path[:t + 1, j, 0], path[:t + 1, j, 1], "k.-", lw=1)
#         # current centroids as big X markers, colored to match their cluster
#         ax.scatter(path[t, :, 0], path[t, :, 1], c=np.arange(k),
#                    cmap="tab10", vmin=0, vmax=9, marker="X", s=200,
#                    edgecolors="k")
#         ax.set_title(f"Iteration {t}")
#
#     for ax in axes[len(steps):]:
#         ax.axis("off")
#     plt.tight_layout()
#     plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "kmeans")
    args = parser.parse_args()
    run_experiments(args.output_dir, args.seed, args.max_iterations)


    # Visualizing history
    # X, _ = make_toy_data()  # make_blobs returns (X, y), so unpack it
    # rng = np.random.default_rng(0)
    # init = get_initial_clusters(4, X, rng)
    #
    # cent_hist, assign_hist = kmeans_history(X, init, max_iterations=15)
    # plot_steps(X, cent_hist, assign_hist, steps=[0, 1, 2, 3, 5, 15])


if __name__ == "__main__":
    main()
