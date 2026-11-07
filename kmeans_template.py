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
    # Tie rule: np.argmin returns the FIRST minimum, so a point equidistant from
    # several centroids goes to the one with the lowest index. Squared distances
    # are compared (same ordering as Euclidean, no square root needed).
    import numpy as np
    data = np.asarray(data, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    sq_dist = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(sq_dist, axis=1).astype(int)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # Empty-cluster policy: a cluster that received no points keeps its previous
    # centroid. This matters here because the initializer samples with
    # replacement, so two clusters can start on the same row; the higher-index
    # duplicate loses every tie and stays empty, frozen in place.
    import numpy as np
    data = np.asarray(data, dtype=float)
    new_centroids = np.array(centroids, dtype=float, copy=True)
    for j in range(new_centroids.shape[0]):
        members = data[assignments == j]
        if len(members):
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


def run_experiments(output_dir, seed=0, max_iterations=100):
    """Q2(b), 15 points: run all five k values and save cluster plots.

    Use the four-component truth labels only for a reference plot. K-means
    must cluster the features without access to those labels.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, truth = make_toy_data()
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data[:, 0], data[:, 1], c=truth, cmap="tab10", s=4, vmin=0, vmax=9)
    ax.set(title="Reference: the four make_blobs groups (true labels)",
           xlabel="x1", ylabel="x2")
    fig.tight_layout()
    fig.savefig(output_dir / "kmeans_truth.png", dpi=150)
    plt.close(fig)

    results = {}
    for k in K_VALUES:
        init = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, init, max_iterations)
        sizes = np.bincount(assignments, minlength=k)
        sse = float(((data - centroids[assignments]) ** 2).sum())
        results[k] = {"centroids": centroids, "assignments": assignments,
                      "sizes": sizes, "sse": sse,
                      "duplicate_init": len(np.unique(init, axis=0)) < k}
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(data[:, 0], data[:, 1], c=assignments, cmap="tab10", s=4,
                   vmin=0, vmax=9)
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X",
                   s=160, edgecolors="white", linewidths=1.5, label="final centroids")
        ax.set(title=f"K-means, k = {k} (SSE = {sse:.0f})", xlabel="x1", ylabel="x2")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / f"kmeans_k{k}.png", dpi=150)
        plt.close(fig)
        print(f"k={k:2d}  SSE={sse:9.1f}  cluster sizes={sizes.tolist()}"
              + ("  (duplicate initial centroid)" if results[k]["duplicate_init"] else ""))
    print(f"Saved plots to {output_dir}")
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
