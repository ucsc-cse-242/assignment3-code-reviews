"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
import typing as ty
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# I HATE PYTHON TYPING I HATE NUMPY I HATE IT I HATE IT
# WHERE ARE MY DEPENDENT TYPES PYTHON
type Point = np.ndarray  # ℝᵈ
type Points = np.ndarray  # ?_ ⨯ ℝᵈ
type Centroids = Points  # k ⨯ ℝᵈ
type Data = Points  # n ⨯ ℝᵈ
type CentroidIndex = int
type CentroidIndices = np.ndarray  # intᵏ

K_VALUES = (2, 3, 4, 6, 10)


def distance(p1: Point, p2: Point) -> float:
    return np.linalg.norm(p1 - p2)


nearest_centroid_idx: ty.Callable[[Centroids], ty.Callable[[Point], CentroidIndex]] = (
    lambda centroids: lambda p: int(np.argmin([distance(p, c) for c in centroids]))
)


def update_assignments(data: Data, centroids: Centroids) -> CentroidIndices:
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.
    """

    return np.apply_along_axis(nearest_centroid_idx(centroids), axis=1, arr=data)


def farthest_point(data: Data, centroids: Data) -> Point:
    """
    Called by update_centroids when a centroid doesn't own any points.
    """
    dists = [min(distance(p, c) for c in centroids) for p in data]
    return data[int(np.argmax(dists))]


def update_centroids(data: Data, centroids: Data, assignments) -> Centroids:
    """Return an updated (k, d) centroid array based on cluster assignments.

    For an empty cluster, place it at a point that's the farthest away from
    all other clusters. But only do this once per update. If we already did it,
    don't do it again and just leave the centroid alone.
    """
    centroid_owns = [data[assignments == j] for j in range(len(centroids))]
    rescued = False
    new_centroids = []
    for old, owned in zip(centroids, centroid_owns):
        if len(owned) > 0:
            new_centroids.append(owned.mean(axis=0))
            continue
        if rescued:
            # leave it alone
            new_centroids.append(np.array(old))
        else:
            new_centroids.append(farthest_point(data, centroids))
            rescued = True
    return np.array(new_centroids)


def kmeans(data, centroids, max_iterations=100):
    """Assignment 4's provided alternating-update loop, with a final assignment."""
    for _ in range(max_iterations):
        assignments = update_assignments(data, centroids)
        centroids = update_centroids(data, centroids, assignments)
    assignments = update_assignments(data, centroids)
    return centroids, assignments


def make_toy_data() -> tuple[Data, CentroidIndices, Centroids]:
    """Generate the exact make_blobs configuration from Assignment 4."""
    return make_blobs(
        n_samples=4000, centers=4, cluster_std=0.60, random_state=0, return_centers=True
    )


def get_initial_clusters(k, data: Points, rng: np.random.Generator):
    """Sample k rows with replacement, as in the source's randint initializer.

    This adaptation accepts a seeded NumPy generator for reproducibility.
    Repeated initial rows are possible; your empty-cluster policy must cope.
    """
    return data[rng.integers(0, data.shape[0], size=k)].copy()


def make_plot(title: str, data: Data, labels: CentroidIndices, centroids: Centroids):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(title)
    ax.scatter(data[:, 0], data[:, 1], c=labels, s=5, cmap="tab10")
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=200,
        c="red",
        edgecolors="black",
        label="centroids",
    )
    ax.legend()
    return fig


def run_experiments(output_dir, seed=0, max_iterations=100):
    """Q2(b), 15 points: run all five k values and save cluster plots.

    Use the four-component truth labels only for a reference plot. K-means
    must cluster the features without access to those labels.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, true_labels, true_centroids = make_toy_data()
    rng = np.random.default_rng(seed)

    fig = make_plot("k-means (k=4): truth", data, true_labels, true_centroids)
    fig.savefig(output_dir / "kmeans_k4_truth.png", dpi=150)
    plt.close(fig)

    for k in K_VALUES:
        init = get_initial_clusters(k, data, rng)
        centroids, labels = kmeans(data, init, max_iterations)

        fig = make_plot(f"k-means (k={k})", data, labels, centroids)
        fig.savefig(output_dir / f"kmeans_{k}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "kmeans",
    )
    args = parser.parse_args()
    run_experiments(args.output_dir, args.seed, args.max_iterations)


if __name__ == "__main__":
    main()
