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
    # TODO: Compute distances and assign each point to a centroid.
    import numpy as np
 
    distances = np.zeros((data.shape[0], centroids.shape[0]))
    for j in range(centroids.shape[0]):
        distances[:, j] = np.sum((data - centroids[j]) ** 2, axis=1)
    return np.argmin(distances, axis=1)

def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # TODO: Update the centroid of each nonempty cluster.
    # TODO: Handle empty clusters using your documented policy.
    import numpy as np
 
    new_centroids = np.array(centroids, dtype=float)
    for j in range(new_centroids.shape[0]):
        points = data[assignments == j]
        if len(points) > 0:
            new_centroids[j] = points.mean(axis=0)
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
    # TODO: Load toy data, create a seeded RNG, and initialize centroids.
    # TODO: Run kmeans for every value in K_VALUES.
    # TODO: Plot each set of cluster assignments and its final centroids.
    # TODO: Save plots under output_dir and discuss the effect of k.
    import numpy as np
    import matplotlib.pyplot as plt
 
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, true_labels = make_toy_data()
    rng = np.random.default_rng(seed)
 
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap="tab10",
                vmin=0, vmax=9, s=4)
    plt.title("Reference: true blob labels")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig(output_dir / "reference.png", dpi=150)
    plt.close()
 
    for k in K_VALUES:
        initial_centroids = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, initial_centroids, max_iterations)
        wcss = np.sum((data - centroids[assignments]) ** 2)
        sizes = np.bincount(assignments, minlength=k)
        print(f"k = {k}: WCSS = {wcss:.2f}, cluster sizes = {sizes.tolist()}")
 
        plt.figure(figsize=(6, 5))
        plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap="tab10",
                    vmin=0, vmax=9, s=4)
        plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X",
                    s=150, label="final centroids")
        plt.title(f"K-means with k = {k}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.savefig(output_dir / f"kmeans_k{k}.png", dpi=150)
        plt.close()
 
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
