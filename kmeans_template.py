"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.
    """
    # TODO: Compute distances and assign each point to a centroid.
    assignments = []
    for point in data:
        distances = np.linalg.norm(centroids - point, axis=1)
        assignments.append(np.argmin(distances))

    assignments = np.array(assignments)
    return assignments
     

def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # TODO: Update the centroid of each nonempty cluster.
    # mean of cluster
    new_centroids = centroids.copy()

    for k in range(len(centroids)):
        cluster_points = data[assignments == k]
        # TODO: Handle empty clusters by keeping the previous centroid
        if len(cluster_points) > 0:
            new_centroids[k] = np.mean(cluster_points, axis=0)

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

# Plotting helper
"""
plot_clusters
input: data, centroids, assignments, output directory for the plot
output: void
"""
def plot_clusters(X, centroids, assignments, output_dir):
    plt.figure(figsize=(7, 5))

    # plot points colored by respective cluster assignment
    plt.scatter(
        X[:, 0], X[:, 1],
        c=assignments,
        cmap="tab10",
        s=80,
        label="Data points"
    )

    # plot centroids (X)
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c = "Red",
        marker="X",
        s = 250,
        edgecolors="black",
        label="Centroids"
    )

    # display graph
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("K-Means Clustering")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir)
    plt.close()

def run_experiments(output_dir, seed=0, max_iterations=100):
    """Q2(b), 15 points: run all five k values and save cluster plots.

    Use the four-component truth labels only for a reference plot. K-means
    must cluster the features without access to those labels.
    """
    # TODO: Load toy data, create a seeded RNG, and initialize centroids.
    X, y = make_toy_data()
    rng = np.random.default_rng(seed=seed)
    
    # TODO: Run kmeans for every value in K_VALUES.
    for k in K_VALUES:
        clusters = get_initial_clusters(k, X, rng)
        centroids, assignments = kmeans(X, clusters, max_iterations)
        # TODO: Plot each set of cluster assignments and its final centroids.
        # TODO: Save plots under output_dir and discuss the effect of k.
        output_dir.mkdir(parents=True, exist_ok=True)
        # HELPER defined above
        plot_clusters(X, centroids, assignments, output_dir / f"kmeans_k{k}.png")
    return


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