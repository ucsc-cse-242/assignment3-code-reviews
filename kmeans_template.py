"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """
    Assign each observation to its nearest centroid.

    Inputs:
        data: Array of shape (n, d) containing observations.
        centroids: Array of shape (k, d) containing current centroids.

    Output:
        Integer array of shape (n,) containing cluster assignments.

    If distances are equal, assign the observation to the centroid with the lowest index.
    """
    

    # Compute distances and assign each point to a centroid.
    distances = np.sum(
        (data[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2,
        axis=2
    )

    # Assign each point to its nearest centroid
    # np.argmin chooses the lowest index in case of a tie
    assignments = np.argmin(distances, axis=1)

    return assignments



def update_centroids(data, centroids, assignments):
    """
    Update centroids using the mean of assigned observations.

    Inputs:
        data: Array of shape (n, d) containing observations.
        centroids: Array of shape (k, d) containing current centroids.
        assignments: Array of shape (n,) containing cluster assignments.

    Output:
        Array of shape (k, d) containing updated centroids.

    If a cluster is empty, retain its previous centroid.
    The original centroids array is not modified.
    """


    # Copy the existing centroids
    new_centroids = centroids.copy()

    # Loop through each cluster
    for j in range(len(centroids)):

        # Select observations assigned to cluster j
        cluster_points = data[assignments == j]

        # Update the centroid if the cluster is not empty
        if len(cluster_points) > 0:
            new_centroids[j] = np.mean(cluster_points, axis=0)

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
    """
    Run K-means for different k values and save plots.

    Inputs:
        output_dir: Where to save plots.
        seed: Random seed (default 0).
        max_iterations: Maximum iterations (default 100).

    Output:
        Saves plots for k = 2, 3, 4, 6, 10.

    Steps:
        1. Load toy data, create a seeded RNG, and initialize centroids.
        2. Run kmeans for every value in K_VALUES.
        3. Plot each set of cluster assignments and its final centroids.
        4. Save plots under output_dir and discuss the effect of k.

    """

    # Create output directory if it does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the dataset
    data, true_labels = make_toy_data()

    # Create a seeded random generator
    rng = np.random.default_rng(seed)

    # Run K-means for each k value
    for k in K_VALUES:

        # Initialize centroids using the provided function
        initial_centroids = get_initial_clusters(k, data, rng)

        # Run our K-means implementation
        final_centroids, assignments = kmeans(
            data, initial_centroids, max_iterations
        )

        # Plot the cluster assignments
        plt.figure(figsize=(8, 6))

        plt.scatter(
            data[:, 0],
            data[:, 1],
            c=assignments,
            cmap="tab10",
            s=5,
            alpha=0.7
        )

        # Plot the final centroids
        plt.scatter(
            final_centroids[:, 0],
            final_centroids[:, 1],
            c="black",
            marker="X",
            s=150,
            label="Final centroids"
        )

        plt.title(f"K-means Clustering (k = {k})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.tight_layout()

        # Save each plot
        plt.savefig(output_dir / f"kmeans_k{k}.png", dpi=300)
        plt.close()

        print(f"Saved plot for k = {k}")


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
