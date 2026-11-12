"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path
import os
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
    # Compute distances and assign each point to a centroid.
    assignments = []
    k = len(centroids)  # Number of clusters.
    # Iterate through all instances, and for each, get the distance to all the centroids and pick the min.
    for i in range(data.shape[0]):
        assignment, min_distance = 0, np.inf
        for a in range(k):
            distance = np.linalg.norm(data[i] - centroids[a]) # Euclidean distance.
            # If multiple centroids are equidistant, we always take the assignment from the first centroid.
            if distance < min_distance:
                assignment = a
                min_distance = distance
        assignments.append(assignment)
    return assignments


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # Update the centroids by iterating through each cluster, and taking the mean. 
    k = len(centroids)  # Number of clusters.
    assignments_per_k = {a: 0 for a in range(k)}  # Keep track of how many instances each centroid has.
    centroid_sums = np.zeros((k, 2))  # Keep a running sum of all instances belonging to that centroid.
    for assignment, d in zip(assignments, data):
        centroid_sums[assignment] += d
        assignments_per_k[assignment] += 1
    # Return the new centroids by taking the mean; if a centroid is empty, retain previous centroid 
    return [centroid_sums[a] / assignments_per_k[a] if centroid_sums[a].sum() > 0 else centroids[a] for a in range(k)]


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


def plot_results(filename, data, centroids, assignments, gold_assignments=[]):
    """Plot the cluster assignments and centroids for the data, with optional reference assignment overlay.

    Creates a 2D plot of all the data, with cluster assignments as colors.
    Centroids are represented by an 'x'.
    """
    k = len(centroids)  # Number of clusters.
    centroids = np.array(centroids)
    fig, ax = plt.subplots(figsize=(7, 5))
    # Plot cluster assignments.
    ax.scatter(
        data[:, 0], data[:, 1],
        c=assignments,
        cmap='tab10'
    )
    # Plot centroids.
    ax.scatter(
        centroids[:, 0], 
        centroids[:, 1],
        c=range(k), cmap='tab10',
        marker='X', edgecolors='black',
        zorder=3
    )
    # plt.show()
    plt.savefig(filename)


def run_experiments(output_dir, seed=0, max_iterations=100):
    """Q2(b), 15 points: run all five k values and save cluster plots.

    Use the four-component truth labels only for a reference plot. K-means
    must cluster the features without access to those labels.
    """
    # Load toy data, create a seeded RNG, and initialize centroids.
    rng = np.random.default_rng(seed)
    data, gold_assignments = make_toy_data() # data is a (4000, 2) array of random ints; assignments are (4000) ints from 0-(k-1).
    # Run kmeans for every value in K_VALUES.
    for k in K_VALUES:
        centroids = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, centroids, max_iterations)
        os.makedirs(output_dir, exist_ok=True)
        # Plot each set of cluster assignments and its final centroids and save plots under output_dir.
        plot_results(os.path.join(output_dir, f"{k}.png"), data, centroids, assignments)
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
