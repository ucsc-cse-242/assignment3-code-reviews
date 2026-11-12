"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path
import numpy as np

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.
    """
    # calculates squared euclidean distance from every point to every centroid.
    # shape points and centroids similarly so they can be paired
    points = data[:, None, :]
    centers = centroids[None, :, :]

    # gets difference between points and centroids
    differences = points - centers

    #squares the difference
    square = differences ** 2

    #adds all features together
    distances = square.sum(axis=2)

    #assigns each point to nearest centroid
    #ties go to the lowest indexed centroid because np.argmin returns the first minimum
    return np.argmin(distances, axis=1)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # use copy of old centroids to keep old one from being modified
    new_centroids = np.array(centroids, dtype=float, copy=True)

    # gets number of clusters and stores into k
    k = centroids.shape[0]

    # iterates through each cluster
    for j in range(k):
        # gets points assigned to cluster j
        members = data[assignments == j]

        #checks if cluster is nonempty
        if len(members) > 0:
            # nonempty cluster means we move centroid to the mean of its points
            new_centroids[j] = members.mean(axis=0)
        # empty cluster means we just keep the previous centroid so we don't update at all
        # previous centroid is already stored in new_centroids from the copy

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
    import matplotlib.pyplot as plt

    #makes output directory and sets it up
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # generates the provided toy dataset, ignoring true labels as its unnecessary for k means
    data, _ = make_toy_data()

    # seeded random generator to make results reproducible
    gen = np.random.default_rng(seed)

    # running k means for all values of k
    for k in K_VALUES:
        #gets k starting centroids from data
        initial_centroids = get_initial_clusters(k, data, gen)

        # runs k means and stores results
        centroids, assignments = kmeans(data, initial_centroids, max_iterations)

        # plots points (colored by assigned cluster)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(data[:, 0], data[:, 1], c=assignments, s=5, cmap="tab10", vmin=0, vmax=9)

        # plots final centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X",
                   s=200, edgecolors="white", label="final centroids")

        ax.set(title=f"K-means with k = {k}", xlabel="x1", ylabel="x2")
        ax.legend()

        # saves plot
        path = output_dir / f"kmeans_k{k}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"k = {k}: saved {path}")


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
