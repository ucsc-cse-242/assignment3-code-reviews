"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path
import math
import numpy as np
import statistics
import random
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

    if len(centroids) == 0:
        raise ValueError("No Centroids")

    for point in data:
        nearestCentroid = 0
        nearestDist = math.dist(point, centroids[0])
        for i in range(len(centroids)):
            dist = math.dist(point, centroids[i])
            # This conditional intrinsically breaks ties. 
            # It will prefer centroids earlier in the input array
            if dist < nearestDist:
                nearestDist = dist
                nearestCentroid = i
        assignments.append(nearestCentroid)

    assignments = np.array(assignments)
    return assignments


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    newCents = np.array(centroids, copy=True)

    clusters = {}
    for i in range(len(data)):
        if assignments[i] not in clusters:
            clusters[assignments[i]] = [data[i]]
        else:
            clusters[assignments[i]].append(data[i])
        
    # TODO: Update the centroid of each nonempty cluster.
    for cluster in clusters:
        newCents[cluster] = np.mean(clusters[cluster], axis = 0)
        
    # TODO: Handle empty clusters using your documented policy.
    # This function handles empty clusters by assigning them to a random point in the data.
    for i in range(len(newCents)):
        if i not in clusters:
            newCents[i] = data[random.randrange(len(data))]

    return newCents


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
    toyData, truth = make_toy_data()
    rng = np.random.default_rng(seed)
    centroids = {}
    for k in K_VALUES:
        centroids[k] = get_initial_clusters(k, toyData, rng)

    # TODO: Run kmeans for every value in K_VALUES.
    finalCentroids = {}
    finalAssignments = {}
    for k in K_VALUES:
        finalCentroids[k], finalAssignments[k] = kmeans(toyData, centroids[k], max_iterations=max_iterations)
    
    # TODO: Plot each set of cluster assignments and its final centroids.
    for k in K_VALUES:
        fig, ax = plt.subplots()

        points = ax.scatter(
            toyData[:, 0],
            toyData[:, 1],
            c=finalAssignments[k],
            cmap="tab10",
            vmin=-0.5,
            vmax=k - 0.5,
            s=10,
            alpha=0.6,
        )

        ax.scatter(
            finalCentroids[k][:, 0],
            finalCentroids[k][:, 1],
            color="black",
            marker="X",
            s=180,
            edgecolors="white",
            label="Final centroids",
        )

        fig.colorbar(points, ax=ax, ticks=range(k), label="Cluster")
        ax.set_title(f"K-means with k = {k}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()

        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"kmeans_k{k}.png", bbox_inches="tight")
        plt.close(fig)
    # TODO: Save plots under output_dir and discuss the effect of k.
    # k seems to be very important, because in these specific cases, there is a clear distinction that k = 4 is the most sensible choice(at least visually).
    # When you have less the k = 4, you get centroids in between clusters and it is clearly trying to just make it work.
    # When you have more than k = 4, it forces clusters to have multiple centroids forced into them.
    # This is very clear when you look at k10, as the bottom right most cluster has 5 centroids in it, while the middle two only have one centroid each.
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
