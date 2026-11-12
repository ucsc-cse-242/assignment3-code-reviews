"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path

import os
import numpy as np

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """
    Given our data (n,d) and our centroids (k,d) we update the assignemnt of points to centroids using euclidian distance

    Returns assignments (n,) where each data point x_n is assigned to a cluster 
    """
    k = centroids.shape[0]
    n = data.shape[0]

    # expand dimensions so that there are now shape missmatches
    data = np.repeat(np.expand_dims(data, 1), k, axis=1) # n, k, d
    centroids = np.repeat(np.expand_dims(centroids, 0), n, axis=0) # n, k, d

    # calculate the euclidian distance of each point to each cluster centre
    diff = data - centroids
    euclidian_distance = np.sqrt(((diff)**2).sum(axis=2)) # n, k | the distance of every point to all clusters

    # get the cluster assignments
    assignments = np.argmin(euclidian_distance, axis=1)
    return assignments

def update_centroids(data, centroids, assignments):
    """
    Returns an updated (k, d) centroid array based on cluster assignments.
    On an empty cluster we retain its previous centroid.
    """
    # Update the centroid of each nonempty cluster.
    k = centroids.shape[0]

    new_centroids = []
    for centroid_index in range(k):
        assigned = data[assignments==centroid_index]
        
        # handle empty cluster: retain previous value
        if assigned.size == 0:
            updated_centroid = centroids[centroid_index]
        # update non-empty cluster: take its assigned values mean
        else:
            updated_centroid = assigned.mean(0)
        new_centroids.append(updated_centroid)

    centroids = np.stack(new_centroids)
    return centroids


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
    Runs all five k values and saves cluster plots on our toy dataset. 
    The data is generated once and k-means is run on the same data, centroids
    are initialized as random samples from the data
    """
    # Load toy data, create a seeded RNG, and initialize centroids.
    X, Y = make_toy_data()
    rng = np.random.default_rng(seed=seed)
    
    # Run kmeans for every value in K_VALUES.
    for k in K_VALUES:
        centroids = get_initial_clusters(k, X, rng=rng)
        centroids, assignments = kmeans(X, centroids, max_iterations=max_iterations)
    
        # Plot each set of cluster assignments and its final centroids.
        import matplotlib.pyplot as plt

        plt.scatter(X[:, 0], X[:, 1], c=assignments)
        plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='x', s=200)
        # Save plots under output_dir and discuss the effect of k.
        plt.savefig(output_dir / f'k:{k}.png')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "kmeans")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    run_experiments(args.output_dir, args.seed, args.max_iterations)


if __name__ == "__main__":
    main()
