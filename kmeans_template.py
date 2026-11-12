"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.

    rule: lower (first encountered) centroid label kept on tie
    """
    n = data.shape[0]
    c_n = centroids.shape[0]
    assignments = np.zeros(n, dtype=int)

    for i in range(n):
        nearest = (0, np.inf) # (centroid index, distance)
        # find nearest centroid to point i
        for j in range(c_n):
            d = np.linalg.norm(data[i] - centroids[j])
            if (d < nearest[1]): # keep old label on tie
                nearest = (j, d)

        # update assignment to nearest centroid
        assignments[i] = nearest[0]

    return assignments

def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.

    Empty clusters: centroid is not moved
    """
    # Update the centroid of each nonempty cluster.
    n = data.shape[0]
    c_n = centroids.shape[0]

    # sort data into groups based on assignments
    clusters = [ [] for i in range(c_n) ]
    for i in range(n):
        clusters[ assignments[i] ].append( data[i] )

    new_centroids = np.zeros((c_n,2))
    for i in range(c_n):
        if ( len(clusters[i]) > 0 ):
            # change this centroid to the mean of the points associated with this centroid
            new_centroids[i] = np.mean(clusters[i], axis=0)
        else: # if cluster empty, use current centroid position
            new_centroids[i] = centroids[i]

    return new_centroids


def kmeans(data, centroids, max_iterations=100):
    """Assignment 4's provided alternating-update loop, with a final assignment."""
    for _ in range(max_iterations):
        print("interation: " + str(_))
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
    # Load toy data, create a seeded RNG, and initialize centroids.
    # Run kmeans for every value in K_VALUES.
    # Plot each set of cluster assignments and its final centroids.
    # Save plots under output_dir and discuss the effect of k.

    for k in K_VALUES:
        # generate data
        data, assignments_true = make_toy_data()
        
        rng = np.random.default_rng(seed)
        # initial centroids
        centroids = get_initial_clusters(k, data, rng)

        # initial assignments
        assignments = update_assignments(data, centroids)
        # kmeans
        centroids, assignments = kmeans(data, centroids, max_iterations=max_iterations)


        plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap=plt.colormaps['Paired'].resampled(k))
        plt.scatter(centroids[:, 0], centroids[:, 1], color="blue")
        plt.title("Clusters at k=" + str(k))

        plt_path = str(output_dir) + "\\scatter_k_" + str(k) + ".png"
        plt.savefig(plt_path)
        plt.clf()



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
