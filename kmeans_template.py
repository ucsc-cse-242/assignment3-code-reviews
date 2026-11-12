"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path
import numpy
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
    res = []

    for i in range(0, len(data)):
        dist_arr = []
        for j in range(0, len(centroids)):
            dist = numpy.sum((data[i] - centroids[j]) ** 2)
            dist_arr.append(dist)

        closest = numpy.argmin(dist_arr)
        res.append(closest)

    return numpy.array(res)

def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # TODO: Update the centroid of each nonempty cluster.
    # TODO: Handle empty clusters using your documented policy.
    updated_arr = centroids.copy()

    for i in range(0, len(updated_arr)):
        points = []
        for j in range(0, len(data)):
            if(assignments[j] == i):
                points.append(data[j])
        if (len(points) > 0):
            updated_arr[i] = numpy.mean(points, axis=0)


    return updated_arr

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
    
    data, labels = make_toy_data()

    rng = numpy.random.default_rng(seed)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run K-means for each value of k
    for k in K_VALUES:
        # Choose initial centroids
        centroids = get_initial_clusters(k, data, rng)

        # Run K-means
        final_centroids, assignments = kmeans(
            data, centroids, max_iterations
        )

        # Create plot
        plt.figure(figsize=(7, 6))

        # Plot data points colored by assigned cluster
        plt.scatter(
            data[:, 0],
            data[:, 1],
            c=assignments,
            cmap="tab10",
            s=5
        )

        # Plot final centroids
        plt.scatter(
            final_centroids[:, 0],
            final_centroids[:, 1],
            c="black",
            marker="X",
            s=150,
            label="Centroids"
        )

        plt.title(f"K-means Clustering (k={k})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()

        # Save plot
        plt.savefig(output_dir / f"kmeans_k{k}.png")
        plt.close()

        print(f"Finished k={k}")



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
