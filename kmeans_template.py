"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path
import numpy as np

K_VALUES = (2, 3, 4, 6, 10)

def distance(point, centroid):
    """Plain Euclidean distance between two 1-D arrays"""
    import math

    total = 0.0

    # go through each coordinate
    for i in range(len(point)):
        # difference along this axis
        diff = point[i] - centroid[i]
        # square it and accumulate
        total += diff * diff
        
    # square root of the sum of squares
    return math.sqrt(total)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.
    """
    
    # number of data points
    n = data.shape[0]
    # number of clusters
    k = centroids.shape[0]
    # one cluster id per point
    assignments = np.zeros(n, dtype=int)

    for i in range(n):
        # assuming centroid 0 is the closest to begin
        best_index = 0
        best_distance = distance(data[i], centroids[0])

        # check the remaining centroids and remember the nearest one
        for j in range(1, k):
            d = distance(data[i], centroids[j])
            # TIE RULE: strict '<' updates best_distance if the distance is less than best_distance
            if d < best_distance:
                best_distance = d
                best_index = j
            
        # store index of best centroid
        assignments[i] = best_index

    return assignments


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    k, d = centroids.shape
    # create a copy so the argument array remains unchanged
    new_centroids = centroids.copy()

    for j in range(k):
        # step 1: collect every data row currently assigned to cluster j
        members = []
        for i in range(data.shape[0]):
            if assignments[i] == j:
                members.append(data[i])

        # step 2: EMPTY CLUSTER : keep the old centroid and move on
        if len(members) == 0:
            continue

        # step 3: new centroid = mean of its members, one coordinate at a time
        for col in range(d):
            total = 0.0
            for point in members:
                total += point[col]
            new_centroids[j, col] = total / len(members)

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
    # lazy loading modules
    import matplotlib
    # use the headless backend to save plots to file only
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    # create results/kmeans/ dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data, true_labels = make_toy_data()
    rng = np.random.default_rng(seed)

    # reference plot using the true generating labels
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=true_labels, s=6, cmap="tab10")
    plt.title("Reference: true blob labels (4 groups)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig(output_dir / "reference_true_labels.png", dpi=150)
    plt.close()

    for k in K_VALUES:
        # new random initial centroids for this k
        initial = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, initial, max_iterations)

        # points coloured by cluster, final centroids as black X markers
        plt.figure(figsize=(6, 5))
        plt.scatter(data[:, 0], data[:, 1], c=assignments, s=6, cmap="tab10")
        plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X",
                    s=120, edgecolors="white", label="final centroids")
        plt.title(f"K-means with k = {k}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend(loc="best")
        plt.savefig(output_dir / f"kmeans_k{k}.png", dpi=150)
        plt.close()
        print(f"k={k}: saved kmeans_k{k}.png")

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