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
    differences = data[:, None, :] - centroids[None, :, :] # get the differences between each point and each centroid
    squared_distances = np.sum(differences ** 2, axis=2) # get the squared distances between each point and each centroid
    return np.argmin(squared_distances, axis=1)  # assigning each point to the nearest centroid


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """

    import numpy as np
    new_centroids = np.array(centroids, dtype=float, copy=True) # create a copy of the centroids to avoid modifying the input array in place
    for i in range(new_centroids.shape[0]): # loop through each centroid
        cluster_points = assignments == i # get the points assigned to the current centroid
        if np.any(cluster_points): # check if there are any points assigned to the current centroid
            new_centroids[i] = data[cluster_points].mean(axis=0) # update the centroid to the mean of the assigned points
        #else: # if there are no points assigned to the current centroid, retain its previous value (do nothing bc already copied)
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    data, true_labels = make_toy_data() # load the toy data and the true labels for the reference plot
    rng = np.random.default_rng(seed) # create a seeded random number generator for reproducibility
 
    # get the reference plot of the true labels for comparison with the k-means results
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data[:, 0], data[:, 1], c=true_labels, cmap="tab10", s=4)
    ax.set(title="Reference: 4 true generating blobs", xlabel="x1", ylabel="x2")
    fig.tight_layout()
    fig.savefig(output_dir / "reference_true_blobs.png", dpi=150)
    plt.close(fig)
 
    summary = [] # store the summary of the experiments for each k value
    for k in K_VALUES:
        initial = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, initial, max_iterations) # run k-means for the current k value
        sizes = np.bincount(assignments, minlength=k) # get the sizes of each cluster
        sse = float(np.sum((data - centroids[assignments]) ** 2)) # compute the sum of squared errors for the current clustering 
        summary.append((k, sse, sizes)) # store the summary of the current k value in the summary list
 
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(data[:, 0], data[:, 1], c=assignments, cmap="tab10",
                   s=4, vmin=0, vmax=9)
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X",
                   s=160, edgecolors="white", linewidths=1.5,
                   label="final centroids")
        ax.set(title=f"K-means, k = {k} (SSE = {sse:.0f})",
               xlabel="x1", ylabel="x2")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / f"kmeans_k{k}.png", dpi=150)
        plt.close(fig)
 
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"seed={seed}, max_iterations={max_iterations}\n")
        for k, sse, sizes in summary:
            line = f"k={k:2d}  SSE={sse:9.1f}  cluster sizes={sizes.tolist()}"
            print(line)
            f.write(line + "\n")
    print(f"Saved plots to {output_dir}")
    return summary

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
