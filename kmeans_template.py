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

    data = np.asarray(data, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    # Euclidean squared distance from every point to every centroid
    # differences has a shape of (n, k, d), whereas squared_distance has a shape of (n, k)
    differences = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    squared_distance = np.sum(differences ** 2, axis=2)

    # Since np.argmin returns the first minimum value, a point that is equally close to several centroids goes to the lowest cluster index.
    # This also covers the case where only the lowest-index copy receives points, and the other copies start out empty.
    return np.argmin(squared_distance, axis=1).astype(int)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # TODO: Update the centroid of each nonempty cluster.
    # TODO: Handle empty clusters using your documented policy.
    import numpy as np
    
    data = np.asarray(data, dtype=float)
    assignments = np.asarray(assignments)

    # Copy the centroid so you don't actually modify the inputted array
    copied_centroids = np.array(centroids, dtype=float, copy=True)

    for i in range(copied_centroids.shape[0]):
        members = data[assignments == i] 
        if members.shape[0] > 0:
            #Move centroid to mean of its points
            copied_centroids[i] = members.mean(axis=0)
    return copied_centroids

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
    import numpy as np
    import matplotlib.pyplot as plt
 
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # The true blob labels are discarded; K-means sees only the features.
    data, _ = make_toy_data()
    rng = np.random.default_rng(seed)
 
    for k in K_VALUES:
        initial = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, initial, max_iterations)
 
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(data[:, 0], data[:, 1], c=assignments, cmap="tab10",
                   s=6, alpha=0.6, vmin=0, vmax=9)
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X",
                   s=160, edgecolors="white", linewidths=1.2,
                   label="final centroids")
        ax.set(title=f"K-means clusters, k = {k}", xlabel="x1", ylabel="x2")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / f"kmeans_k{k}.png", dpi=150)
        plt.close(fig)

    # TODO: Save plots under output_dir and discuss the effect of k.
    # As k increases, K-means breaks the data into more groups. The data were made from 4 separate groups, and with k=4, K-means finds them basically perfectly. 
    # When k is smaller than 4, K-means has to combine groups. With k=2, it separated the top-left group and lumps together the other one, where the middle is empty. 
    # With k=3, it combines the two left groups. When k is larger than 4, K-means breaks the groups up further. With k=6, it splits a group into 3 parts. With k=10, it splits two groups. 
    # Which groups are split depends on where the random starting centers start, instead of any real pattern.


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

#AI usage: I used Claude for explaining concepts, optimizing code, bettering my commented explanations
