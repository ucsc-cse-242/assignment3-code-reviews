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
    # Compute distances and assign each point to a centroid. 
    # raise NotImplementedError("Implement the assignment step from scratch.") 
    # check data and centroids are not empty and have compatible shapes
    if data.size == 0 or centroids.size == 0:
        raise ValueError("Data and centroids must not be empty.")
    if data.shape[1] != centroids.shape[1]:
        raise ValueError("Data and centroids must have the same number of features.")
    # Compute the squared Euclidean distances from each point to each centroid.
    # The result will be a (n, k) array.
    distances = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    # Assign each point to the index of the closest centroid. Simply choose the first centroid in case of ties.
    assignments = np.argmin(distances, axis=1) # np.argmin chooses the first occurrence of the minimum value. 
    return assignments


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # Use new_centroids to store the updated centroids.
    new_centroids = np.zeros_like(centroids)
    for i in range(centroids.shape[0]):
        cluster_points = data[assignments == i]
        # Update the centroid of each nonempty cluster.
        if len(cluster_points) > 0:
            new_centroids[i] = cluster_points.mean(axis=0)
        else:
            # Handle empty clusters by retaining the previous centroid.
            new_centroids[i] = centroids[i]
    return new_centroids
    # raise NotImplementedError("Implement the centroid step from scratch.")


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


def plot_clusters(data, assignments, centroids, output_path):
    """Plot each set of cluster assignments and its final centroids.

    Use a consistent color scheme for each cluster index. Save the plot to
    output_path. You may use matplotlib or seaborn.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap='viridis', s=10, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=50, label='Centroids')
    # add coordinate labels for centroids
    for i, (x, y) in enumerate(centroids):
        plt.text(x, y+0.2, f'[{i}]', fontsize=10, ha='center', va='center', color='red')
        plt.text(x, y-0.2, f'({x:.2f}, {y:.2f})', fontsize=10, ha='center', va='center', color='black', weight='bold')
    plt.title(f'K-means Clustering (k={len(centroids)})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_reference(data, labels, output_path):
    """Plot the four-component truth labels for reference.

    Use a consistent color scheme for each label. Save the plot to output_path.
    You may use matplotlib or seaborn.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10, alpha=0.5)
    plt.title('Reference Plot (Ground Truth Labels)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_experiments(output_dir, seed=0, max_iterations=100):
    """Q2(b), 15 points: run all five k values and save cluster plots.

    Use the four-component truth labels only for a reference plot. K-means
    must cluster the features without access to those labels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load toy data, create a seeded RNG, and initialize centroids. 
    data = make_toy_data()
    rng = np.random.default_rng(seed)
    plot_reference(data[0], data[1], output_dir / "kmeans_reference.png")
    # Run kmeans for every value in K_VALUES. 
    for k in K_VALUES:
        print(f"Running K-means for k={k}...")
        centroids = get_initial_clusters(k, data[0], rng) # initialize centroids for current k
        final_centroids, assignments = kmeans(data[0], centroids, max_iterations) 
        print(f"Final centroids for k={k}:\n{final_centroids}\n")
        # Save plots under output_dir and discuss the effect of k.
        plot_clusters(data[0], assignments, final_centroids, output_dir / f"kmeans_k_{k}.png")
    return 0

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
