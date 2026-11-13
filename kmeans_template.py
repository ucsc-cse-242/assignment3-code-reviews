"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""
import numpy as np
import matplotlib.pyplot as plt
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
    # Compute differences between every point and every centroid.
    differences = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]

    # Compute squared Euclidean distances.
    squared_distances = np.sum(differences ** 2, axis=2)

    # Choose the closest centroid for each point.
    # np.argmin chooses the lowest index in a distance tie.
    assignments = np.argmin(squared_distances, axis=1)

    return assignments


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # TODO: Update the centroid of each nonempty cluster.
    # TODO: Handle empty clusters using your documented policy.
    # Copy the centroids to avoid modifying the originals.
    new_centroids = np.array(centroids, dtype=float, copy=True)

    # Update each cluster separately.
    for cluster_index in range(len(centroids)):

        # Select points assigned to the current cluster.
        members = data[assignments == cluster_index]

        if len(members) > 0:
            # Move the centroid to the mean of its points.
            new_centroids[cluster_index] = np.mean(members, axis=0)

        # If the cluster is empty, keep its previous centroid.

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
    # Generate the data. The true labels are not used in clustering.
    data, _true_labels = make_toy_data()

    # Reproducible random number generator.
    rng = np.random.default_rng(seed)

    # Create the output folder if it does not exist.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for k in K_VALUES:

        # Choose k initial centroids from the dataset.
        initial_centroids = get_initial_clusters(k, data, rng)

        # Run the supplied K-means algorithm.
        centroids, assignments = kmeans(
            data, initial_centroids, max_iterations
        )

        # Plot the points, colored by their assigned clusters.
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(
            data[:, 0], data[:, 1],
            c=assignments,
            cmap="tab10",
            vmin=-0.5,
            vmax=k - 0.5,
            s=9,
            alpha=0.65
        )

        # Draw final centroids as large black X marks.
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            marker="X",
            s=190,
            c="black",
            edgecolors="white",
            linewidths=1.0,
            label="Final centroids"
        )

        # Label the centroids.
        for index, (x1, x2) in enumerate(centroids):
            ax.annotate(
                str(index),
                (x1, x2),
                xytext=(8, 8),
                textcoords="offset points",
                fontweight="bold"
            )

        ax.set_title(f"K-means clustering (k = {k})")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()
        ax.grid(alpha=0.2)

        fig.tight_layout()

        # Save a separate image for each k.
        output_path = output_dir / f"kmeans_k{k}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        # Print useful experiment statistics.
        cluster_sizes = np.bincount(assignments, minlength=k)
        inertia = np.sum(
            (data - centroids[assignments]) ** 2
        )

        print(
            f"k={k:2d} | cluster sizes={cluster_sizes.tolist()} "
            f"| within-cluster SSE={inertia:.2f} "
            f"| saved {output_path}"
        )


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
