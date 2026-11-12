"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

SVM_C = 1.0
SVM_GAMMA = 0.01
POSITIVE_FLIP_PROBABILITY = 0.35
NEGATIVE_FLIP_PROBABILITY = 0.20


def generate_clean_split(rng, samples_per_class=1000):
    """Return features and labels for one independent clean split.

    Positive class: mean (1, 1), covariance I.
    Negative class: mean (-1, -1), covariance 3I.
    The negative covariance entries are variances, not standard deviations.
    """
    import numpy as np
    positive = rng.multivariate_normal((1, 1), np.eye(2), samples_per_class)
    negative = rng.multivariate_normal((-1, -1), 3 * np.eye(2), samples_per_class)
    features = np.vstack((positive, negative))
    labels = np.concatenate((np.ones(samples_per_class, dtype=int),
                             -np.ones(samples_per_class, dtype=int)))
    return features, labels


def corrupt_training_labels(clean_labels, rng):
    """
    Randomly flip training labels based on their original class.

    Inputs:
        clean_labels: Array of original labels (+1 or -1).
        rng: Seeded random number generator.

    Output:
        Array containing the noisy training labels.

    Flip +1 with probability 0.35 and -1 with probability 0.20. Do not modify the original labels.
    """

    # Copy the original labels
    noisy_labels = clean_labels.copy()

    # Assign flip probabilities based on original labels
    flip_probs = np.where(
        clean_labels == 1,
        POSITIVE_FLIP_PROBABILITY,
        NEGATIVE_FLIP_PROBABILITY
    )

    # Randomly determine which labels to flip
    flips = rng.random(len(clean_labels)) < flip_probs

    # Flip the selected labels
    noisy_labels[flips] *= -1

    return noisy_labels


def repair_labels_knn(train_features, noisy_labels, neighbors):
    """
    Repair noisy labels using K-NN majority voting.

    Inputs:
        train_features: Training observations.
        noisy_labels: Noisy training labels.
        neighbors: Number of nearest neighbors.

    Output:
        Array of repaired training labels.

    Steps:
        1. Find the nearest neighbors for each observation using Euclidean distance.
        2. Exclude the observation itself.
        3. Find the majority label of its five neighbors.
        4. Return the repaired labels.

    If distances tie, choose the observation with the lowest index.
    """
    
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(metric="euclidean")
    knn.fit(train_features)

    # Find all neighbors
    distances, indices = knn.kneighbors(
        train_features,
        n_neighbors=len(train_features)
    )

    # Exclude itself and break distance ties by lowest index
    neighbor_indices = []

    for i, row in enumerate(indices):
        valid = row != i

        # Sort by distance, then index
        order = np.lexsort((row[valid], distances[i][valid]))

        # Select the five nearest other observations
        neighbor_indices.append(row[valid][order[:neighbors]])

    neighbor_indices = np.array(neighbor_indices)

    # Get the original noisy labels of the neighbors
    neighbor_labels = noisy_labels[neighbor_indices]

    # Replace labels using majority voting
    repaired_labels = np.where(
        np.sum(neighbor_labels, axis=1) > 0, 1, -1
    )

    return repaired_labels

def train_and_evaluate(train_features, train_labels, test_features, test_labels):
    """Train the required RBF SVM and return accuracy on the clean test set."""

    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    model = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)
    model.fit(train_features, train_labels)
    return float(accuracy_score(test_labels, model.predict(test_features)))


def run_comparison(repeats, seed, neighbors, output_path):
    """
    Compare SVM accuracy before and after K-NN label repair.

    Inputs:
        repeats: Number of trials.
        seed: Random seed.
        neighbors: Number of nearest neighbors.
        output_path: Where to save results.

    Output:
        Saves trial accuracies and prints mean accuracies.

    Steps:
        1. Generate independent clean training and test splits.
        2. For each repeat, corrupt training labels and evaluate the baseline.
        3. Train an SVM using noisy labels.
        4. Repair the same noisy labels with K-NN.
        5. Train a fresh SVM on repaired labels; evaluate on clean test data.
        6. Save and compare the accuracies.
    """

    # Create seeded random generator
    rng = np.random.default_rng(seed)

    # Generate independent clean training and test sets
    train_features, train_labels = generate_clean_split(rng)
    test_features, test_labels = generate_clean_split(rng)

    # Store accuracy results
    results = []

    for trial in range(repeats):

        # Corrupt training labels
        noisy_labels = corrupt_training_labels(train_labels, rng)

        # Train and evaluate SVM using noisy labels
        baseline_accuracy = train_and_evaluate(
            train_features, noisy_labels,
            test_features, test_labels
        )

        # Repair labels using K-NN
        repaired_labels = repair_labels_knn(
            train_features, noisy_labels, neighbors
        )

        # Train and evaluate SVM using repaired labels
        knn_accuracy = train_and_evaluate(
            train_features, repaired_labels,
            test_features, test_labels
        )

        # Store results for this trial
        results.append({
            "trial": trial + 1,
            "baseline_accuracy": baseline_accuracy,
            "knn_accuracy": knn_accuracy
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results to CSV
    results_df.to_csv(output_path, index=False)

    # Calculate mean accuracies
    baseline_mean = results_df["baseline_accuracy"].mean()
    knn_mean = results_df["knn_accuracy"].mean()

    # Calculate difference
    difference = knn_mean - baseline_mean

    # Print results
    print(f"Mean baseline accuracy: {baseline_mean:.4f}")
    print(f"Mean K-NN accuracy: {knn_mean:.4f}")
    print(f"Difference: {difference:.4f}")
    


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=5,
                        choices=(5,), help="Use five neighbors for this assignment.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "noise_comparison.csv")
    args = parser.parse_args()
    run_comparison(args.repeats, args.seed, args.neighbors, args.output)


if __name__ == "__main__":
    main()
