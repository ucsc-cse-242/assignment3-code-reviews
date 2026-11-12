"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

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
    """Independently flip +1 with probability .35 and -1 with probability .20.

    Select flip probabilities from the ORIGINAL labels. Do not allow a label
    flipped in one direction to be flipped again by the other rule.
    Return a new array, leaving clean_labels unchanged.
    """
    # Draw flips and return the noisy labels.
    import numpy as np
    noisy_labels = clean_labels.copy()
    positive_mask = clean_labels == 1
    negative_mask = clean_labels == -1
    noisy_labels[positive_mask] = np.where(rng.random(positive_mask.sum()) < POSITIVE_FLIP_PROBABILITY, -1, 1)
    noisy_labels[negative_mask] = np.where(rng.random(negative_mask.sum()) < NEGATIVE_FLIP_PROBABILITY, 1, -1)
    return noisy_labels


def repair_labels_knn(train_features, noisy_labels, neighbors):
    """Q3(b): propose a K-NN label-repair method; return repaired labels.

    For this simplified assignment, use 5 neighbors, Euclidean distance,
    and replace each label with the majority label of those neighbors.
    Five binary votes cannot tie; break equal-distance neighbor ties consistently.
    Exclude each point itself from its neighborhood. Base all repairs on
    the original noisy labels, so row ordering does not affect the outcome.
    sklearn.neighbors may be used; from-scratch K-NN is not a source requirement.
    """
    # Find neighbors using training features only.
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    nbrs = NearestNeighbors(n_neighbors=neighbors + 1, algorithm='auto').fit(train_features)
    distances, indices = nbrs.kneighbors(train_features)
    # Exclude the first neighbor (itself) and get the majority vote for each point
    repaired_labels = noisy_labels.copy()
    for i, neighbors_indices in enumerate(indices):
        neighbor_labels = noisy_labels[neighbors_indices[1:]]  # Exclude itself
        majority_label = 1 if np.sum(neighbor_labels) > 0 else -1
        repaired_labels[i] = majority_label
    return repaired_labels


def train_and_evaluate(train_features, train_labels, test_features, test_labels):
    """Train the required RBF SVM and return accuracy on the clean test set."""
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    model = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)
    model.fit(train_features, train_labels)
    return float(accuracy_score(test_labels, model.predict(test_features)))


def run_comparison(repeats, seed, neighbors, output_path):
    """Q3(a-b), 20 points each: compare the baseline with K-NN label repair.

    Suggested adaptation: generate independent clean training/test sets once,
    then repeat corruption and training (default: 20 noise trials). In each
    trial, use identical data and corrupted labels for both approaches.
    Do not give the repair method the clean training or test labels.
    """
    # Generate independent clean training and test splits.
    import numpy as np
    rng = np.random.default_rng(seed)
    train_features, train_labels = generate_clean_split(rng)
    test_features, test_labels = generate_clean_split(rng)
    # For each repeat, corrupt training labels and evaluate the baseline.
    baseline_accuracy = []
    noisy_labels = []
    knn_repair_accuracy = []
    for _ in range(repeats):
        noisy_labels = corrupt_training_labels(train_labels, rng)
        baseline_accuracy = train_and_evaluate(train_features, noisy_labels,
                                               test_features, test_labels)
    # Repair the same noisy labels with K-NN.
    repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)
    # Train a fresh SVM on repaired labels; evaluate on clean test data.
    knn_repair_accuracy = train_and_evaluate(train_features, repaired_labels,
                                             test_features, test_labels)
    # Save per-trial accuracy for all methods to a CSV at output_path.
    from pathlib import Path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write data to CSV with columns: trial, baseline_accuracy, knn_repair_accuracy
    with open(output_path, "w") as f:
        f.write("trial,baseline_accuracy,knn_repair_accuracy\n")
        for trial in range(repeats):
            f.write(f"{trial},{baseline_accuracy},{knn_repair_accuracy}\n")
    # Report mean accuracy and the change from baseline.
    print(f"Mean baseline accuracy: {np.mean(baseline_accuracy):.2f}")
    print(f"Mean K-NN repair accuracy: {np.mean(knn_repair_accuracy):.2f}")
    print(f"Change from baseline: {np.mean(knn_repair_accuracy) - np.mean(baseline_accuracy):.2f}")


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
