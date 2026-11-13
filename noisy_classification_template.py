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
    # TODO: Draw flips and return the noisy labels.
    # raise NotImplementedError("Implement asymmetric training-label noise.")
    noisy_labels = clean_labels.copy()
    flip_positive = (clean_labels == 1) & (rng.random(clean_labels.shape) < POSITIVE_FLIP_PROBABILITY)
    flip_negative = (clean_labels == -1) & (rng.random(clean_labels.shape) < NEGATIVE_FLIP_PROBABILITY)
    flip = flip_positive | flip_negative
    noisy_labels[flip] = -noisy_labels[flip]
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
    # TODO: Find neighbors using training features only.
    # TODO: Replace labels using the majority vote of their five neighbors.
    # raise NotImplementedError("Implement K-NN majority-vote label repair.")
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=neighbors + 1)
    nn.fit(train_features)
    _, indices = nn.kneighbors(train_features)

    n = train_features.shape[0]
    neighbor_indices = np.empty((n, neighbors), dtype=int)
    for i in range(n):
        row = indices[i]
        row = row[row != i]          # exclude the point itself
        neighbor_indices[i] = row[:neighbors]

    neighbor_labels = noisy_labels[neighbor_indices]
    votes = neighbor_labels.sum(axis=1)
    return np.where(votes > 0, 1, -1)

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
    # TODO: Generate independent clean training and test splits.
    # TODO: For each repeat, corrupt training labels and evaluate the baseline.
    # TODO: Repair the same noisy labels with K-NN.
    # TODO: Train a fresh SVM on repaired labels; evaluate on clean test data.
    # TODO: Save per-trial accuracy for all methods to a CSV at output_path.
    # TODO: Report mean accuracy and the change from baseline.
    # raise NotImplementedError("Complete the repeated noise-repair comparison.")
    import csv
    import numpy as np

    rng = np.random.default_rng(seed)
    train_features, train_labels = generate_clean_split(rng)
    test_features, test_labels = generate_clean_split(rng)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_accuracies = []
    repaired_accuracies = []

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "baseline_accuracy", "knn_accuracy"])
        for trial in range(repeats):
            noisy_labels = corrupt_training_labels(train_labels, rng)
            baseline_acc = train_and_evaluate(train_features, noisy_labels, test_features, test_labels)

            repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)
            repaired_acc = train_and_evaluate(train_features, repaired_labels, test_features, test_labels)

            baseline_accuracies.append(baseline_acc)
            repaired_accuracies.append(repaired_acc)
            writer.writerow([trial, baseline_acc, repaired_acc])

    mean_baseline = float(np.mean(baseline_accuracies))
    mean_repaired = float(np.mean(repaired_accuracies))
    print(f"Baseline mean accuracy: {mean_baseline:.4f}")
    print(f"K-NN repaired mean accuracy: {mean_repaired:.4f}")
    print(f"Change from baseline: {mean_repaired - mean_baseline:+.4f}")


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
