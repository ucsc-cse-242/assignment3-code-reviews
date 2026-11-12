"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import csv
import argparse
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

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

    # Array of flip probs depending on label (+1 or -1)
    flip_prob = np.where(clean_labels == 1,
                         POSITIVE_FLIP_PROBABILITY,  # 0.35
                         NEGATIVE_FLIP_PROBABILITY)  # 0.20

    # Random values [0, 1)
    draws = rng.random(clean_labels.shape)

    # Booleans if label should flip
    flip = draws < flip_prob

    # Array of corrupted labels
    return np.where(flip, -clean_labels, clean_labels)


def repair_labels_knn(train_features, noisy_labels, neighbors: int):
    """Q3(b): propose a K-NN label-repair method; return repaired labels.

    For this simplified assignment, use 5 neighbors, Euclidean distance,
    and replace each label with the majority label of those neighbors.
    Five binary votes cannot tie; break equal-distance neighbor ties consistently.
    Exclude each point itself from its neighborhood. Base all repairs on
    the original noisy labels, so row ordering does not affect the outcome.
    sklearn.neighbors may be used; from-scratch K-NN is not a source requirement.

    Equal-distance ties between candidate neighbors are resolved by
      sklearn's brute-force search, which is deterministic for a given input.
    """

    nn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean",
                          algorithm="brute")
    nn.fit(train_features)

    # With no query set given, each training point is not its own neighbor.
    # Result has shape (n, neighbors): row numbers of each point's neighbors.
    neighbor_idx = nn.kneighbors(return_distance=False)

    # (n, neighbors) array of the neighbors' noisy labels, summed per point.
    # Positive sum -> majority +1, negative sum -> majority -1.
    votes = noisy_labels[neighbor_idx].sum(axis=1)

    # Replace labels using the majority vote of their five neighbors.
    return np.where(votes > 0, 1, -1)


def train_and_evaluate(train_features, train_labels, test_features, test_labels):
    """Train the required RBF SVM and return accuracy on the clean test set."""
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    model = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)
    model.fit(train_features, train_labels)
    return float(accuracy_score(test_labels, model.predict(test_features)))


def run_comparison(repeats, seed, neighbors: int, output_path):
    """Q3(a-b), 20 points each: compare the baseline with K-NN label repair.

    Suggested adaptation: generate independent clean training/test sets once,
    then repeat corruption and training (default: 20 noise trials). In each
    trial, use identical data and corrupted labels for both approaches.
    Do not give the repair method the clean training or test labels.
    """

    # TODO: Save per-trial accuracy for all methods to a CSV at output_path.
    # TODO: Report mean accuracy and the change from baseline.
    rng = np.random.default_rng(seed)

    # Generate independent clean training and test splits.
    X_train, y_train = generate_clean_split(rng)
    X_test, y_test = generate_clean_split(rng)

    baseline_accuracies = []
    knn_accuracies = []

    # For each repeat, corrupt training labels and evaluate the baseline.
    for _ in range(repeats):
        # Baseline
        y_train_noisy = corrupt_training_labels(y_train, rng)
        baseline_accuracy = train_and_evaluate(X_train, y_train_noisy, X_test, y_test)
        baseline_accuracies.append(baseline_accuracy)

        # KNN Repair the same noisy labels with K-NN.
        y_train_repaired = repair_labels_knn(X_train, y_train_noisy, neighbors)
        # Train a fresh SVM on repaired labels; evaluate on clean test data.
        repaired_accuracy = train_and_evaluate(X_train, y_train_repaired, X_test, y_test)
        knn_accuracies.append(repaired_accuracy)


    # Save per-trial accuracies: one row per trial.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "baseline_accuracy", "knn_accuracy"])
        for trial, (base_acc, knn_acc) in enumerate(
                zip(baseline_accuracies, knn_accuracies), start=1):
            writer.writerow([trial, base_acc, knn_acc])

    # Report mean accuracies and the change from baseline.
    baseline_mean = np.mean(baseline_accuracies)
    knn_mean = np.mean(knn_accuracies)
    print(f"Baseline mean accuracy: {baseline_mean:.4f}")
    print(f"K-NN repair mean accuracy: {knn_mean:.4f}")
    print(f"Change from baseline (K-NN - baseline): {knn_mean - baseline_mean:+.4f}")


    # print(train_and_evaluate(X_train, y_train, X_test, y_test))   # Non noisy baseline: 0.867
    """
    Discussion: 
    Baseline mean accuracy: 0.8280
    K-NN repair mean accuracy: 0.8644
    Change from baseline (K-NN - baseline): +0.0364
    
    The KNN repair helped; it improved mean accuracy by 0.0364. KNN repair can help when its a point
    that is flipped when surrounded by neighbors that had the same original label.
    But KNN repair can hurt near where the blobs overlap so the point has neighbors with truly a 
    different value than what the point should have, but then it will get erroneously flipped.
    """


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
