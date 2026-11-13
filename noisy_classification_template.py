"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import argparse
from pathlib import Path
import csv
import numpy as np

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
    noisy = clean_labels.copy()
    for i in range(len(clean_labels)):
        if clean_labels[i] == 1:
            p = POSITIVE_FLIP_PROBABILITY
        else:
            p = NEGATIVE_FLIP_PROBABILITY
        if rng.random() < p:
            noisy[i] = -clean_labels[i]
    return noisy



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
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=neighbors + 1)
    knn.fit(train_features)
    idx = knn.kneighbors(train_features, return_distance=False)

    new_labels = noisy_labels.copy()
    for i in range(len(noisy_labels)):
        votes = noisy_labels[idx[i][1:]]  # first one is the point itself
        if votes.sum() > 0:
            new_labels[i] = 1
        else:
            new_labels[i] = -1
    return new_labels



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
    rng = np.random.default_rng(seed)
    train_x, train_y = generate_clean_split(rng)
    test_x, test_y = generate_clean_split(rng)

    acc1 = []
    acc2 = []
    for trial in range(repeats):
        noisy_y = corrupt_training_labels(train_y, rng)
        acc1.append(train_and_evaluate(train_x, noisy_y, test_x, test_y))
        new_y = repair_labels_knn(train_x, noisy_y, neighbors)
        acc2.append(train_and_evaluate(train_x, new_y, test_x, test_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "baseline_accuracy", "knn_accuracy"])
        for trial in range(repeats):
            writer.writerow([trial, acc1[trial], acc2[trial]])

    mean1 = np.mean(acc1)
    mean2 = np.mean(acc2)
    print(f"Baseline mean accuracy: {mean1:.4f}")
    print(f"KNN mean accuracy: {mean2:.4f}")
    print(f"Difference (KNN - baseline): {mean2 - mean1:.4f}")



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
