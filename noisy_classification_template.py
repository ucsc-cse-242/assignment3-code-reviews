"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

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
    # Setup data and use original labels
    noisy_labels = clean_labels.copy()
    positive_mask = clean_labels == 1
    negative_mask = clean_labels == -1
    draws = rng.random(len(clean_labels))

    # Do the flips
    flip_positive = positive_mask & (draws < POSITIVE_FLIP_PROBABILITY)
    flip_negative = negative_mask & (draws < NEGATIVE_FLIP_PROBABILITY)
    noisy_labels[flip_positive] = -1
    noisy_labels[flip_negative] = 1
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
    # Ask for one extra neighbor since each point is its own closest match.
    nn = NearestNeighbors(n_neighbors=neighbors + 1, metric="euclidean")
    nn.fit(train_features)
    _, indices = nn.kneighbors(train_features)

    repaired_labels = np.empty_like(noisy_labels)
    for i in range(len(train_features)):
        # Drop the point itself (not just column 0, in case of duplicate points).
        others = [j for j in indices[i] if j != i][:neighbors]
        votes = noisy_labels[others]  # always the ORIGINAL noisy labels
        repaired_labels[i] = 1 if votes.sum() > 0 else -1
    return repaired_labels


def train_and_evaluate(train_features, train_labels, test_features, test_labels):
    """Train the required RBF SVM and return accuracy on the clean test set."""
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
    # Set up a seeded random generator.
    rng = np.random.default_rng(seed)
 
    # Generate a clean training set and a clean test set.
    train_features, clean_train_labels = generate_clean_split(rng)
    test_features, test_labels = generate_clean_split(rng)
 
    rows = []
    for trial in range(1, repeats + 1):
        # Corrupt the training labels.
        noisy_labels = corrupt_training_labels(clean_train_labels, rng)
 
        # Repair the noisy labels using their nearest neighbors
        # (only noisy labels are used, never the clean ones).
        repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)
 
        # Train and test twice: once on the noisy labels, once on the repaired ones.
        baseline_acc = train_and_evaluate(train_features, noisy_labels,
                                          test_features, test_labels)
        knn_acc = train_and_evaluate(train_features, repaired_labels,
                                     test_features, test_labels)
 
        # Record both accuracies.
        rows.append((trial, baseline_acc, knn_acc))
 
    # Save all trial results to a CSV (trial, baseline_accuracy, knn_accuracy).
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "baseline_accuracy", "knn_accuracy"])
        writer.writerows(rows)
 
    # Print the average of each accuracy and the difference between them.
    baseline_mean = np.mean([r[1] for r in rows])
    knn_mean = np.mean([r[2] for r in rows])
    print(f"Baseline mean accuracy: {baseline_mean:.4f}")
    print(f"K-NN repair mean accuracy: {knn_mean:.4f}")
    print(f"Change from baseline: {knn_mean - baseline_mean:+.4f}")
    print(f"Saved per-trial results to {output_path}")


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
