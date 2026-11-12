"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import argparse
import csv
from pathlib import Path

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
    clean_labels = np.asarray(clean_labels)
    if clean_labels.ndim != 1 or not np.all(np.isin(clean_labels, (-1, 1))):
        raise ValueError("clean_labels must be a one-dimensional array of -1/+1 labels.")

    # Each probability is selected from the original label, so no example can
    # be flipped once as positive and then flipped again as negative.
    flip_probabilities = np.where(
        clean_labels == 1,
        POSITIVE_FLIP_PROBABILITY,
        NEGATIVE_FLIP_PROBABILITY,
    )
    flips = rng.random(clean_labels.shape) < flip_probabilities
    return np.where(flips, -clean_labels, clean_labels)


def repair_labels_knn(train_features, noisy_labels, neighbors):
    """Q3(b): propose a K-NN label-repair method; return repaired labels.

    For this simplified assignment, use 5 neighbors, Euclidean distance,
    and replace each label with the majority label of those neighbors.
    Five binary votes cannot tie; break equal-distance neighbor ties consistently.
    Exclude each point itself from its neighborhood. Base all repairs on
    the original noisy labels, so row ordering does not affect the outcome.
    sklearn.neighbors may be used; from-scratch K-NN is not a source requirement.
    """
    train_features = np.asarray(train_features, dtype=float)
    noisy_labels = np.asarray(noisy_labels)
    if train_features.ndim != 2:
        raise ValueError("train_features must be a two-dimensional array.")
    if noisy_labels.shape != (train_features.shape[0],):
        raise ValueError("noisy_labels must contain one label per training row.")
    if not np.all(np.isin(noisy_labels, (-1, 1))):
        raise ValueError("noisy_labels must contain only -1/+1 labels.")
    if not isinstance(neighbors, (int, np.integer)):
        raise ValueError("neighbors must be an integer.")
    if neighbors < 1 or neighbors >= train_features.shape[0]:
        raise ValueError("neighbors must be between 1 and n_samples - 1.")

    # Sorting by (distance, original row index) makes equal-distance ties
    # deterministic, and the original labels are used for every vote.
    squared_distances = np.sum(
        (train_features[:, np.newaxis, :] - train_features[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    np.fill_diagonal(squared_distances, np.inf)
    neighbor_indices = np.argsort(squared_distances, axis=1, kind="stable")[:, :neighbors]
    neighbor_votes = noisy_labels[neighbor_indices]
    vote_totals = np.sum(neighbor_votes, axis=1)
    return np.where(vote_totals >= 0, 1, -1).astype(noisy_labels.dtype, copy=False)


def train_and_evaluate(train_features, train_labels, test_features, test_labels):
    """Train the required RBF SVM and return accuracy on the clean test set."""
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    model = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)
    model.fit(train_features, train_labels)
    return float(accuracy_score(test_labels, model.predict(test_features)))


def run_comparison(repeats, seed, neighbors, output_path):
    """Q3(a-b), 20 points each: compare the baseline with K-NN label repair.

    The random generator uses ``seed`` to create one fixed training split and
    one fixed test split. Those same splits are used for every noise trial;
    only the training-label corruption changes between trials.
    Do not give the repair method the clean training or test labels.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if neighbors != 5:
        raise ValueError("This assignment requires exactly five neighbors.")

    rng = np.random.default_rng(seed)
    train_features, clean_train_labels = generate_clean_split(rng)
    test_features, clean_test_labels = generate_clean_split(rng)
    results = []

    for trial in range(1, repeats + 1):
        noisy_labels = corrupt_training_labels(clean_train_labels, rng)
        baseline_accuracy = train_and_evaluate(
            train_features, noisy_labels, test_features, clean_test_labels
        )
        repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)
        knn_accuracy = train_and_evaluate(
            train_features, repaired_labels, test_features, clean_test_labels
        )
        results.append((trial, baseline_accuracy, knn_accuracy))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("trial", "baseline_accuracy", "knn_accuracy"))
        writer.writerows(results)

    baseline_mean = float(np.mean([row[1] for row in results]))
    knn_mean = float(np.mean([row[2] for row in results]))
    print(f"Random seed: {seed}")
    print(f"Trials: {repeats} (fixed training/test sets)")
    print(f"Mean baseline accuracy: {baseline_mean:.4f}")
    print(f"Mean K-NN accuracy: {knn_mean:.4f}")
    print(f"Mean change: {knn_mean - baseline_mean:+.4f}")
    return results


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
