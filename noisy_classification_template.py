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
    import numpy as np
    clean_labels = np.asarray(clean_labels)
    noisy_labels = clean_labels.copy()
    flips = np.zeros(clean_labels.shape, dtype=bool)
    positive = clean_labels == 1
    negative = clean_labels == -1
    flips[positive] = rng.random(np.count_nonzero(positive)) < POSITIVE_FLIP_PROBABILITY
    flips[negative] = rng.random(np.count_nonzero(negative)) < NEGATIVE_FLIP_PROBABILITY
    noisy_labels[flips] *= -1
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
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    train_features = np.asarray(train_features)
    noisy_labels = np.asarray(noisy_labels)
    if not 1 <= neighbors < len(train_features):
        raise ValueError("neighbors must be positive and smaller than the training set")
    search = NearestNeighbors(n_neighbors=len(train_features), metric="euclidean")
    search.fit(train_features)
    distances, indices = search.kneighbors(train_features)
    repaired = np.empty_like(noisy_labels)
    for row in range(len(train_features)):
        valid = indices[row] != row
        row_indices = indices[row][valid]
        row_distances = distances[row][valid]
        order = np.lexsort((row_indices, row_distances))[:neighbors]
        votes = noisy_labels[row_indices[order]]
        repaired[row] = 1 if votes.sum() > 0 else -1
    return repaired


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
    import csv
    import numpy as np
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    rng = np.random.default_rng(seed)
    train_features, clean_train_labels = generate_clean_split(rng)
    test_features, clean_test_labels = generate_clean_split(rng)
    rows = []
    for trial in range(1, repeats + 1):
        noisy_labels = corrupt_training_labels(clean_train_labels, rng)
        baseline = train_and_evaluate(train_features, noisy_labels, test_features, clean_test_labels)
        repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)
        knn_accuracy = train_and_evaluate(train_features, repaired_labels, test_features, clean_test_labels)
        rows.append((trial, baseline, knn_accuracy))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("trial", "baseline_accuracy", "knn_accuracy"))
        writer.writerows(rows)
    baseline_mean = float(np.mean([row[1] for row in rows]))
    knn_mean = float(np.mean([row[2] for row in rows]))
    print(f"Baseline mean accuracy: {baseline_mean:.4f}")
    print(f"K-NN repair mean accuracy: {knn_mean:.4f}")
    print(f"Difference: {knn_mean - baseline_mean:+.4f}")
    print(f"Saved trial results to {output_path}")
    return rows


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
