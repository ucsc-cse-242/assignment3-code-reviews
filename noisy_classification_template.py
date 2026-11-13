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

    One uniform draw per example is compared against the flip probability
    chosen by its original label, so each label is flipped at most once.
    """
    import numpy as np
    clean_labels = np.asarray(clean_labels)
    flip_probabilities = np.where(clean_labels == 1, POSITIVE_FLIP_PROBABILITY,
                                  NEGATIVE_FLIP_PROBABILITY)
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

    Neighbors are found from the full pairwise Euclidean distance matrix with
    the diagonal set to infinity, which excludes each point itself. A stable
    sort breaks equal-distance ties in favor of the lower row index.
    """
    import numpy as np
    distances = np.linalg.norm(
        train_features[:, None, :] - train_features[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbor_indices = np.argsort(distances, axis=1, kind="stable")[:, :neighbors]
    # Every vote reads the original noisy labels, so row order does not matter.
    votes = noisy_labels[neighbor_indices].sum(axis=1)
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

    Clean training labels are used only after repair, to report how many
    training labels were correct before and after repair.
    """
    import csv
    import numpy as np

    rng = np.random.default_rng(seed)
    train_features, train_labels = generate_clean_split(rng)
    test_features, test_labels = generate_clean_split(rng)

    rows = []
    noisy_label_accuracy = []
    repaired_label_accuracy = []
    for trial in range(1, repeats + 1):
        noisy_labels = corrupt_training_labels(train_labels, rng)
        baseline_accuracy = train_and_evaluate(
            train_features, noisy_labels, test_features, test_labels)

        repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)
        knn_accuracy = train_and_evaluate(
            train_features, repaired_labels, test_features, test_labels)

        rows.append({"trial": trial, "baseline_accuracy": baseline_accuracy,
                     "knn_accuracy": knn_accuracy})
        noisy_label_accuracy.append(np.mean(noisy_labels == train_labels))
        repaired_label_accuracy.append(np.mean(repaired_labels == train_labels))
        print(f"Trial {trial:2d}: baseline = {baseline_accuracy:.4f}, "
              f"K-NN repair = {knn_accuracy:.4f}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "baseline_accuracy",
                                               "knn_accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    baseline_mean = np.mean([row["baseline_accuracy"] for row in rows])
    knn_mean = np.mean([row["knn_accuracy"] for row in rows])
    print(f"\nMean baseline accuracy:    {baseline_mean:.4f}")
    print(f"Mean K-NN repair accuracy: {knn_mean:.4f}")
    print(f"Difference (K-NN - baseline): {knn_mean - baseline_mean:+.4f}")
    print(f"Training labels correct: noisy = {np.mean(noisy_label_accuracy):.4f}, "
          f"repaired = {np.mean(repaired_label_accuracy):.4f}")
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
