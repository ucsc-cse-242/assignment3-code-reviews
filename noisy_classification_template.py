"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances

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
    # Choose probabilities using the ORIGINAL labels, before flipping.
    flip_probabilities = np.where(
        clean_labels == 1,
        POSITIVE_FLIP_PROBABILITY,
        NEGATIVE_FLIP_PROBABILITY,
    )

    # Each training example has its own independent random flip.
    should_flip = rng.random(len(clean_labels)) < flip_probabilities

    # Copy so the original clean training labels are never changed.
    noisy_labels = np.array(clean_labels, copy=True)
    noisy_labels[should_flip] *= -1
    return noisy_labels
    #raise NotImplementedError("Implement asymmetric training-label noise.")


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
    if neighbors < 1 or neighbors >= len(train_features):
        raise ValueError("neighbors must be between 1 and n_samples - 1")

    # Pairwise squared Euclidean distances among TRAINING points only.
    # Squared distance has the same ordering as Euclidean distance.
    distances = pairwise_distances(train_features, metric="sqeuclidean")

    # A point cannot vote for itself, even when its distance is zero.
    np.fill_diagonal(distances, np.inf)

    # Stable sorting breaks distance ties by the original row index.
    neighbor_indices = np.argsort(distances, axis=1, kind="stable")[:, :neighbors]

    # Every vote uses the unchanged noisy label array: no sequential edits.
    votes = np.sum(noisy_labels[neighbor_indices], axis=1)
    repaired_labels = np.where(votes > 0, 1, -1)
    return repaired_labels.astype(np.asarray(noisy_labels).dtype, copy=False)
    #raise NotImplementedError("Implement K-NN majority-vote label repair.")


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

    # Generate two independent splits ONCE and leave the test labels clean.
    train_features, clean_train_labels = generate_clean_split(rng)
    test_features, clean_test_labels = generate_clean_split(rng)

    results = []
    for trial in range(1, repeats + 1):
        # Corrupt the same original training labels independently per trial.
        noisy_labels = corrupt_training_labels(clean_train_labels, rng)

        # Baseline: train a new SVM using the corrupted labels.
        baseline_accuracy = train_and_evaluate(
            train_features, noisy_labels, test_features, clean_test_labels
        )

        # Repair these same labels with five-neighbor majority voting.
        repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)
        knn_accuracy = train_and_evaluate(
            train_features, repaired_labels, test_features, clean_test_labels
        )

        results.append({
            "trial": trial,
            "baseline_accuracy": baseline_accuracy,
            "knn_accuracy": knn_accuracy,
        })
        print(f"Trial {trial:2d}: baseline={baseline_accuracy:.4f}, "
              f"K-NN repair={knn_accuracy:.4f}")

    # CSV needs exactly one row per trial, with the requested columns.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("trial", "baseline_accuracy", "knn_accuracy"),
        )
        writer.writeheader()
        writer.writerows(results)

    baseline_mean = float(np.mean([row["baseline_accuracy"] for row in results]))
    knn_mean = float(np.mean([row["knn_accuracy"] for row in results]))
    improvement = knn_mean - baseline_mean

    print(f"\nBaseline mean accuracy: {baseline_mean:.4f} ({baseline_mean:.2%})")
    print(f"K-NN mean accuracy:      {knn_mean:.4f} ({knn_mean:.2%})")
    print(f"Change from baseline:    {improvement:+.4f} "
          f"({improvement * 100:+.2f} percentage points)")
    print(f"Saved trial results to: {output_path}")

    return results
    #raise NotImplementedError("Complete the repeated noise-repair comparison.")


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
