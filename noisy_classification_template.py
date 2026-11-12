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
    #raise NotImplementedError("Implement asymmetric training-label noise.")
    import numpy as np
    clean_labels = np.asarray(clean_labels)
    # Flip probability chosen from the ORIGINAL label: +1 -> 0.35, -1 -> 0.20
    flip_prob = np.where(clean_labels == 1,
                         POSITIVE_FLIP_PROBABILITY, NEGATIVE_FLIP_PROBABILITY)
    # One uniform draw per example, so each example is flipped at most once
    flip = rng.random(clean_labels.shape[0]) < flip_prob
    noisy = clean_labels.copy()        # leave clean_labels unchanged
    noisy[flip] = -noisy[flip]
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
    #raise NotImplementedError("Implement K-NN majority-vote label repair.")
    import numpy as np
    X = np.asarray(train_features, dtype=float)
    noisy = np.asarray(noisy_labels)
    # Pairwise squared Euclidean distances between training points (n x n)
    sq = np.sum(X ** 2, axis=1)
    dist = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    # Exclude each point from its own neighborhood
    np.fill_diagonal(dist, np.inf)
    # Stable sort: equal-distance ties are broken by the lower row index
    nn_idx = np.argsort(dist, axis=1, kind="stable")[:, :neighbors]
    # Majority vote of the neighbors' ORIGINAL noisy labels (5 votes cannot tie)
    votes = noisy[nn_idx].sum(axis=1)
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
    #raise NotImplementedError("Complete the repeated noise-repair comparison.")
    import csv
    import numpy as np
    rng = np.random.default_rng(seed)

    # Independent clean training and test sets, generated once and kept fixed
    train_X, train_y = generate_clean_split(rng)
    test_X, test_y = generate_clean_split(rng)

    rows = []
    for trial in range(1, repeats + 1):
        noisy_y = corrupt_training_labels(train_y, rng)

        # (a) Baseline: SVM trained on noisy labels
        baseline_acc = train_and_evaluate(train_X, noisy_y, test_X, test_y)

        # (b) K-NN repair of the SAME noisy labels, then a fresh SVM
        repaired_y = repair_labels_knn(train_X, noisy_y, neighbors)
        knn_acc = train_and_evaluate(train_X, repaired_y, test_X, test_y)

        rows.append({"trial": trial,
                     "baseline_accuracy": baseline_acc,
                     "knn_accuracy": knn_acc})
        print(f"trial {trial:2d}: baseline={baseline_acc:.4f}  knn={knn_acc:.4f}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "baseline_accuracy", "knn_accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    mean_base = np.mean([r["baseline_accuracy"] for r in rows])
    mean_knn = np.mean([r["knn_accuracy"] for r in rows])
    print(f"Mean baseline accuracy:    {mean_base:.4f}")
    print(f"Mean K-NN repair accuracy: {mean_knn:.4f}")
    print(f"Difference (knn - baseline): {mean_knn - mean_base:+.4f}")
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
