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
    flip_prob = np.where(clean_labels == 1, POSITIVE_FLIP_PROBABILITY,
                         NEGATIVE_FLIP_PROBABILITY)
    flips = rng.random(clean_labels.shape[0]) < flip_prob
    noisy = clean_labels.copy()
    noisy[flips] = -clean_labels[flips]
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
    # Ask for neighbors+1 so the point itself (distance 0) can be dropped.
    # Distance ties are broken by sklearn's brute-force search order
    # (stable sort, so the lower row index wins), which is deterministic.
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    noisy_labels = np.asarray(noisy_labels)
    nn = NearestNeighbors(n_neighbors=neighbors + 1, algorithm="brute",
                          metric="euclidean").fit(train_features)
    _, idx = nn.kneighbors(train_features)
    n = len(noisy_labels)
    neighbor_idx = np.empty((n, neighbors), dtype=int)
    for i in range(n):
        others = idx[i][idx[i] != i][:neighbors]  # exclude self robustly
        neighbor_idx[i] = others
    votes = noisy_labels[neighbor_idx].sum(axis=1)  # labels are +/-1
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
    import csv
    import numpy as np
    rng = np.random.default_rng(seed)
    train_x, train_y = generate_clean_split(rng)
    test_x, test_y = generate_clean_split(rng)

    rows = []
    for trial in range(1, repeats + 1):
        noisy_y = corrupt_training_labels(train_y, rng)
        baseline = train_and_evaluate(train_x, noisy_y, test_x, test_y)
        repaired_y = repair_labels_knn(train_x, noisy_y, neighbors)
        knn = train_and_evaluate(train_x, repaired_y, test_x, test_y)
        rows.append((trial, baseline, knn))
        # train_y is used here only to report diagnostics, never by the repair.
        print(f"trial {trial:2d}: noisy-label rate={np.mean(noisy_y != train_y):.3f} "
              f"repaired-label error={np.mean(repaired_y != train_y):.3f} "
              f"baseline={baseline:.4f} knn={knn:.4f}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "baseline_accuracy", "knn_accuracy"])
        writer.writerows(rows)

    base_mean = float(np.mean([r[1] for r in rows]))
    knn_mean = float(np.mean([r[2] for r in rows]))
    print(f"Mean baseline accuracy: {base_mean:.4f}")
    print(f"Mean K-NN-repair accuracy: {knn_mean:.4f}")
    print(f"Difference (knn - baseline): {knn_mean - base_mean:+.4f}")
    print(f"Saved per-trial results to {output_path}")
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
