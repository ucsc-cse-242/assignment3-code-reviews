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
    # one uniform draw per example
    u = rng.random(clean_labels.shape[0])
    flip_probability = np.where(clean_labels == 1,
                                POSITIVE_FLIP_PROBABILITY,
                                NEGATIVE_FLIP_PROBABILITY)
    flip = u < flip_probability
    noisy = clean_labels.copy()
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
    # lazy loading modules
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    train_features = np.asarray(train_features)
    noisy_labels = np.asarray(noisy_labels)

    # fetch neighbors + 1 so the point itself (distance 0) can be dropped.
    # sklearn returns neighbors sorted by distance
    # TIE RULE: for exact distance the the point with the lower row index comes first.
    nn = NearestNeighbors(n_neighbors=neighbors + 1, metric="euclidean")
    nn.fit(train_features)
    _, idx = nn.kneighbors(train_features)

    # remove each point itself from its own neighbor list.  
    # normally it is in column 0 but with duplicate feature 
    # rows it may occur in some other position elsewhere
    self_idx = np.arange(train_features.shape[0])[:, np.newaxis]
    keep = idx != self_idx
    others = np.array([row[m][:neighbors] for row, m in zip(idx, keep)])

    # majority vote over the original noisy labels
    votes = noisy_labels[others].sum(axis=1)
    return np.where(votes > 0, 1, -1).astype(noisy_labels.dtype)


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
    # lazy loading modules
    import csv
    import numpy as np

    rng = np.random.default_rng(seed)
    
    train_features, train_labels_clean = generate_clean_split(rng)
    test_features, test_labels = generate_clean_split(rng)

    # SVM trained on the clean training labels.
    clean_accuracy = train_and_evaluate(train_features, train_labels_clean, test_features, test_labels)

    rows = []
    for trial in range(1, repeats + 1):
        # corrupt the fixed clean training labels and train the baseline.
        noisy_labels = corrupt_training_labels(train_labels_clean, rng)
        baseline_accuracy = train_and_evaluate(train_features, noisy_labels, test_features, test_labels)

        # Repair the SAME noisy labels with K-NN using only training
        # features and noisy labels, then train a fresh SVM.
        repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)
        knn_accuracy = train_and_evaluate(train_features, repaired_labels, test_features, test_labels)

        noise_rate = float(np.mean(noisy_labels != train_labels_clean))
        repaired_noise_rate = float(np.mean(repaired_labels != train_labels_clean))
        rows.append({"trial": trial,
                     "baseline_accuracy": baseline_accuracy,
                     "knn_accuracy": knn_accuracy,
                     "noisy_label_rate": noise_rate,
                     "repaired_label_rate": repaired_noise_rate})
        print(f"trial {trial:2d}: baseline {baseline_accuracy:.4f}  "
              f"knn-repair {knn_accuracy:.4f}  "
              f"(label noise {noise_rate:.3f} -> {repaired_noise_rate:.3f})")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    baseline_mean = float(np.mean([r["baseline_accuracy"] for r in rows]))
    knn_mean = float(np.mean([r["knn_accuracy"] for r in rows]))
    baseline_std = float(np.std([r["baseline_accuracy"] for r in rows]))
    knn_std = float(np.std([r["knn_accuracy"] for r in rows]))
    print(f"Clean-label reference accuracy : {clean_accuracy:.4f}")
    print(f"Baseline (noisy labels)  mean  : {baseline_mean:.4f} (std {baseline_std:.4f})")
    print(f"K-NN repaired labels     mean  : {knn_mean:.4f} (std {knn_std:.4f})")
    print(f"Difference (knn - baseline)    : {knn_mean - baseline_mean:+.4f}")
    print(f"Saved per-trial results to {output_path}")
    return {"clean_accuracy": clean_accuracy, "baseline_mean": baseline_mean, "knn_mean": knn_mean, "difference": knn_mean - baseline_mean}


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