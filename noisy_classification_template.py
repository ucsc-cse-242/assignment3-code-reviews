"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

# Outside-help acknowledgment: this solution was written with the help of
# Claude, a generative AI assistant made by Anthropic.

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

    Method: each example gets a flip probability chosen from its ORIGINAL label
    (0.35 for +1, 0.20 for -1) and one uniform draw u_i. The label is flipped
    exactly when u_i < p_i. The output is built in a single np.where from the
    untouched original array, so no example can be flipped twice.
    """
    clean = np.asarray(clean_labels)
    if not np.all(np.isin(clean, (-1, 1))):
        raise ValueError("Labels must be +1 or -1.")
    flip_probability = np.where(clean == 1, POSITIVE_FLIP_PROBABILITY,
                                NEGATIVE_FLIP_PROBABILITY)
    flip = rng.random(clean.shape[0]) < flip_probability
    return np.where(flip, -clean, clean).astype(int)


def repair_labels_knn(train_features, noisy_labels, neighbors):
    """Q3(b): propose a K-NN label-repair method; return repaired labels.

    For this simplified assignment, use 5 neighbors, Euclidean distance,
    and replace each label with the majority label of those neighbors.
    Five binary votes cannot tie; break equal-distance neighbor ties consistently.
    Exclude each point itself from its neighborhood. Base all repairs on
    the original noisy labels, so row ordering does not affect the outcome.
    sklearn.neighbors may be used; from-scratch K-NN is not a source requirement.

    Method:
    1. Compute all pairwise squared Euclidean distances between training points.
       They rank neighbours exactly as Euclidean distance does. Each point's
       distance to itself is set to +inf so it can never be its own neighbour.
    2. Sort each row with a stable sort and keep the first `neighbors` columns.
       Tie rule: when several candidates are equally far away, the one with
       the lower training index comes first.
    3. Each repaired label is the sign of the sum of the neighbours' labels.
       Those labels are read from the ORIGINAL noisy array, and every label is
       computed at once, so no repair can influence another. Five +/-1 votes
       always sum to an odd number, so there is no tie. For an even
       `neighbors` a tied vote would keep the point's own noisy label.

    Only the training features and the noisy training labels are used here.
    """
    features = np.asarray(train_features, dtype=float)
    noisy = np.asarray(noisy_labels).astype(int)
    n_points = features.shape[0]
    if noisy.shape != (n_points,):
        raise ValueError("Need exactly one noisy label per training point.")
    if not 1 <= neighbors < n_points:
        raise ValueError("neighbors must be between 1 and n_points - 1.")

    squared_distances = np.zeros((n_points, n_points))
    for column in range(features.shape[1]):
        delta = features[:, column, np.newaxis] - features[np.newaxis, :, column]
        squared_distances += delta * delta
    np.fill_diagonal(squared_distances, np.inf)  # exclude each point itself

    neighbor_index = np.argsort(squared_distances, axis=1,
                                kind="stable")[:, :neighbors]
    votes = noisy[neighbor_index].sum(axis=1)  # read from the original noisy labels
    repaired = np.where(votes > 0, 1, np.where(votes < 0, -1, noisy))
    return repaired.astype(int)


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

    Randomness: one seed is split into three independent streams with
    np.random.SeedSequence: one for the training set, one for the test set,
    and one for the label noise. The run is fully reproducible, and the noise
    draws are independent of the data draws.

    Writes a CSV with exactly the columns trial, baseline_accuracy,
    knn_accuracy (one row per trial) and returns a summary dictionary.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    train_seed, test_seed, noise_seed = np.random.SeedSequence(seed).spawn(3)

    # Independent clean training and test sets, generated once and kept fixed.
    train_features, train_clean_labels = generate_clean_split(
        np.random.default_rng(train_seed))
    test_features, test_labels = generate_clean_split(
        np.random.default_rng(test_seed))
    noise_rng = np.random.default_rng(noise_seed)

    print(f"Training set: {len(train_clean_labels)} points; clean test set: "
          f"{len(test_labels)} points; RBF SVM C = {SVM_C}, gamma = {SVM_GAMMA}; "
          f"flip probabilities +1: {POSITIVE_FLIP_PROBABILITY}, "
          f"-1: {NEGATIVE_FLIP_PROBABILITY}; K-NN repair with k = {neighbors}; "
          f"{repeats} trials; seed = {seed}")

    rows = []
    noise_before, noise_after = [], []
    for trial in range(1, repeats + 1):
        noisy_labels = corrupt_training_labels(train_clean_labels, noise_rng)

        # (a) Baseline: SVM trained directly on the noisy labels.
        baseline_accuracy = train_and_evaluate(
            train_features, noisy_labels, test_features, test_labels)

        # (b) The repair sees ONLY the training features and the noisy labels,
        # the same ones the baseline used in this trial. A fresh SVM with the
        # same parameters is then trained on the repaired labels.
        repaired_labels = repair_labels_knn(train_features, noisy_labels,
                                            neighbors)
        knn_accuracy = train_and_evaluate(
            train_features, repaired_labels, test_features, test_labels)

        rows.append((trial, baseline_accuracy, knn_accuracy))

        # Diagnostics for the report only: the training-label error rate before
        # and after repair. The clean training labels are used here to MEASURE
        # the noise. They are never passed to repair_labels_knn or to the SVM.
        noise_before.append(float(np.mean(noisy_labels != train_clean_labels)))
        noise_after.append(float(np.mean(repaired_labels != train_clean_labels)))
        print(f"trial {trial:2d}: baseline accuracy {baseline_accuracy:.4f} | "
              f"K-NN repair accuracy {knn_accuracy:.4f} | training-label error "
              f"{noise_before[-1]:.1%} -> {noise_after[-1]:.1%}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "baseline_accuracy", "knn_accuracy"])
        writer.writerows(rows)

    baseline = np.array([row[1] for row in rows])
    repaired = np.array([row[2] for row in rows])
    difference = repaired - baseline
    summary = {
        "mean_baseline_accuracy": float(baseline.mean()),
        "mean_knn_accuracy": float(repaired.mean()),
        "mean_difference": float(difference.mean()),
        "std_baseline_accuracy": float(baseline.std(ddof=1)) if repeats > 1 else 0.0,
        "std_knn_accuracy": float(repaired.std(ddof=1)) if repeats > 1 else 0.0,
        "knn_better_trials": int(np.sum(difference > 0)),
        "knn_worse_trials": int(np.sum(difference < 0)),
        "tied_trials": int(np.sum(difference == 0)),
        "mean_training_label_error_before": float(np.mean(noise_before)),
        "mean_training_label_error_after": float(np.mean(noise_after)),
    }

    print()
    print(f"Saved per-trial accuracies to {output_path}")
    print(f"Mean clean-test accuracy, baseline (noisy labels): "
          f"{summary['mean_baseline_accuracy']:.4f} "
          f"(sd {summary['std_baseline_accuracy']:.4f})")
    print(f"Mean clean-test accuracy, K-NN repaired labels:    "
          f"{summary['mean_knn_accuracy']:.4f} "
          f"(sd {summary['std_knn_accuracy']:.4f})")
    print(f"Difference (K-NN repair - baseline): {summary['mean_difference']:+.4f} "
          f"({100 * summary['mean_difference']:+.2f} percentage points); K-NN better "
          f"in {summary['knn_better_trials']}/{repeats} trials, worse in "
          f"{summary['knn_worse_trials']}, tied in {summary['tied_trials']}")
    if repeats > 1 and np.std(difference) > 0:
        from scipy import stats
        t_statistic, p_value = stats.ttest_rel(repaired, baseline)
        summary["paired_t_statistic"] = float(t_statistic)
        summary["paired_t_p_value"] = float(p_value)
        print(f"Paired t-test over trials: t = {t_statistic:.2f}, "
              f"p = {p_value:.2g}")
    print("Mean training-label error: "
          f"{summary['mean_training_label_error_before']:.1%} before repair -> "
          f"{summary['mean_training_label_error_after']:.1%} after repair "
          "(diagnostic only)")
    return summary


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
