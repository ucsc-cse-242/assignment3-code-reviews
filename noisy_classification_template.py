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
    noisy_labels = np.array(clean_labels)  # new array; clean_labels is untouched
    for i in range(len(clean_labels)):
        u = rng.random()
        # Decide using the ORIGINAL label, so a point is flipped at most once.
        if clean_labels[i] == 1 and u < POSITIVE_FLIP_PROBABILITY:
            noisy_labels[i] = -1
        elif clean_labels[i] == -1 and u < NEGATIVE_FLIP_PROBABILITY:
            noisy_labels[i] = 1
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

    # Find neighbors using training features only. Calling kneighbors() with no
    # query points excludes each point from its own neighborhood. Brute force
    # is deterministic, so equal-distance ties are broken the same way every
    # run (exact ties are practically impossible with continuous features).
    nn = NearestNeighbors(n_neighbors=neighbors, algorithm="brute")
    nn.fit(train_features)
    neighbor_idx = nn.kneighbors(return_distance=False)  # shape (n, neighbors)

    # Majority vote, always reading the ORIGINAL noisy labels.
    repaired_labels = np.array(noisy_labels)
    for i in range(len(noisy_labels)):
        vote_sum = 0
        for j in neighbor_idx[i]:
            vote_sum = vote_sum + noisy_labels[j]
        if vote_sum > 0:
            repaired_labels[i] = 1
        else:
            repaired_labels[i] = -1  # 5 votes of +/-1 never sum to 0
    return repaired_labels


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

    # Generate independent clean training and test splits (once).
    train_features, train_clean_labels = generate_clean_split(rng)
    test_features, test_labels = generate_clean_split(rng)

    baseline_accuracies = []
    repaired_accuracies = []
    for trial in range(repeats):
        # Corrupt training labels and evaluate the baseline.
        noisy_labels = corrupt_training_labels(train_clean_labels, rng)
        baseline_acc = train_and_evaluate(train_features, noisy_labels,
                                          test_features, test_labels)

        # Repair the same noisy labels with K-NN, train a fresh SVM, evaluate.
        repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)
        repaired_acc = train_and_evaluate(train_features, repaired_labels,
                                          test_features, test_labels)

        baseline_accuracies.append(baseline_acc)
        repaired_accuracies.append(repaired_acc)

    # Save per-trial accuracy for both methods to a CSV.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "baseline_accuracy", "knn_repaired_accuracy"])
        for trial in range(repeats):
            writer.writerow([trial, baseline_accuracies[trial], repaired_accuracies[trial]])

    # Report mean accuracy and the change from baseline.
    baseline_mean = np.mean(baseline_accuracies)
    repaired_mean = np.mean(repaired_accuracies)
    print("Saved per-trial results to", output_path)
    print("Mean baseline accuracy:     ", baseline_mean)
    print("Mean K-NN repaired accuracy:", repaired_mean)
    print("Change from baseline:       ", repaired_mean - baseline_mean)


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
