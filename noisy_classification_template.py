"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

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
    # algorithm: iter and apply random chance, then append to new array
    noisy_labels = []

    for i, label in enumerate(clean_labels):
        if label == 1:
            if rng.random() < POSITIVE_FLIP_PROBABILITY:
                noisy_labels.append(-1)
            else:
                noisy_labels.append(1)

        else:
            if rng.random() < NEGATIVE_FLIP_PROBABILITY:
                noisy_labels.append(1)
            else:
                noisy_labels.append(-1)

    return np.array(noisy_labels)


def repair_labels_knn(train_features, noisy_labels, neighbors=5):
    """Q3(b): propose a K-NN label-repair method; return repaired labels.

    For this simplified assignment, use 5 neighbors, Euclidean distance,
    and replace each label with the majority label of those neighbors.
    Five binary votes cannot tie; break equal-distance neighbor ties consistently.
    Exclude each point itself from its neighborhood. Base all repairs on
    the original noisy labels, so row ordering does not affect the outcome.
    sklearn.neighbors may be used; from-scratch K-NN is not a source requirement.
    """
    repaired_labels = []
    # TODO: Find neighbors using training features only.
    for i, point in enumerate(train_features):
        distances = np.linalg.norm(train_features - point, axis=1)
        # eclude curr point
        distances[i] = np.inf
        nearest = np.argsort(distances, kind="stable")[:neighbors]
        neighbors_labels = noisy_labels[nearest]
        # TODO: Replace labels using the majority vote of their five neighbors.
        majority_label = 1 if np.sum(neighbors_labels == 1) > neighbors / 2 else -1
        repaired_labels.append(majority_label)

    return np.array(repaired_labels)


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

    rng = np.random.default_rng(seed=seed)
    # TODO: Generate independent clean training and test splits.
    X_train, y_train = generate_clean_split(rng)
    X_test, y_test = generate_clean_split(rng)

    # TODO: For each repeat, corrupt training labels and evaluate the baseline.
    results = []

    for trial in range(repeats):
        noisy_labels = corrupt_training_labels(y_train, rng)

        # baseline svm
        baseline_accuracy = train_and_evaluate(X_train, noisy_labels, X_test, y_test)

        # TODO: Repair the same noisy labels with K-NN.
        repaired_labels = repair_labels_knn(X_train, noisy_labels, neighbors)
        # TODO: Train a fresh SVM on repaired labels; evaluate on clean test data.
        knn_accuracy = train_and_evaluate(X_train, repaired_labels, X_test, y_test)

        # TODO: Save per-trial accuracy for all methods to a CSV at output_path.
        results.append({
            "trial": trial + 1,
            "baseline_accuracy": baseline_accuracy,
            "knn_accuracy": knn_accuracy,
            "improvement": knn_accuracy - baseline_accuracy
        })

    df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # TODO: Report mean accuracy and the change from baseline.
    baseline_mean = df["baseline_accuracy"].mean()
    knn_mean = df["knn_accuracy"].mean()

    print(f"Baseline mean accuracy: {baseline_mean:.4f}")
    print(f"KNN mean accuracy: {knn_mean:.4f}")
    print(f"Change from baseline: {knn_mean - baseline_mean:+.4f}")

    return df

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