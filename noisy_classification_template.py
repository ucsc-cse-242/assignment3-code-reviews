"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import numpy as np
import os
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
    # Draw flips and return the noisy labels.
    PROB_POS_FLIP = 0.35
    PROB_NEG_FLIP = 0.20
    noisy_labels = clean_labels.copy()  # to ensure the clean_label input is not flipped in-place.
    for i in range(noisy_labels.shape[0]):
        r = rng.random()
        if (noisy_labels[i] == 1 and r < PROB_POS_FLIP) or (noisy_labels[i] == -1 and r < PROB_NEG_FLIP):
            noisy_labels[i] *= -1
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
    from sklearn.neighbors import NearestNeighbors
    # Find neighbors using training features only. Use neighbors + 1 bc it will include the self by default.
    nbrs = NearestNeighbors(n_neighbors=neighbors + 1).fit(train_features)
    # Get distances, which is (2000, 6) with distances ascending; indices is (2000, 6) with each row listing the indices corresponding to the distances
    distances, indices = nbrs.kneighbors(train_features)
    # Exclude each self point to get 5 non-zero neighbors distances and their corresponding labels for each point.
    distances, indices = distances[:, 1:], indices[:, 1:]
    # Replace labels using the majority vote of their five neighbors. Use new list instead of modifying labels in-place.
    n = train_features.shape[0]
    clean_labels = np.ones((n))
    for i in range(n):
        neighbors_label_sum = noisy_labels[indices[i]].sum()
        clean_labels[i] = np.sign(neighbors_label_sum)
    return clean_labels


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
    # Generate independent clean training and test splits.
    rng = np.random.default_rng(seed)
    train_features, train_labels = generate_clean_split(rng)
    test_features, test_labels = generate_clean_split(rng)
    accuracies = np.zeros((repeats, 3))  # holds trial, baseline_accuracy, knn_accuracy
    # Train RBF SVM on clean train and test sets.
    clean_accuracy = train_and_evaluate(train_features, train_labels, test_features, test_labels)
    print(f"{clean_accuracy=}")
    for r in range(repeats):
        # For each repeat, corrupt training labels and evaluate the baseline.
        noisy_labels = corrupt_training_labels(train_labels, rng)
        # Train RBF SVM on noisy train and clean test set.
        noisy_accuracy = train_and_evaluate(train_features, noisy_labels, test_features, test_labels)
        # Repair the same noisy labels with K-NN.
        repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)
        # Train a fresh SVM on repaired labels; evaluate on clean test data.
        repair_accuracy = train_and_evaluate(train_features, repaired_labels, test_features, test_labels)
        accuracies[r] = [r, noisy_accuracy, repair_accuracy]
    # Save per-trial accuracy for all methods to a CSV at output_path.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, accuracies, delimiter=",", fmt="%g", header="trial,baseline_accuracy,knn_accuracy")
    # Report mean accuracy of noisy and repair trials and the difference between them.
    mean_noisy_accuracy = np.mean(accuracies[:, 1])
    mean_repair_accuracy = np.mean(accuracies[:, 2])
    accuracy_delta = mean_repair_accuracy - mean_noisy_accuracy
    print(f"{mean_noisy_accuracy=}, {mean_repair_accuracy=}, {accuracy_delta=}")
    return


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
