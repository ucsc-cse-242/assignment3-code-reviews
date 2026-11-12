"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import argparse
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import numpy as np
import csv

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
    labels = np.concatenate(
        (np.ones(samples_per_class, dtype=int), -np.ones(samples_per_class, dtype=int))
    )
    return features, labels


def corrupt_training_labels(clean_labels, rng):
    """Independently flip +1 with probability .35 and -1 with probability .20.

    Select flip probabilities from the ORIGINAL labels. Do not allow a label
    flipped in one direction to be flipped again by the other rule.
    Return a new array, leaving clean_labels unchanged.
    """
    # TODO: Draw flips and return the noisy labels.

    # Copy for new_labels to maintain consistent array sizing
    new_label = clean_labels.copy()

    # Obtain random values from 0-1 based on total elements of clean_labels
    random_val = rng.random(clean_labels.shape)

    # Increment through clean_labels, check through the current index of clean_label and random_val, and if random_val
    # is less than the probability, then overwrite the new_label current index with the new value
    for idx in range(len(clean_labels)):
        if clean_labels[idx] == 1:
            if random_val[idx] < POSITIVE_FLIP_PROBABILITY:
                new_label[idx] = -1

        else:
            if random_val[idx] < NEGATIVE_FLIP_PROBABILITY:
                new_label[idx] = 1

    return new_label


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

    # Obtaining 5 + neighbors since first neighbor is always themselves
    near_neighbors = NearestNeighbors(n_neighbors=neighbors + 1, algorithm="auto").fit(
        train_features
    )

    distances, indices = near_neighbors.kneighbors(
        train_features, n_neighbors=neighbors + 1
    )

    # Cuts the first column out from indices since the first point of each neighbor is themselves, therefore is always 0
    neighbor_indices = indices[:, 1:]

    # Neighbor_indices contain positions in noisy_labels, can map them and return the corresponding label into neighbor_labels
    neighbor_labels = noisy_labels[neighbor_indices]

    # Created copy for repaired_labels to maintain consistent array size
    repaired_labels = noisy_labels.copy()

    # Enumerated through neighbor_labels and sum each label values
    ## If positive, repair label with +1, else -1
    for idx, label in enumerate(neighbor_labels):
        currentTotal = 0
        for noise in label:
            currentTotal += noise

        if currentTotal > 0:
            repaired_labels[idx] = 1

        else:
            repaired_labels[idx] = -1

    return repaired_labels

    # TODO: Replace labels using the majority vote of their five neighbors.


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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    results = []

    # Generate independent clean training and test splits.
    train_features, train_labels = generate_clean_split(rng)
    test_features, test_labels = generate_clean_split(rng)

    # For each repeat, corrupt training labels and evaluate the baseline.
    for idx in range(repeats):
        corrupt_labels = corrupt_training_labels(train_labels, rng)

        baseline = train_and_evaluate(
            train_features, corrupt_labels, test_features, test_labels
        )
        # Repair the same noisy labels with K-NN.
        repaired_labels = repair_labels_knn(train_features, corrupt_labels, neighbors)

        # Train a fresh SVM on repaired labels; evaluate on clean test data.
        knn_accuracy = train_and_evaluate(
            train_features, repaired_labels, test_features, test_labels
        )

        results.append(
            {
                "trial": idx + 1,
                "baseline_accuracy": baseline,
                "knn_accuracy": knn_accuracy,
            }
        )

    # Append all elements from results onto a .csv file
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "trial",
                "baseline_accuracy",
                "knn_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    # Obtained both baseline and new mean for comparison
    baseline_mean = np.mean([result["baseline_accuracy"] for result in results])
    knn_mean = np.mean([result["knn_accuracy"] for result in results])

    print("Baseline Mean: ", baseline_mean)
    print("KNN Mean: ", knn_mean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        choices=(5,),
        help="Use five neighbors for this assignment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "noise_comparison.csv",
    )
    args = parser.parse_args()
    run_comparison(args.repeats, args.seed, args.neighbors, args.output)


if __name__ == "__main__":
    main()
