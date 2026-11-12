"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import argparse
from pathlib import Path
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
    labels = np.concatenate((np.ones(samples_per_class, dtype=int),
                             -np.ones(samples_per_class, dtype=int)))
    return features, labels


def corrupt_training_labels(clean_labels, rng):
    """Independently flip +1 with probability .35 and -1 with probability .20.

    Select flip probabilities from the ORIGINAL labels. Do not allow a label
    flipped in one direction to be flipped again by the other rule.
    Return a new array, leaving clean_labels unchanged.
    """
    # makes sure the labels are numpy arrays
    clean_labels = np.asarray(clean_labels)

    # makes one random number per example
    random_values = rng.random(len(clean_labels))

    # chooses each examples flip probability from og labels
    flip_probability = np.where(clean_labels == 1, POSITIVE_FLIP_PROBABILITY, NEGATIVE_FLIP_PROBABILITY)

    # flips example if its random value is below its probability
    # does one draw per example to prevent flipping twice
    flip = random_values < flip_probability

    # gives flipped examples opposite labels while non-flipped ones keep their original ones
    # np.where builds a new array to prevent clean_labels from changing to help keep original labels
    noisy_labels = np.where(flip, -clean_labels, clean_labels)

    #returns the calculated noisy labels
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
    # formatting noisy labels
    noisy_labels = np.asarray(noisy_labels)
    # storing shape of our training features
    n = train_features.shape[0]

    # makes new array for repaired labels
    repaired_labels = np.empty(n, dtype=int)

    for i in range(n):
        # gets squared euclidean distance from point i to each training point
        differences = train_features - train_features[i]
        distances = (differences ** 2).sum(axis=1)

        # excludes each point from its own neighborhood
        distances[i] = np.inf

        # indices of the 5 nearest neighbors
        # equal distances are broken by lower index first via stable sort
        nearest = np.argsort(distances, kind="stable")[:neighbors]

        # gets a majority vote of the neighbors og noisy labels
        # labels are +-1, so the sum is positive when +1 is the majority and negative vice versa
        vote = noisy_labels[nearest].sum()
        # 5 votes always has a majoirty, so never 0
        repaired_labels[i] = 1 if vote > 0 else -1

    #returns calced labels
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
    # uses seeded random generator for reproducibility
    gen = np.random.default_rng(seed)

    # generates independent clean training and test sets once
    train_features, train_labels = generate_clean_split(gen)
    test_features, test_labels = generate_clean_split(gen)

    #inits variables to be used later
    baseline_accuracies = []
    knn_accuracies = []
    rows = []

    # repeats corruption & training
    for trial in range(1, repeats + 1):
        # corrupts training labels
        noisy_labels = corrupt_training_labels(train_labels, gen)

        # baseline - trains SVM on noisy labels, evaluates on clean test set
        baseline_accuracy = train_and_evaluate(train_features, noisy_labels,
                                               test_features, test_labels)
        # adds calced baseline accuracies to list of accuracies
        baseline_accuracies.append(baseline_accuracy)

        # knn repair using the same features and noisy labels as baseline
        repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)

        # training fresh SVM on repaired labels & evaluates on clean test set
        knn_accuracy = train_and_evaluate(train_features, repaired_labels,
                                          test_features, test_labels)
        #appends accuracies of to list of accuracies for knn
        knn_accuracies.append(knn_accuracy)

        # stores the trial's results for the csv
        rows.append((trial, baseline_accuracy, knn_accuracy))
        
        # prints results
        print(f"trial {trial}: baseline = {baseline_accuracy:.4f}, knn = {knn_accuracy:.4f}")
        
    # saves per trial accuracies to csv
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "baseline_accuracy", "knn_accuracy"])
        writer.writerows(rows)
    print(f"saved results to {output_path}")

    # reports mean accuracies and the change from baseline
    baseline_mean = np.mean(baseline_accuracies)
    knn_mean = np.mean(knn_accuracies)
    print(f"mean baseline accuracy: {baseline_mean:.4f}")
    print(f"mean knn accuracy: {knn_mean:.4f}")
    print(f"difference (knn - baseline): {knn_mean - baseline_mean:+.4f}")


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
