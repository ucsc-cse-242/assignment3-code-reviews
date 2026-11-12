"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import argparse
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
    new_labels = []
    for l in clean_labels:
        r = rng.random() # draw random between 0 and 1
        if (r <= POSITIVE_FLIP_PROBABILITY and l > 0): # flip positive
            new_labels.append(-1)
        elif (r <= NEGATIVE_FLIP_PROBABILITY and l < 0): # flip negative
            new_labels.append(1)
        else:
            new_labels.append(l)

    return new_labels


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
    nearest = NearestNeighbors(n_neighbors=(neighbors + 1) ) # plus one because same point will be excluded
    nearest.fit(train_features)
    repaired_labels = []
    _, neigh_ind = nearest.kneighbors( train_features ) # find each index's nearest indexes
    for i in range(len(neigh_ind)):
        sum = 0
        for ind in neigh_ind[i]:
            if (not ind == i): # don't count the same point
                # labels are either +1 or -1, 
                # can sum them and check if result is + or -
                sum += noisy_labels[ind]
        if (sum > 0):
            repaired_labels.append(1)
        else:
            repaired_labels.append(-1)
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
    # Generate independent clean training and test splits.
    # For each repeat, corrupt training labels and evaluate the baseline.
    # Repair the same noisy labels with K-NN.
    # Train a fresh SVM on repaired labels; evaluate on clean test data.
    # Save per-trial accuracy for all methods to a CSV at output_path.
    # Report mean accuracy and the change from baseline.
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(seed)
    # generate features and labels
    features, clean_labels = generate_clean_split(rng)
    # split data
    train_features, test_features, train_clean_labels, test_clean_labels = train_test_split(
        features, clean_labels, train_size=0.8)

    baseline = []
    repaired = []
    for _ in range(repeats):
        # add noise to train data
        train_noisy_labels = corrupt_training_labels(train_clean_labels, rng)
        # evaluate baseline with noisy data
        baseline.append( train_and_evaluate(train_features, train_noisy_labels, test_features, test_clean_labels) )

        # repair noisy data with knn
        train_repaired_labled = repair_labels_knn(train_features, train_noisy_labels, neighbors)
        # evaluate with repaired data
        repaired.append( train_and_evaluate(train_features, train_repaired_labled, test_features, test_clean_labels) )

    results = [ [i, baseline[i], repaired[i], repaired[i]-baseline[i]] for i in range(len(baseline)) ]
    np.savetxt(
        output_path,
        results,
        delimiter=",",
        fmt="%.3f",
        header="trial, baseline_accuracy, knn_accuracy, difference"
        )

    # helpful plot
    plt.scatter([i for i in range(len(baseline))], baseline, color="blue")
    plt.scatter([i for i in range(len(baseline))], repaired, color="orange")
    plt.show()


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
