"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

# plain (non-relative) import so the file runs as a script:
#   python noisy_classification.py
from utils import _euclidean_distance


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


def corrupt_training_labels(clean_labels: np.ndarray, rng)->np.ndarray:
    """Independently flip +1 with probability .35 and -1 with probability .20.

    Select flip probabilities from the original labels. A label flipped in
    one direction cannot be flipped again by the other rule.
    Returns a new array and leaves clean_labels unchanged.

    Inputs:
        @param clean_labels: numpy ndarray representing original labels
        @param rng: numpy.random.default_rng
    Output:
        - a new array with corrupted labels
    """
    # flip probability for each example, chosen from its original label
    # ref: https://numpy.org/doc/2.5/reference/generated/numpy.where.html
    flip_probability = np.where(
        clean_labels == 1, POSITIVE_FLIP_PROBABILITY, NEGATIVE_FLIP_PROBABILITY
    )
    # one independent draw per example -> each label is flipped at most once
    flips = rng.random(clean_labels.shape[0]) < flip_probability

    # create a new array containing the flipped labels
    corrupted_labels = np.where(flips, -clean_labels, clean_labels)
    return corrupted_labels


def repair_labels_knn(
    train_features: np.ndarray, noisy_labels: np.ndarray, neighbors: int = 5
) -> np.ndarray:
    """repair labels for each point based on each point's k nearest neighbors

    Inputs:
        @param train_features: ndarray of features
        @param noisy_labels: ndarray of labels that contain noise
        @param neighbors: integer stating how many neighbors to use
    Output:
        - new array containing new labels after repair

    Implementation Notes:
        - implemented using numpy and approach from lecture/textbook, but
          an alternate approach can use sklearn.neighbors.KNeighborsClassifier
        - uses euclidean distance and does not allow other distance metrics
        - equal-dstance neighbors go to lower row index
        - in case neighbor vote ends in tie, defaults to original index
        - vote uses original labels, so row order does not affect result
    """
    # distances[j, i] = euclidean distance between point j and point i, (n, n)
    n = train_features.shape[0]
    distances = np.empty((n, n))
    for i in range(n):
        distances[:, i] = _euclidean_distance(train_features, train_features[i])

    # exclude each point from its own neighborhood (distance to itself is 0)
    np.fill_diagonal(distances, np.inf)

    # sort distance to other instances and get k closest neighbors
    # kind="stable" -> equal-distance ties go to the lower row index
    # ref: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
    k_neighbors = np.argsort(distances, axis=1, kind="stable")[:, :neighbors]

    # run voting algorithm on neighbors:
    #   vote_label = sum of y[n] over the k neighbors; return sign(vote_label)
    vote_label = noisy_labels[k_neighbors].sum(axis=1)
    majority_vote = np.sign(vote_label).astype(int)

    # if number of neighbors is even and neighbor vote is tie: use original label
    majority_vote[vote_label == 0] = noisy_labels[vote_label == 0]

    return majority_vote


def train_and_evaluate(train_features, train_labels, test_features, test_labels):
    """Train the required RBF SVM and return accuracy on the clean test set."""
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    model = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)
    model.fit(train_features, train_labels)
    return float(accuracy_score(test_labels, model.predict(test_features)))


def run_comparison(
    repeats: int, seed: int, neighbors: int, output_path: Path
) -> list:
    """compare the baseline with K-NN label repair over a set of experiements

    Training and test sets are generated at the start and then corruption and
    retraining are repeated over a set of trials.

    Inputs:
        @param repeats: number of experiments to run
        @param seed: integer seed to pass into default rng
        @param neighbors: integer number of neighbors used in KNN repair
        @param output_path: path to save the results as csv
    Outputs:
        - list of accuracy scores from each experiment
    """
    # generate independent clean training and test splits ONCE (fixed for
    # every trial); one seeded rng makes the whole run reproducible
    rng = np.random.default_rng(seed=seed)
    train_features, train_labels = generate_clean_split(rng, 1000)
    test_features, test_labels = generate_clean_split(rng, 1000)

    accuracy_scores = []
    for trial in range(1, repeats + 1):
        # Q3(a) baseline: corrupt training labels, train SVM, test on clean labels
        noisy_labels = corrupt_training_labels(train_labels, rng)
        baseline_accuracy = train_and_evaluate(
            train_features, noisy_labels, test_features, test_labels
        )
        # repair the noisy labels with K-NN, then train a fresh SVM
        repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)
        knn_accuracy = train_and_evaluate(
            train_features, repaired_labels, test_features, test_labels
        )
        accuracy_scores.append(
            {
                "trial": trial,
                "baseline_accuracy": baseline_accuracy,
                "knn_accuracy": knn_accuracy,
            }
        )
        print(f"trial {trial:2d}: baseline {baseline_accuracy:.4f}, "
              f"knn {knn_accuracy:.4f}")

    # save per-trial accuracy for both methods to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["trial", "baseline_accuracy", "knn_accuracy"]
        )
        writer.writeheader()
        writer.writerows(accuracy_scores)
    print(f"Saved {output_path}")

    # report mean accuracy and the change from baseline
    baseline_mean = np.mean([row["baseline_accuracy"] for row in accuracy_scores])
    knn_mean = np.mean([row["knn_accuracy"] for row in accuracy_scores])
    print(f"Mean baseline accuracy: {baseline_mean:.4f}")
    print(f"Mean K-NN repair accuracy: {knn_mean:.4f}")
    print(f"Difference (K-NN - baseline): {knn_mean - baseline_mean:+.4f}")
    return accuracy_scores


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
