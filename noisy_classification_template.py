"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import argparse
from pathlib import Path
import numpy as np

from numpy.random import Generator
from numpy.typing import NDArray

Features = NDArray[np.float64]
Labels = NDArray[np.int_]

SVM_C = 1.0
SVM_GAMMA = 0.01
POSITIVE_FLIP_PROBABILITY = 0.35
NEGATIVE_FLIP_PROBABILITY = 0.20


def generate_clean_split(rng: Generator, samples_per_class: int = 1000) -> tuple[Features, Labels]:
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


def corrupt_training_labels(clean_labels: Labels, rng: Generator) -> Labels:
    """Independently flip +1 with probability .35 and -1 with probability .20.

    Select flip probabilities from the ORIGINAL labels. Do not allow a label
    flipped in one direction to be flipped again by the other rule.
    Return a new array, leaving clean_labels unchanged.
    """
    # map clean_labels to an array of probabilities where
    # +1 becomes POSITIVE_FLIP_PROBABILITY and -1 becomes NEGATIVE_FLIP_PROBABILITY
    probs = np.where(clean_labels == 1, POSITIVE_FLIP_PROBABILITY, NEGATIVE_FLIP_PROBABILITY)
    # run random number generator on probs to get an array of booleans
    flips = rng.random(clean_labels.shape) < probs
    # flip the sign for those which gets True 
    return np.where(flips, -clean_labels, clean_labels)


def repair_labels_knn(train_features: Features, noisy_labels: Labels, neighbors: int) -> Labels:
    """Q3(b): propose a K-NN label-repair method; return repaired labels.

    For this simplified assignment, use 5 neighbors, Euclidean distance,
    and replace each label with the majority label of those neighbors.
    Five binary votes cannot tie; break equal-distance neighbor ties consistently.
    Exclude each point itself from its neighborhood. Base all repairs on
    the original noisy labels, so row ordering does not affect the outcome.
    sklearn.neighbors may be used; from-scratch K-NN is not a source requirement.
    """
    repaired = np.empty_like(noisy_labels)

    # iterative pairwisely
    for i, a in enumerate(train_features):
        distances = []
        for j, b in enumerate(train_features):
            if i == j:
                # exclude self
                continue
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            # ignore sqrt as it doesn't affect sorting
            dist = dx ** 2 + dy ** 2
            distances.append((dist, b[0], b[1], j)) # include b[0] and b[1] here for breaking the tie 
        distances.sort()
        nearest = distances[:neighbors]
        # majority vote
        vote = sum(noisy_labels[j] for _, _, _, j in nearest)
        repaired[i] = 1 if vote > 0 else -1
    return repaired


def train_and_evaluate(train_features: Features, train_labels: Labels, test_features: Features, test_labels: Labels) -> float:
    """Train the required RBF SVM and return accuracy on the clean test set."""
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    model = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)
    model.fit(train_features, train_labels)
    return float(accuracy_score(test_labels, model.predict(test_features)))


def run_comparison(repeats: int, seed: int, neighbors: int, output_path: Path) -> None:
    """Q3(a-b), 20 points each: compare the baseline with K-NN label repair.

    Suggested adaptation: generate independent clean training/test sets once,
    then repeat corruption and training (default: 20 noise trials). In each
    trial, use identical data and corrupted labels for both approaches.
    Do not give the repair method the clean training or test labels.
    """
    rng = np.random.default_rng(seed)

    train_features, train_labels = generate_clean_split(rng)
    test_features, test_labels = generate_clean_split(rng)

    # results
    baseline_result = []
    repaired_result = []

    for _ in range(repeats):
        # corrupt
        noisy_labels = corrupt_training_labels(train_labels, rng)
        # evaluate the baseline
        baseline_accuracy = train_and_evaluate(train_features, noisy_labels, test_features, test_labels)
        # repair with knn
        repaired_labels = repair_labels_knn(train_features, noisy_labels, neighbors)
        # evaluate it
        repaired_accuracy = train_and_evaluate(train_features, repaired_labels, test_features, test_labels)
        baseline_result.append(baseline_accuracy)
        repaired_result.append(repaired_accuracy)


    # Save results to csv
    import csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['trial', 'baseline', 'repaired'])
        for i in range(repeats):
            writer.writerow([i, baseline_result[i], repaired_result[i]])
    
    # Report mean and improvement
    baseline_mean = np.mean(baseline_result)
    repaired_mean = np.mean(repaired_result)

    improvement = repaired_mean - baseline_mean
    
    print(f"Baseline mean accuracy: {baseline_mean:.4f}")
    print(f"Repaired mean accuracy: {repaired_mean:.4f}")
    print(f"Change from baseline: {improvement:+.4f}")

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
