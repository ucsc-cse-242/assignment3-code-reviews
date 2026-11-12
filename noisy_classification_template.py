"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import argparse
from pathlib import Path

import os
import numpy as np
from copy import copy

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

    Select flip probabilities from the ORIGINAL labels. Labels are flipped a single time independently
    Returns a new array, of noisy labels of shape (n,)
    """
    # Draw flips and return the noisy labels.
    noisy_labels = copy(clean_labels)

    pos_flip_rate = POSITIVE_FLIP_PROBABILITY
    neg_flip_rate = NEGATIVE_FLIP_PROBABILITY

    pos_idx = np.where(clean_labels == 1)[0]
    neg_idx = np.where(clean_labels == -1)[0] 

    rand_values = rng.random(clean_labels.size)  

    pos_flip = pos_idx[rand_values[pos_idx] < pos_flip_rate]
    neg_flip = neg_idx[rand_values[neg_idx] < neg_flip_rate]

    noisy_labels[pos_flip] = -1
    noisy_labels[neg_flip] = 1

    return noisy_labels


def repair_labels_knn(train_features, noisy_labels, neighbors):
    """
    Using the top 5 neighbors via Euclidean distance, this function replaces each label with the majority label 
    of those neighbors. The majority vote is +1 if the sum of votes > 0 if the sum of votes is < 0 the vote is -1. 

    Returns the repaired labels (n,)
    """
    # Find neighbors using training features only.
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=neighbors).fit(train_features)

    neighbor_idx = nn.kneighbors(return_distance=False)
    votes = noisy_labels[neighbor_idx].sum(axis=1)
    return np.where(votes > 0, 1, -1)


def train_and_evaluate(train_features, train_labels, test_features, test_labels):
    """Train the required RBF SVM and return accuracy on the clean test set."""
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    model = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)
    model.fit(train_features, train_labels)
    return float(accuracy_score(test_labels, model.predict(test_features)))


def run_comparison(repeats, seed, neighbors, output_path):
    """Compares the baseline with K-NN label repair.
    This function generates an independent clean training/test sets once,
    then repeat corruption and training (default: 20 noise trials). In each
    trial, it uses identical data and corrupted labels for both approaches.

    Saves the results to a csv file
    """
    rng = np.random.default_rng(seed=seed)

    # Generate independent clean training and test splits once.
    X_train, y_train = generate_clean_split(rng, samples_per_class=1000)
    X_test, y_test = generate_clean_split(rng, samples_per_class=1000)

    baseline_accs = []
    knn_accs = []
    for _ in range(repeats):
        # For each repeat, corrupt training labels and evaluate the baseline.
        y_train_noisy = corrupt_training_labels(y_train, rng)
        baseline_acc = train_and_evaluate(X_train, y_train_noisy, X_test, y_test)

        # Repair the same noisy labels with K-NN, then train a fresh SVM.
        y_train_repaired = repair_labels_knn(X_train, y_train_noisy, neighbors=neighbors)
        knn_acc = train_and_evaluate(X_train, y_train_repaired, X_test, y_test)

        baseline_accs.append(baseline_acc)
        knn_accs.append(knn_acc)

    # Report mean accuracy and the change from baseline.
    baseline_accs = np.array(baseline_accs)
    knn_accs = np.array(knn_accs)

    print(f"Baseline (noisy) acc: {baseline_accs.mean()*100:.3f}% +/- {baseline_accs.std()*100:.3f}%\n"
          f"K-NN repaired acc:    {knn_accs.mean()*100:.3f}% +/- {knn_accs.std()*100:.3f}%\n"
          f"Change (K-NN - baseline): {(knn_accs.mean() - baseline_accs.mean())*100:+.3f}%")

    # Save per-trial accuracy for all methods to a CSV at output_path.
    import pandas as pd
    df = pd.DataFrame({"trial": np.arange(repeats),
                       "baseline_accuracy": baseline_accs,
                       "knn_accuracy": knn_accs})
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=5,
                        choices=(5,), help="Use five neighbors for this assignment.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "noise_comparison.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_comparison(args.repeats, args.seed, args.neighbors, args.output)


if __name__ == "__main__":
    main()
