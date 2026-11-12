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
    # TODO: Draw flips and return the noisy labels.
    import numpy as np

    flip_probability = np.where(clean_labels==1, POSITIVE_FLIP_PROBABILITY, NEGATIVE_FLIP_PROBABILITY)
    random_draws = rng.random(len(clean_labels))
    should_flip = random_draws < flip_probability

    noisy_labels = np.where(should_flip, -clean_labels, clean_labels)
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
    # TODO: Find neighbors using training features only.
    # TODO: Replace labels using the majority vote of their five neighbors.
    import numpy as np

    #Calculat the Euclidean distance between every pair of training points
    differences = train_features[:, np.newaxis, :] - train_features[np.newaxis, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))

    #Excluding points from its own neighborhood,
    np.fill_diagonal(distances, np.inf)

    #Then do nearest neighbors for each point. A sort keeps the original row order for equal distances, so ties go to the lower row index
    neighbor_indices = np.argsort(distances, axis=1, kind = "stable")[:, :neighbors]

    # Majority vote over the neighbors' original noisy labels (+1 or -1). 
    neighbor_votes = noisy_labels[neighbor_indices].sum(axis=1)
    repaired_labels = np.where(neighbor_votes > 0, 1, -1)
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
    # TODO: Generate independent clean training and test splits.
    # TODO: For each repeat, corrupt training labels and evaluate the baseline.
    # TODO: Repair the same noisy labels with K-NN.
    # TODO: Train a fresh SVM on repaired labels; evaluate on clean test data.
    # TODO: Save per-trial accuracy for all methods to a CSV at output_path.
    # TODO: Report mean accuracy and the change from baseline.
    import csv
    import numpy as np

    rng = np.random.default_rng(seed)

    #Training and test sets, both independent
    train_features, clean_train_labels = generate_clean_split(rng)
    test_features, test_labels = generate_clean_split(rng)

    rows=[]
    for trial in range(1, repeats + 1):
        noisy_train_labels = corrupt_training_labels(clean_train_labels, rng)

        # For part a: Baseline showign SVM trained on the noisy labels
        baseline_accuracy = train_and_evaluate(train_features, noisy_train_labels, test_features, test_labels)

        # For part b: K-NN repair of the same noisy labels, then a fresh SVM
        repaired_train_labels = repair_labels_knn(train_features, noisy_train_labels, neighbors)
        knn_accuracy = train_and_evaluate(train_features, repaired_train_labels, test_features, test_labels)
        rows.append((trial, baseline_accuracy, knn_accuracy))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["trial", "baseline_accuracy", "knn_accuracy"])
        writer.writerows(rows)

    mean_baseline_accuracy = np.mean([row[1] for row in rows])
    mean_knn_accuracy = np.mean([row[2] for row in rows])
    print(f"Mean baseline accuracy: {mean_baseline_accuracy:.4f}")
    print(f"Mean K-NN repair accuracy: {mean_knn_accuracy:.4f}")
    print(f"K-NN - baseline: {mean_knn_accuracy - mean_baseline_accuracy:+.4f}")



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

#AI usage: I used Claude for explaining concepts, optimizing code, bettering my commented explanations
