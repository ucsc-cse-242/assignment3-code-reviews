"""Supplementary analysis for Question 3: why K-NN repair changes SVM accuracy.

Re-creates the same splits and noisy labels as noisy_classification_template.py
(same seed and RNG call order) and reports label error rates and prediction
balance. Clean training labels are used here only to measure label quality,
never by the repair method.
"""

import argparse

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.svm import SVC

import noisy_classification_template as q3


def bayes_accuracy(features, labels):
    positive = multivariate_normal((1, 1), np.eye(2)).pdf(features)
    negative = multivariate_normal((-1, -1), 3 * np.eye(2)).pdf(features)
    predictions = np.where(positive > negative, 1, -1)
    return float(np.mean(predictions == labels))


def fit_predict(train_features, train_labels, test_features):
    model = SVC(kernel="rbf", C=q3.SVM_C, gamma=q3.SVM_GAMMA)
    return model.fit(train_features, train_labels).predict(test_features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    train_features, clean_labels = q3.generate_clean_split(rng)
    test_features, test_labels = q3.generate_clean_split(rng)

    clean_predictions = fit_predict(train_features, clean_labels, test_features)
    print(f"Bayes-optimal test accuracy:    {bayes_accuracy(test_features, test_labels):.4f}")
    print(f"SVM trained on clean labels:    {np.mean(clean_predictions == test_labels):.4f}")

    stats = {"noisy": [], "repaired": []}
    for _ in range(args.repeats):
        noisy_labels = q3.corrupt_training_labels(clean_labels, rng)
        repaired_labels = q3.repair_labels_knn(train_features, noisy_labels, 5)
        for name, labels in (("noisy", noisy_labels), ("repaired", repaired_labels)):
            predictions = fit_predict(train_features, labels, test_features)
            stats[name].append((
                np.mean(labels != clean_labels),
                np.mean(labels[clean_labels == 1] != 1),
                np.mean(labels[clean_labels == -1] != -1),
                np.mean(predictions == 1),
                np.mean(predictions[test_labels == 1] != 1),
                np.mean(predictions[test_labels == -1] != -1),
            ))

    print(f"\nAverages over {args.repeats} trials:")
    header = ("training labels", "label error", "positive error", "negative error",
              "predicted positive", "missed positives", "false positives")
    print(" | ".join(header))
    for name, rows in stats.items():
        values = " | ".join(f"{value:.4f}" for value in np.mean(rows, axis=0))
        print(f"{name} | {values}")


if __name__ == "__main__":
    main()
