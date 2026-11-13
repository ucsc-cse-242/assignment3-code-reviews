import argparse
import csv
from pathlib import Path
import numpy as np

SVM_C = 1.0
SVM_GAMMA = .01
POSITIVE_FLIP_PROBABILITY = .35
NEGATIVE_FLIP_PROBABILITY = .20


def generate_clean_split(rng, samples_per_class=1000):
    a = rng.multivariate_normal((1, 1), np.eye(2), samples_per_class)
    b = rng.multivariate_normal((-1, -1), 3*np.eye(2), samples_per_class)
    x = np.vstack((a, b))
    y = np.r_[np.ones(samples_per_class, dtype=int), -np.ones(samples_per_class, dtype=int)]
    return x, y


def corrupt_training_labels(clean_labels, rng):
    y = np.asarray(clean_labels)
    p = np.where(y == 1, POSITIVE_FLIP_PROBABILITY, NEGATIVE_FLIP_PROBABILITY)
    return np.where(rng.random(len(y)) < p, -y, y)


def repair_labels_knn(train_features, noisy_labels, neighbors):
    x, y = np.asarray(train_features), np.asarray(noisy_labels)
    if neighbors != 5 or len(x) <= neighbors:
        raise ValueError('need five other neighbors')
    d = ((x[:, None]-x[None, :])**2).sum(axis=2).astype(float)
    np.fill_diagonal(d, np.inf)
    i = np.argsort(d, axis=1, kind='stable')[:, :neighbors]
    return np.where(y[i].sum(axis=1) > 0, 1, -1)


def train_and_evaluate(train_features, train_labels, test_features, test_labels):
    from sklearn.svm import SVC
    m = SVC(kernel='rbf', C=SVM_C, gamma=SVM_GAMMA)
    m.fit(train_features, train_labels)
    return float(m.score(test_features, test_labels))


def run_comparison(repeats, seed, neighbors, output_path):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    rng = np.random.default_rng(seed)
    x, y = generate_clean_split(rng)
    tx, ty = generate_clean_split(rng)
    rows = []
    for i in range(repeats):
        z = corrupt_training_labels(y, rng)
        b = train_and_evaluate(x, z, tx, ty)
        r = repair_labels_knn(x, z, neighbors)
        k = train_and_evaluate(x, r, tx, ty)
        rows.append((i+1, b, k))
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['trial', 'baseline_accuracy', 'knn_accuracy'])
        w.writerows(rows)
    a = np.array(rows)
    b, k = a[:, 1:].mean(axis=0)
    print('Baseline:', b, 'KNN:', k, 'difference:', k-b)
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--neighbors', type=int, default=5, choices=(5,))
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results/noise_comparison.csv')
    a = p.parse_args()
    run_comparison(a.repeats, a.seed, a.neighbors, a.output)


if __name__ == '__main__':
    main()
