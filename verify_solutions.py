"""Focused checks for SVM geometry, K-means edge cases, and label-repair rules."""
import numpy as np
from sklearn.svm import SVC
from assignment3_template import SVM_X, SVM_Y, identify_support_vectors, derive_svm_separator
from kmeans_template import update_assignments, update_centroids, kmeans
from noisy_classification_template import corrupt_training_labels, repair_labels_knn


def main():
    x, y = np.array(SVM_X, dtype=float), np.array(SVM_Y)
    w, b, margin = derive_svm_separator()
    functional = y * (x @ w + b)
    np.testing.assert_array_equal(functional, [1, 3, 3, 3, 1, 1])
    assert tuple(np.flatnonzero(np.isclose(functional, 1)) + 1) == identify_support_vectors()
    np.testing.assert_allclose(margin, 1 / np.linalg.norm(w))
    independent = SVC(kernel='linear', C=1e6, tol=1e-10).fit(x, y)
    np.testing.assert_allclose(independent.coef_[0], w, atol=1e-8)
    np.testing.assert_allclose(independent.intercept_[0], b, atol=1e-8)

    data = np.array([[0., 0.], [2., 0.], [1., 0.], [10., 0.]])
    centers = np.array([[0., 0.], [2., 0.], [99., 0.]])
    before = centers.copy()
    labels = update_assignments(data, centers)
    np.testing.assert_array_equal(labels, [0, 1, 0, 1])  # Point 2 ties.
    updated = update_centroids(data, centers, labels)
    np.testing.assert_allclose(updated, [[.5, 0], [6., 0], [99., 0]])
    np.testing.assert_array_equal(centers, before)
    duplicate_centers = np.array([[0., 0.], [0., 0.]])
    final, assignment = kmeans(data, duplicate_centers, 10)
    assert np.isfinite(final).all() and np.isfinite(assignment).all()
    rng = np.random.default_rng(15)
    cloud = rng.normal(size=(60, 2)); c = cloud[:4].copy(); old = np.inf
    for _ in range(12):
        a = update_assignments(cloud, c)
        objective = np.sum((cloud - c[a]) ** 2)
        assert objective <= old + 1e-10
        c = update_centroids(cloud, c, a)
        assert np.sum((cloud - c[a]) ** 2) <= objective + 1e-10
        old = objective

    class FixedDraws:
        def random(self, shape):
            assert shape == (4,)
            return np.array([.34, .35, .19, .20])
    clean = np.array([1, 1, -1, -1]); original = clean.copy()
    np.testing.assert_array_equal(corrupt_training_labels(clean, FixedDraws()), [-1, 1, 1, -1])
    np.testing.assert_array_equal(clean, original)
    # Duplicates and equal distances exercise self exclusion and stable tie breaking.
    points = np.array([[0, 0], [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [3, 2]], float)
    noisy = np.array([1, -1, 1, -1, 1, -1, 1]); noisy_copy = noisy.copy()
    expected = []
    for i, point in enumerate(points):
        order = sorted((np.linalg.norm(point-other), j)
                       for j, other in enumerate(points) if j != i)
        expected.append(1 if sum(noisy[j] for _, j in order[:5]) > 0 else -1)
    np.testing.assert_array_equal(repair_labels_knn(points, noisy, 5), expected)
    np.testing.assert_array_equal(noisy, noisy_copy)
    # With no distance ties, a row permutation cannot change simultaneous repairs.
    points = rng.normal(size=(17, 2)); noisy = rng.choice([-1, 1], size=17)
    permutation = rng.permutation(17)
    repaired = repair_labels_knn(points, noisy, 5)
    np.testing.assert_array_equal(repair_labels_knn(points[permutation], noisy[permutation], 5),
                                  repaired[permutation])
    print('PASS: SVM geometry/library cross-check; K-means ties, empties, nonmutation, descent;')
    print('      original-label corruption; K-NN self exclusion, ties, simultaneous voting, permutation.')


if __name__ == '__main__':
    main()
