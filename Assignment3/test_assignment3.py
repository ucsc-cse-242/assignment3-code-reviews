import csv
import inspect
import math

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

import assignment3_template as q1
import kmeans_template as q2
import noisy_classification_template as q3


class TestHardMarginSVM:
    def test_support_vectors(self):
        assert sorted(q1.identify_support_vectors()) == [1, 5, 6]

    def test_boundary(self):
        weights, bias, _ = q1.derive_svm_separator()
        assert weights == pytest.approx((2.0, 2.0))
        assert bias == pytest.approx(-3.0)

    def test_geometric_margin(self):
        _, _, margin = q1.derive_svm_separator()
        assert margin == pytest.approx(1 / (2 * math.sqrt(2)))

    def test_constraints_hold_and_are_tight_only_on_support_vectors(self):
        weights, bias, _ = q1.derive_svm_separator()
        support_ids = set(q1.identify_support_vectors())
        for point_id, (x, y) in enumerate(zip(q1.SVM_X, q1.SVM_Y), start=1):
            functional_margin = y * (np.dot(weights, x) + bias)
            if point_id in support_ids:
                assert functional_margin == pytest.approx(1.0)
            else:
                assert functional_margin > 1.0

    def test_margin_is_distance_to_closest_points(self):
        weights, bias, margin = q1.derive_svm_separator()
        distances = [abs(np.dot(weights, x) + bias) / np.linalg.norm(weights)
                     for x in q1.SVM_X]
        assert min(distances) == pytest.approx(margin)

    def test_matches_library_hard_margin_svm(self):
        model = SVC(kernel="linear", C=1e10).fit(q1.SVM_X, q1.SVM_Y)
        weights, bias, _ = q1.derive_svm_separator()
        assert model.coef_[0] == pytest.approx(weights, abs=1e-4)
        assert model.intercept_[0] == pytest.approx(bias, abs=1e-4)
        assert sorted(model.support_ + 1) == sorted(q1.identify_support_vectors())

    def test_plot_is_saved(self, tmp_path):
        path = q1.plot_svm_data(tmp_path / "svm.png")
        assert path.exists() and path.stat().st_size > 0


class TestKMeansSteps:
    def test_assigns_to_nearest_centroid(self):
        data = np.array([[0.0, 0.0], [0.1, 0.2], [5.0, 5.0], [4.8, 5.3]])
        centroids = np.array([[5.0, 5.0], [0.0, 0.0]])
        assert q2.update_assignments(data, centroids).tolist() == [1, 1, 0, 0]

    def test_assignment_shape_and_integer_type(self):
        rng = np.random.default_rng(0)
        assignments = q2.update_assignments(rng.normal(size=(50, 3)),
                                            rng.normal(size=(4, 3)))
        assert assignments.shape == (50,)
        assert np.issubdtype(assignments.dtype, np.integer)

    def test_distance_tie_goes_to_lowest_index(self):
        data = np.array([[0.0, 0.0]])
        centroids = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        assert q2.update_assignments(data, centroids).tolist() == [0]

    def test_duplicate_centroids_use_lowest_index(self):
        data = np.array([[0.0, 0.0], [3.0, 3.0]])
        centroids = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert q2.update_assignments(data, centroids).tolist() == [0, 0]

    def test_matches_brute_force_nearest_centroid(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(200, 2))
        centroids = rng.normal(size=(5, 2))
        expected = [min(range(5), key=lambda j: np.linalg.norm(x - centroids[j]))
                    for x in data]
        assert q2.update_assignments(data, centroids).tolist() == expected

    def test_centroids_are_cluster_means(self):
        data = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 14.0]])
        centroids = np.zeros((2, 2))
        updated = q2.update_centroids(data, centroids, np.array([0, 0, 1, 1]))
        assert updated == pytest.approx(np.array([[1.0, 0.0], [11.0, 12.0]]))

    def test_empty_cluster_keeps_previous_centroid(self):
        data = np.array([[0.0, 0.0], [2.0, 2.0]])
        centroids = np.array([[0.0, 0.0], [7.0, -3.0]])
        updated = q2.update_centroids(data, centroids, np.array([0, 0]))
        assert updated[0] == pytest.approx([1.0, 1.0])
        assert updated[1] == pytest.approx([7.0, -3.0])

    def test_does_not_modify_input_centroids(self):
        data = np.array([[0.0, 0.0], [2.0, 2.0]])
        centroids = np.array([[5.0, 5.0], [6.0, 6.0]])
        original = centroids.copy()
        q2.update_centroids(data, centroids, np.array([0, 1]))
        assert np.array_equal(centroids, original)

    def test_handles_duplicate_initial_centroids(self):
        data, _ = q2.make_toy_data()
        initial = np.repeat(data[:1], 3, axis=0)
        centroids, assignments = q2.kmeans(data, initial, max_iterations=20)
        assert np.all(np.isfinite(centroids))
        assert assignments.shape == (len(data),)


class TestKMeansExperiment:
    def test_toy_data_configuration(self):
        data, labels = q2.make_toy_data()
        expected, _ = make_blobs(n_samples=4000, centers=4, cluster_std=0.60,
                                 random_state=0)
        assert np.array_equal(data, expected)
        assert len(np.unique(labels)) == 4

    def test_k4_recovers_generating_groups(self):
        data, true_labels = q2.make_toy_data()
        rng = np.random.default_rng(0)
        for k in (2, 3):
            q2.get_initial_clusters(k, data, rng)
        initial = q2.get_initial_clusters(4, data, rng)
        _, assignments = q2.kmeans(data, initial, max_iterations=100)
        assert adjusted_rand_score(true_labels, assignments) > 0.99

    def test_converged_centroids_are_fixed_point(self):
        data, _ = q2.make_toy_data()
        initial = q2.get_initial_clusters(4, data, np.random.default_rng(3))
        centroids, assignments = q2.kmeans(data, initial, max_iterations=100)
        assert q2.update_centroids(data, centroids, assignments) == pytest.approx(centroids)

    def test_run_experiments_saves_one_plot_per_k(self, tmp_path):
        q2.run_experiments(tmp_path, seed=0, max_iterations=20)
        for k in q2.K_VALUES:
            path = tmp_path / f"kmeans_k{k}.png"
            assert path.exists() and path.stat().st_size > 0


class TestLabelCorruption:
    def test_does_not_modify_clean_labels(self):
        labels = np.array([1, 1, -1, -1] * 50)
        original = labels.copy()
        q3.corrupt_training_labels(labels, np.random.default_rng(0))
        assert np.array_equal(labels, original)

    def test_output_is_plus_or_minus_one(self):
        labels = np.array([1, -1] * 100)
        noisy = q3.corrupt_training_labels(labels, np.random.default_rng(0))
        assert set(np.unique(noisy)) <= {-1, 1}
        assert noisy.shape == labels.shape

    def test_flip_rates_match_class_probabilities(self):
        n = 200_000
        labels = np.concatenate((np.ones(n, dtype=int), -np.ones(n, dtype=int)))
        noisy = q3.corrupt_training_labels(labels, np.random.default_rng(0))
        positive_rate = np.mean(noisy[:n] != labels[:n])
        negative_rate = np.mean(noisy[n:] != labels[n:])
        assert positive_rate == pytest.approx(0.35, abs=0.005)
        assert negative_rate == pytest.approx(0.20, abs=0.005)

    def test_reproducible_with_seed(self):
        labels = np.array([1, -1] * 100)
        first = q3.corrupt_training_labels(labels, np.random.default_rng(42))
        second = q3.corrupt_training_labels(labels, np.random.default_rng(42))
        assert np.array_equal(first, second)


class TestKnnRepair:
    def test_signature_has_no_access_to_clean_or_test_labels(self):
        params = list(inspect.signature(q3.repair_labels_knn).parameters)
        assert params == ["train_features", "noisy_labels", "neighbors"]

    def test_isolated_mislabel_is_corrected(self):
        features = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [-0.1, 0.0],
                             [0.0, -0.1], [0.1, 0.1], [9.0, 9.0], [9.1, 9.0],
                             [9.0, 9.1], [8.9, 9.0], [9.0, 8.9], [9.1, 9.1]])
        noisy = np.array([-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1])
        repaired = q3.repair_labels_knn(features, noisy, 5)
        assert repaired.tolist() == [1] * 6 + [-1] * 6

    def test_point_is_excluded_from_its_own_neighborhood(self):
        features = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        noisy = np.array([1, -1, -1, 1, 1, -1])
        repaired = q3.repair_labels_knn(features, noisy, 5)
        assert repaired[0] == -1
        assert repaired[5] == 1

    def test_repairs_use_original_noisy_labels(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(300, 2))
        noisy = rng.choice([-1, 1], size=300)
        repaired = q3.repair_labels_knn(features, noisy, 5)

        nearest = NearestNeighbors(n_neighbors=6).fit(features).kneighbors(
            features, return_distance=False)[:, 1:]
        expected = np.where(noisy[nearest].sum(axis=1) > 0, 1, -1)
        assert np.array_equal(repaired, expected)

    def test_row_order_does_not_change_result(self):
        rng = np.random.default_rng(1)
        features = rng.normal(size=(200, 2))
        noisy = rng.choice([-1, 1], size=200)
        order = rng.permutation(200)
        repaired = q3.repair_labels_knn(features, noisy, 5)
        shuffled = q3.repair_labels_knn(features[order], noisy[order], 5)
        assert np.array_equal(repaired[order], shuffled)

    def test_does_not_modify_noisy_labels(self):
        rng = np.random.default_rng(2)
        features = rng.normal(size=(50, 2))
        noisy = rng.choice([-1, 1], size=50)
        original = noisy.copy()
        q3.repair_labels_knn(features, noisy, 5)
        assert np.array_equal(noisy, original)

    def test_equal_distance_ties_prefer_lower_index(self):
        features = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0],
                             [0.0, -1.0], [1.0, 1.0], [-1.0, -1.0]])
        noisy = np.array([1, 1, 1, -1, -1, 1, -1])
        assert q3.repair_labels_knn(features, noisy, 5)[0] == 1


class TestNoiseComparison:
    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("q3") / "comparison.csv"
        q3.run_comparison(repeats=20, seed=0, neighbors=5, output_path=path)
        with open(path, newline="") as f:
            return path, list(csv.DictReader(f))

    def test_csv_columns_and_rows(self, results):
        path, rows = results
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        assert header == ["trial", "baseline_accuracy", "knn_accuracy"]
        assert [int(row["trial"]) for row in rows] == list(range(1, 21))

    def test_accuracies_are_valid(self, results):
        _, rows = results
        for row in rows:
            assert 0.5 < float(row["baseline_accuracy"]) <= 1.0
            assert 0.5 < float(row["knn_accuracy"]) <= 1.0

    def test_knn_repair_improves_mean_accuracy(self, results):
        _, rows = results
        baseline = np.mean([float(row["baseline_accuracy"]) for row in rows])
        knn = np.mean([float(row["knn_accuracy"]) for row in rows])
        assert knn > baseline

    def test_reproducible_with_seed(self, results, tmp_path):
        _, rows = results
        path = tmp_path / "rerun.csv"
        q3.run_comparison(repeats=20, seed=0, neighbors=5, output_path=path)
        with open(path, newline="") as f:
            assert list(csv.DictReader(f)) == rows

    def test_clean_split_matches_specification(self):
        features, labels = q3.generate_clean_split(np.random.default_rng(0),
                                                   samples_per_class=50_000)
        positive, negative = features[labels == 1], features[labels == -1]
        assert len(positive) == len(negative) == 50_000
        assert positive.mean(axis=0) == pytest.approx([1, 1], abs=0.03)
        assert negative.mean(axis=0) == pytest.approx([-1, -1], abs=0.03)
        assert np.cov(positive.T) == pytest.approx(np.eye(2), abs=0.05)
        assert np.cov(negative.T) == pytest.approx(3 * np.eye(2), abs=0.1)
