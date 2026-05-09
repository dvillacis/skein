"""Smoke + regression tests for multinomial classifiers (M3.6)."""

from __future__ import annotations

import numpy as np
import pytest

skein_glm = pytest.importorskip("skein_glm")


def _make_problem(seed: int = 0, n: int = 120, p: int = 5, k: int = 3):
    """Build a 3-class problem with clear signal at features 0 and 2."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    true_b = np.zeros((p, k))
    true_b[0] = [1.5, -1.5, 0.0]
    true_b[2] = [-0.7, -0.7, 1.4]
    eta = x @ true_b + 0.1 * rng.standard_normal((n, k))
    labels = np.argmax(eta, axis=1)
    return x, labels, true_b


def test_multinomial_lasso_classifier_predict_shapes_and_semantics():
    x, labels, _ = _make_problem(0)
    clf = skein_glm.MultinomialLassoClassifier(lambda_=0.05).fit(x, labels)
    assert clf.coef_.shape == (3, 5)
    assert clf.intercept_.shape == (3,)
    np.testing.assert_array_equal(clf.classes_, np.array([0, 1, 2]))

    eta = clf.decision_function(x)
    assert eta.shape == (x.shape[0], 3)

    proba = clf.predict_proba(x)
    assert proba.shape == eta.shape
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(proba >= 0.0)

    preds = clf.predict(x)
    assert preds.shape == (x.shape[0],)
    assert set(preds.tolist()).issubset({0, 1, 2})


def test_multinomial_lasso_classifier_recovers_signal():
    x, labels, _ = _make_problem(1, n=200)
    clf = skein_glm.MultinomialLassoClassifier(lambda_=0.005, max_outer=30).fit(x, labels)
    # Train accuracy should be way above the 33% chance level.
    assert (clf.predict(x) == labels).mean() > 0.7
    # Active-feature row-norms: rows 0 and 2 should be largest.
    row_norms = np.linalg.norm(clf.coef_, axis=0)  # (p,)
    top_two = np.argsort(row_norms)[-2:]
    assert set(top_two.tolist()) == {0, 2}


def test_multinomial_lasso_path_classifier_shapes_and_lambda_order():
    x, labels, _ = _make_problem(2)
    path = skein_glm.MultinomialLassoPathClassifier(
        n_lambdas=15, lambda_min_ratio=1e-2
    ).fit(x, labels)
    assert path.coefs_.shape == (15, 3, 5)
    assert path.intercepts_.shape == (15, 3)
    assert path.lambdas_.shape == (15,)
    # λ decreasing.
    assert all(
        path.lambdas_[i] > path.lambdas_[i + 1] for i in range(len(path.lambdas_) - 1)
    )
    # First λ (≈ λ_max) should drive everything close to zero (the
    # prox-Newton outer loop terminates at outer_tol, leaving sub-1e-3
    # residuals before the inner CD's strong-rule kicks coefficients
    # exactly to zero). Loose tolerance is enough to confirm "no active
    # features at λ_max".
    np.testing.assert_allclose(path.coefs_[0], 0.0, atol=1e-3)


def test_multinomial_mcp_path_recovers_signal_at_smallest_lambda():
    x, labels, _ = _make_problem(3, n=200)
    path = skein_glm.MultinomialMCPPathClassifier(
        gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-3, max_outer=30
    ).fit(x, labels)
    last_b = path.coefs_[-1]  # (K, p)
    row_norms = np.linalg.norm(last_b, axis=0)
    top_two = np.argsort(row_norms)[-2:]
    assert set(top_two.tolist()) == {0, 2}


def test_multinomial_lasso_dense_sparse_equivalence():
    pytest.importorskip("scipy")
    from scipy import sparse

    x, labels, _ = _make_problem(4, n=80, p=6)
    x_csc = sparse.csc_matrix(x)
    # Use a shared explicit λ-grid; the auto-grid is computed slightly
    # differently between dense and sparse code paths.
    path_d = skein_glm.MultinomialLassoPathClassifier(
        n_lambdas=10, lambda_min_ratio=1e-2
    ).fit(x, labels)
    path_s = skein_glm.MultinomialLassoPathClassifier(
        lambdas=path_d.lambdas_,
    ).fit(x_csc, labels)
    np.testing.assert_allclose(path_d.coefs_, path_s.coefs_, atol=1e-6)
    np.testing.assert_allclose(path_d.intercepts_, path_s.intercepts_, atol=1e-6)


def test_multinomial_lasso_dense_sparse_equivalence_with_standardize():
    pytest.importorskip("scipy")
    from scipy import sparse

    x, labels, _ = _make_problem(5, n=80, p=6)
    # Inflate one column so standardization matters.
    x[:, 1] *= 50.0
    x_csc = sparse.csc_matrix(x)
    path_d = skein_glm.MultinomialLassoPathClassifier(
        n_lambdas=10, lambda_min_ratio=1e-2, standardize=True
    ).fit(x, labels)
    path_s = skein_glm.MultinomialLassoPathClassifier(
        lambdas=path_d.lambdas_, standardize=True
    ).fit(x_csc, labels)
    np.testing.assert_allclose(path_d.coefs_, path_s.coefs_, atol=1e-6)
    np.testing.assert_allclose(path_d.intercepts_, path_s.intercepts_, atol=1e-6)


def test_multinomial_lasso_path_cv_picks_active_features():
    x, labels, _ = _make_problem(6, n=180)
    cv = skein_glm.MultinomialLassoPathCV(
        cv=3, random_state=0, n_lambdas=15, lambda_min_ratio=1e-3
    ).fit(x, labels)
    assert cv.coef_.shape == (3, 5)
    assert cv.intercept_.shape == (3,)
    row_norms = np.linalg.norm(cv.coef_, axis=0)
    top_two = np.argsort(row_norms)[-2:]
    assert set(top_two.tolist()) == {0, 2}
    # Sanity: train accuracy clearly above chance.
    assert (cv.predict(x) == labels).mean() > 0.7


def test_multinomial_select_by_ic_runs_for_all_three_criteria():
    x, labels, _ = _make_problem(7, n=120)
    path = skein_glm.MultinomialLassoPathClassifier(
        n_lambdas=15, lambda_min_ratio=1e-3
    ).fit(x, labels)
    for crit in ("aic", "bic", "ebic"):
        best_idx, scores = skein_glm.select_by_ic(path, x, labels, criterion=crit)
        assert 0 <= best_idx < len(path.lambdas_)
        assert scores.shape == path.lambdas_.shape
        assert np.all(np.isfinite(scores))


def test_multinomial_classifier_rejects_single_class_y():
    x = np.zeros((10, 3))
    y = np.zeros(10, dtype=int)  # only one class
    with pytest.raises(ValueError, match="≥ 2 distinct classes"):
        skein_glm.MultinomialLassoClassifier().fit(x, y)


def test_multinomial_scad_rejects_a_below_two():
    x, labels, _ = _make_problem(8, n=40, p=3)
    with pytest.raises(ValueError, match="must be > 2"):
        skein_glm.MultinomialSCADClassifier(a=1.5).fit(x, labels)


def test_multinomial_elastic_net_rejects_alpha_out_of_range():
    x, labels, _ = _make_problem(9, n=40, p=3)
    with pytest.raises(ValueError, match=r"alpha must be in"):
        skein_glm.MultinomialElasticNetClassifier(alpha=1.5).fit(x, labels)


def test_multinomial_elastic_net_alpha_one_matches_lasso_on_shared_grid():
    x, labels, _ = _make_problem(10, n=80, p=5)
    path_lasso = skein_glm.MultinomialLassoPathClassifier(
        n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, labels)
    path_en = skein_glm.MultinomialElasticNetPathClassifier(
        alpha=1.0, lambdas=path_lasso.lambdas_
    ).fit(x, labels)
    # EN with α=1 is exactly the row-grouped lasso.
    np.testing.assert_allclose(path_en.coefs_, path_lasso.coefs_, atol=1e-7)
    np.testing.assert_allclose(path_en.intercepts_, path_lasso.intercepts_, atol=1e-7)


def test_multinomial_string_labels_round_trip():
    x, codes, _ = _make_problem(11, n=60)
    label_map = np.array(["cat", "dog", "fish"])
    labels = label_map[codes]
    clf = skein_glm.MultinomialLassoClassifier(lambda_=0.05).fit(x, labels)
    # classes_ should be sorted unique strings.
    np.testing.assert_array_equal(clf.classes_, np.array(["cat", "dog", "fish"]))
    preds = clf.predict(x)
    assert preds.dtype.kind == "U"
    assert set(preds.tolist()).issubset({"cat", "dog", "fish"})
