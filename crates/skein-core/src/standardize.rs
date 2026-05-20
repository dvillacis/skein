//! Standardization + intercept handling.
//!
//! Convention matches `glmnet` / `ncvreg`:
//! - `x̄_j = (1/n) Σ_i X_ij`
//! - `scale_j = sqrt((1/n) Σ_i (X_ij - x̄_j)²)` (population std, always
//!   relative to the column mean — even when `center_x` is off).
//! - `ȳ = (1/n) Σ_i y_i`
//! - Intercept recovered as `α = ȳ - Σ_j β_j · x̄_j`.
//!
//! Zero-variance columns (`scale_j == 0`) are passed through unchanged in
//! the standardized space (effective divisor 1). Callers should drop
//! constant columns before fitting; we don't error on them so a stray
//! constant column doesn't kill an otherwise valid fit.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[derive(Debug, Clone)]
pub struct StandardizeConfig {
    pub center_x: bool,
    pub scale_x: bool,
    pub fit_intercept: bool,
}

impl Default for StandardizeConfig {
    fn default() -> Self {
        Self {
            center_x: true,
            scale_x: true,
            fit_intercept: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StandardizationStats {
    /// Column means of X. `Some` iff `config.center_x`.
    pub x_means: Option<Array1<f64>>,
    /// Per-column population stds of X (relative to the column mean).
    /// `Some` iff `config.scale_x`. Entries can be 0 for constant columns.
    pub x_scales: Option<Array1<f64>>,
    /// Mean of y. `Some` iff `config.fit_intercept`.
    pub y_mean: Option<f64>,
}

/// Apply standardization to `(X, y)`, returning `(X̃, ỹ, stats)`.
///
/// `X` and `y` are read from views; the result is owned, so callers keep
/// their originals.
pub fn standardize(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &StandardizeConfig,
) -> (Array2<f64>, Array1<f64>, StandardizationStats) {
    let n = x.nrows();
    let p = x.ncols();
    let n_f = n as f64;

    // Means and centered-population stds — computed unconditionally because
    // `scale_j` is always defined relative to the column mean (glmnet
    // convention) regardless of whether we then re-center.
    let means: Array1<f64> = Array1::from_iter((0..p).map(|j| x.column(j).sum() / n_f));
    let scales: Array1<f64> = Array1::from_iter((0..p).map(|j| {
        let m = means[j];
        let sq: f64 = x.column(j).iter().map(|v| (v - m).powi(2)).sum();
        (sq / n_f).sqrt()
    }));

    let mut xs = x.to_owned();
    if config.center_x {
        for j in 0..p {
            let m = means[j];
            for i in 0..n {
                xs[[i, j]] -= m;
            }
        }
    }
    if config.scale_x {
        for j in 0..p {
            let s = scales[j];
            // scale = 0 ⇒ effective divisor 1; column passes through.
            if s > 0.0 {
                for i in 0..n {
                    xs[[i, j]] /= s;
                }
            }
        }
    }

    let (ys, y_mean) = if config.fit_intercept {
        let ym = y.sum() / n_f;
        (&y.to_owned() - ym, Some(ym))
    } else {
        (y.to_owned(), None)
    };

    let stats = StandardizationStats {
        x_means: if config.center_x { Some(means) } else { None },
        x_scales: if config.scale_x { Some(scales) } else { None },
        y_mean,
    };

    (xs, ys, stats)
}

/// Convert standardized-scale `β̃` back to original-scale `(β, α)`.
pub fn destandardize(
    beta_std: ArrayView1<f64>,
    stats: &StandardizationStats,
) -> (Array1<f64>, f64) {
    let mut beta = beta_std.to_owned();
    if let Some(scales) = &stats.x_scales {
        for j in 0..beta.len() {
            if scales[j] > 0.0 {
                beta[j] /= scales[j];
            }
        }
    }

    let alpha = match (stats.y_mean, &stats.x_means) {
        (Some(ym), Some(means)) => {
            let dot: f64 = (0..beta.len()).map(|j| beta[j] * means[j]).sum();
            ym - dot
        }
        // No centering ⇒ no shift correction; `α` is just `ȳ` (or 0 when no intercept).
        (Some(ym), None) => ym,
        (None, _) => 0.0,
    };

    (beta, alpha)
}

/// Vectorized over rows of a `(n_lambdas, n_features)` β-path.
pub fn destandardize_path(
    betas_std: ArrayView2<f64>,
    stats: &StandardizationStats,
) -> (Array2<f64>, Array1<f64>) {
    let n_lams = betas_std.nrows();
    let p = betas_std.ncols();
    let mut betas = Array2::<f64>::zeros((n_lams, p));
    let mut alphas = Array1::<f64>::zeros(n_lams);
    for k in 0..n_lams {
        let (b, a) = destandardize(betas_std.row(k), stats);
        betas.row_mut(k).assign(&b);
        alphas[k] = a;
    }
    (betas, alphas)
}

/// Rescale per-feature penalty weights for use in standardized space.
///
/// When the user wants the original-scale penalty `λ Σ w_j |β_j|` and we
/// solve in standardized space (`β̃_j = β_j · s_j`), the equivalent penalty
/// becomes `λ Σ (w_j / s_j) |β̃_j|`. This helper returns those rescaled
/// weights so the standardized solver and the original-scale model describe
/// the same problem.
///
/// When `stats.x_scales` is `None` (no scaling applied), returns the input
/// unchanged. Zero-scale columns pass through unchanged: the standardized
/// column is identically zero, so `β̃_j` stays at 0 regardless of the
/// weight, and dividing by zero would just produce `inf` for no benefit.
pub fn rescale_weights_for_standardize(
    weights_orig: ArrayView1<f64>,
    stats: &StandardizationStats,
) -> Array1<f64> {
    let mut out = weights_orig.to_owned();
    if let Some(scales) = &stats.x_scales {
        for j in 0..out.len() {
            if scales[j] > 0.0 {
                out[j] /= scales[j];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

    fn small_xy() -> (Array2<f64>, Array1<f64>) {
        // Column 0 has mean 2, variance 2/3 ⇒ std = sqrt(2/3).
        // Column 1 has mean 20, variance 200/3 ⇒ std = sqrt(200/3).
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let y = array![1.0, 2.0, 3.0]; // mean 2
        (x, y)
    }

    fn col_mean(x: &Array2<f64>, j: usize) -> f64 {
        let n = x.nrows() as f64;
        x.column(j).sum() / n
    }

    fn col_pop_var(x: &Array2<f64>, j: usize) -> f64 {
        let n = x.nrows() as f64;
        let m = col_mean(x, j);
        x.column(j).iter().map(|v| (v - m).powi(2)).sum::<f64>() / n
    }

    // ---- standardize: flag matrix ---------------------------------------

    #[test]
    fn standardize_no_op_when_all_disabled() {
        let (x, y) = small_xy();
        let cfg = StandardizeConfig {
            center_x: false,
            scale_x: false,
            fit_intercept: false,
        };
        let (xs, ys, stats) = standardize(x.view(), y.view(), &cfg);
        assert_eq!(xs, x);
        assert_eq!(ys, y);
        assert!(stats.x_means.is_none());
        assert!(stats.x_scales.is_none());
        assert!(stats.y_mean.is_none());
    }

    #[test]
    fn standardize_centers_x_to_zero_mean() {
        let (x, y) = small_xy();
        let cfg = StandardizeConfig {
            center_x: true,
            scale_x: false,
            fit_intercept: false,
        };
        let (xs, _, stats) = standardize(x.view(), y.view(), &cfg);
        for j in 0..x.ncols() {
            assert_abs_diff_eq!(col_mean(&xs, j), 0.0, epsilon = 1e-12);
        }
        let means = stats.x_means.expect("x_means stored when center_x");
        assert_abs_diff_eq!(means[0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(means[1], 20.0, epsilon = 1e-12);
        assert!(stats.x_scales.is_none());
    }

    #[test]
    fn standardize_scales_x_to_unit_population_var_when_centered() {
        let (x, y) = small_xy();
        let cfg = StandardizeConfig {
            center_x: true,
            scale_x: true,
            fit_intercept: false,
        };
        let (xs, _, stats) = standardize(x.view(), y.view(), &cfg);
        for j in 0..x.ncols() {
            assert_abs_diff_eq!(col_pop_var(&xs, j), 1.0, epsilon = 1e-12);
        }
        let scales = stats.x_scales.expect("x_scales stored when scale_x");
        assert_abs_diff_eq!(scales[0], (2.0_f64 / 3.0).sqrt(), epsilon = 1e-12);
        assert_abs_diff_eq!(scales[1], (200.0_f64 / 3.0).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn standardize_scale_only_uses_centered_std_as_divisor() {
        // glmnet convention: scale_j is computed relative to the column mean
        // even when center_x = false. This means scale_only divides without
        // recentering — the column has nonzero mean afterward.
        let (x, y) = small_xy();
        let cfg = StandardizeConfig {
            center_x: false,
            scale_x: true,
            fit_intercept: false,
        };
        let (xs, _, stats) = standardize(x.view(), y.view(), &cfg);
        let scales = stats.x_scales.unwrap();
        for j in 0..x.ncols() {
            assert_abs_diff_eq!(scales[j], col_pop_var(&x, j).sqrt(), epsilon = 1e-12);
            for i in 0..x.nrows() {
                assert_abs_diff_eq!(xs[[i, j]], x[[i, j]] / scales[j], epsilon = 1e-12);
            }
        }
        assert!(stats.x_means.is_none());
    }

    #[test]
    fn standardize_centers_y_when_fit_intercept_else_passes_through() {
        let (x, y) = small_xy();

        let with_int = StandardizeConfig {
            center_x: false,
            scale_x: false,
            fit_intercept: true,
        };
        let (_, ys, stats) = standardize(x.view(), y.view(), &with_int);
        assert_abs_diff_eq!(ys.sum() / ys.len() as f64, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(stats.y_mean.unwrap(), 2.0, epsilon = 1e-12);

        let no_int = StandardizeConfig {
            center_x: false,
            scale_x: false,
            fit_intercept: false,
        };
        let (_, ys2, stats2) = standardize(x.view(), y.view(), &no_int);
        assert_eq!(ys2, y);
        assert!(stats2.y_mean.is_none());
    }

    #[test]
    fn standardize_constant_column_passes_through_with_zero_scale_recorded() {
        let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]];
        let y = array![0.0, 0.0, 0.0];
        let cfg = StandardizeConfig {
            center_x: true,
            scale_x: true,
            fit_intercept: false,
        };
        let (xs, _, stats) = standardize(x.view(), y.view(), &cfg);
        // After centering the constant column, all entries are 0.
        // Effective scale = 1, so they stay 0.
        for i in 0..3 {
            assert_abs_diff_eq!(xs[[i, 0]], 0.0, epsilon = 1e-12);
        }
        // Stats record the actual (zero) scale so the caller can detect it.
        let scales = stats.x_scales.unwrap();
        assert_abs_diff_eq!(scales[0], 0.0, epsilon = 1e-12);
        assert!(scales[1] > 0.0);
    }

    // ---- destandardize ---------------------------------------------------

    #[test]
    fn destandardize_no_op_when_all_disabled() {
        let beta = array![1.0, -2.0, 0.5];
        let stats = StandardizationStats {
            x_means: None,
            x_scales: None,
            y_mean: None,
        };
        let (b, alpha) = destandardize(beta.view(), &stats);
        assert_eq!(b, beta);
        assert_abs_diff_eq!(alpha, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn destandardize_with_only_centering_recovers_correct_intercept() {
        // No scaling ⇒ β unchanged. α = ȳ - <β, x̄>.
        let beta = array![2.0, -1.0];
        let stats = StandardizationStats {
            x_means: Some(array![3.0, 4.0]),
            x_scales: None,
            y_mean: Some(10.0),
        };
        let (b, alpha) = destandardize(beta.view(), &stats);
        assert_eq!(b, beta);
        assert_abs_diff_eq!(alpha, 10.0 - (2.0 * 3.0 + -4.0), epsilon = 1e-12);
    }

    #[test]
    fn destandardize_with_only_scaling_unscales_beta() {
        let beta_std = array![2.0, -1.0];
        let stats = StandardizationStats {
            x_means: None,
            x_scales: Some(array![0.5, 4.0]),
            y_mean: None,
        };
        let (b, alpha) = destandardize(beta_std.view(), &stats);
        assert_abs_diff_eq!(b[0], 2.0 / 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(b[1], -1.0 / 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(alpha, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn destandardize_zero_scale_column_leaves_beta_unchanged() {
        let beta_std = array![0.0, 1.5];
        let stats = StandardizationStats {
            x_means: Some(array![5.0, 0.0]),
            x_scales: Some(array![0.0, 2.0]),
            y_mean: Some(7.0),
        };
        let (b, alpha) = destandardize(beta_std.view(), &stats);
        // Zero-scale ⇒ β passes through (β̃_0 was 0 anyway).
        assert_abs_diff_eq!(b[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(b[1], 1.5 / 2.0, epsilon = 1e-12);
        // α uses the original-scale β: α = 7 - (0·5 + 0.75·0) = 7.
        assert_abs_diff_eq!(alpha, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn destandardize_path_matches_per_row_destandardize() {
        let betas_std = array![[2.0, -1.0], [4.0, -2.0], [0.0, 0.0]];
        let stats = StandardizationStats {
            x_means: Some(array![3.0, 4.0]),
            x_scales: Some(array![0.5, 2.0]),
            y_mean: Some(10.0),
        };
        let (bs, alphas) = destandardize_path(betas_std.view(), &stats);
        for k in 0..betas_std.nrows() {
            let (b_row, alpha_row) = destandardize(betas_std.row(k), &stats);
            for j in 0..betas_std.ncols() {
                assert_abs_diff_eq!(bs[[k, j]], b_row[j], epsilon = 1e-12);
            }
            assert_abs_diff_eq!(alphas[k], alpha_row, epsilon = 1e-12);
        }
    }

    // ---- prediction equivalence (the load-bearing test) -----------------
    //
    // For any β̃, the standardized linear model `X̃ β̃ + ȳ` must equal the
    // de-standardized model `X β + α` on the original X. This is the
    // mathematical guarantee that lets us solve in standardized space.

    #[test]
    fn standardized_and_destandardized_predictions_agree_for_arbitrary_beta() {
        let (x, y) = small_xy();
        let cfg = StandardizeConfig::default();
        let (xs, _, stats) = standardize(x.view(), y.view(), &cfg);
        let beta_std = array![0.7, -1.3];
        let (beta, alpha) = destandardize(beta_std.view(), &stats);

        let pred_std: Array1<f64> = xs.dot(&beta_std) + stats.y_mean.unwrap();
        let pred_orig: Array1<f64> = x.dot(&beta) + alpha;
        for i in 0..x.nrows() {
            assert_abs_diff_eq!(pred_std[i], pred_orig[i], epsilon = 1e-10);
        }
    }

    // ---- end-to-end with the path solver -------------------------------

    #[test]
    fn end_to_end_at_lambda_max_intercept_equals_y_mean() {
        // β̃ = 0 at λ_max ⇒ β = 0 (after destandardize) ⇒ α = ȳ.
        use crate::datafit::LeastSquares;
        use crate::design::DenseMatrix;
        use crate::penalty::Mcp;
        use crate::solver::{lambda_max, solve_path, CdConfig, PathConfig, Screening};

        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];

        let cfg = StandardizeConfig::default();
        let (xs, ys, stats) = standardize(x.view(), y.view(), &cfg);

        let design = DenseMatrix::new(xs);
        let datafit = LeastSquares::new(ys);
        let p = x.ncols();
        let weights = Array1::<f64>::ones(p);
        let lam_max = lambda_max(&design, &datafit, weights.view());

        let path_cfg = PathConfig {
            n_lambdas: 1,
            lambda_min_ratio: 1.0,
            lambdas: Some(vec![lam_max]),
            cd: CdConfig::default(),
            screening: Screening::Strong,
            p0: 10,
        };
        let (betas_std, _) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 100.0, p)),
            &path_cfg,
        );
        let (betas_orig, alphas) = destandardize_path(betas_std.view(), &stats);

        for j in 0..p {
            assert_abs_diff_eq!(betas_orig[[0, j]], 0.0, epsilon = 1e-8);
        }
        let y_mean = y.sum() / y.len() as f64;
        assert_abs_diff_eq!(alphas[0], y_mean, epsilon = 1e-8);
    }

    #[test]
    fn end_to_end_recovers_truth_at_tiny_lambda_no_noise() {
        // y = X β + α exactly. At λ → 0 the fit should reproduce (β, α)
        // up to convergence tolerance.
        use crate::datafit::LeastSquares;
        use crate::design::DenseMatrix;
        use crate::penalty::Mcp;
        use crate::solver::{solve_path, CdConfig, PathConfig, Screening};

        let x = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [2.0, -1.0],
            [-2.0, 0.5]
        ];
        let true_beta = array![2.0, -3.0];
        let true_alpha = 5.0;
        let y = x.dot(&true_beta) + true_alpha;

        let cfg = StandardizeConfig::default();
        let (xs, ys, stats) = standardize(x.view(), y.view(), &cfg);
        let design = DenseMatrix::new(xs);
        let datafit = LeastSquares::new(ys);
        let p = x.ncols();

        let path_cfg = PathConfig {
            n_lambdas: 1,
            lambda_min_ratio: 1.0,
            lambdas: Some(vec![1e-8]),
            cd: CdConfig {
                max_iter: 50_000,
                tol: 1e-14,
                acceleration: Some(5),
            },
            screening: Screening::Strong,
            p0: 10,
        };
        let (betas_std, _) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 100.0, p)),
            &path_cfg,
        );
        let (betas_orig, alphas) = destandardize_path(betas_std.view(), &stats);

        assert_abs_diff_eq!(betas_orig[[0, 0]], true_beta[0], epsilon = 1e-3);
        assert_abs_diff_eq!(betas_orig[[0, 1]], true_beta[1], epsilon = 1e-3);
        assert_abs_diff_eq!(alphas[0], true_alpha, epsilon = 1e-3);
    }

    #[test]
    fn lambda_max_under_standardization_differs_from_raw_when_scales_unequal() {
        // For the small_xy() problem, column scales are very different
        // (sqrt(2/3) vs sqrt(200/3)). λ_max on raw vs standardized X
        // should not coincide.
        use crate::datafit::LeastSquares;
        use crate::design::DenseMatrix;
        use crate::solver::lambda_max;

        let (x, y) = small_xy();
        let p = x.ncols();
        let weights = Array1::<f64>::ones(p);

        let raw_design = DenseMatrix::new(x.clone());
        let raw_datafit = LeastSquares::new(y.clone());
        let lam_raw = lambda_max(&raw_design, &raw_datafit, weights.view());

        let cfg = StandardizeConfig::default();
        let (xs, ys, _) = standardize(x.view(), y.view(), &cfg);
        let std_design = DenseMatrix::new(xs);
        let std_datafit = LeastSquares::new(ys);
        let lam_std = lambda_max(&std_design, &std_datafit, weights.view());

        assert!(
            (lam_raw - lam_std).abs() > 1e-6,
            "λ_max should differ under standardization (raw={}, std={})",
            lam_raw,
            lam_std
        );
    }

    #[test]
    fn predictions_agree_when_only_centering_no_scaling() {
        let (x, y) = small_xy();
        let cfg = StandardizeConfig {
            center_x: true,
            scale_x: false,
            fit_intercept: true,
        };
        let (xs, _, stats) = standardize(x.view(), y.view(), &cfg);
        let beta_std = array![0.4, 0.05];
        let (beta, alpha) = destandardize(beta_std.view(), &stats);
        let pred_std = xs.dot(&beta_std) + stats.y_mean.unwrap();
        let pred_orig = x.dot(&beta) + alpha;
        for i in 0..x.nrows() {
            assert_abs_diff_eq!(pred_std[i], pred_orig[i], epsilon = 1e-10);
        }
    }

    // ---- weight rescaling for standardized space ------------------------

    #[test]
    fn rescale_weights_no_scaling_returns_input() {
        let w = array![0.5, 1.0, 2.0];
        let stats = StandardizationStats {
            x_means: Some(array![0.0, 0.0, 0.0]),
            x_scales: None,
            y_mean: None,
        };
        let out = rescale_weights_for_standardize(w.view(), &stats);
        assert_eq!(out, w);
    }

    #[test]
    fn rescale_weights_divides_by_per_column_scale() {
        let w = array![1.0, 2.0, 3.0];
        let scales = array![0.5, 4.0, 1.5];
        let stats = StandardizationStats {
            x_means: None,
            x_scales: Some(scales.clone()),
            y_mean: None,
        };
        let out = rescale_weights_for_standardize(w.view(), &stats);
        for j in 0..3 {
            assert_abs_diff_eq!(out[j], w[j] / scales[j], epsilon = 1e-12);
        }
    }

    #[test]
    fn rescale_weights_zero_scale_column_passes_through() {
        let w = array![1.5, 2.0];
        let stats = StandardizationStats {
            x_means: None,
            x_scales: Some(array![0.0, 4.0]),
            y_mean: None,
        };
        let out = rescale_weights_for_standardize(w.view(), &stats);
        assert_abs_diff_eq!(out[0], 1.5, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1], 2.0 / 4.0, epsilon = 1e-12);
    }

    #[test]
    fn rescaled_weights_in_std_space_match_orig_weights_in_raw_space() {
        // Solve the same lasso problem two ways:
        // (1) Raw X with original weights
        // (2) Standardized X with rescaled weights, then destandardize
        // The recovered β (original-scale) and α must agree.
        use crate::datafit::LeastSquares;
        use crate::design::DenseMatrix;
        use crate::penalty::Mcp;
        use crate::solver::{solve_path, CdConfig, PathConfig, Screening};

        let x = array![
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 4.0],
            [0.0, 1.0, 1.0],
            [3.0, 2.0, 5.0],
            [1.5, 0.5, 3.0]
        ];
        let true_beta = array![1.0, -2.0, 0.5];
        let true_alpha = 0.0; // no intercept for cleanest equivalence
        let y = x.dot(&true_beta) + true_alpha;
        let p = x.ncols();
        let w_orig = array![1.0, 0.7, 1.3];

        // Path 1: raw X, no standardization, original weights.
        let design_raw = DenseMatrix::new(x.clone());
        let datafit_raw = LeastSquares::new(y.clone());
        let path_cfg = PathConfig {
            n_lambdas: 1,
            lambda_min_ratio: 1.0,
            lambdas: Some(vec![1e-8]),
            cd: CdConfig {
                max_iter: 100_000,
                tol: 1e-14,
                acceleration: Some(5),
            },
            screening: Screening::Strong,
            p0: 10,
        };
        let pen1 = w_orig.clone();
        let (b_raw, _) = solve_path(
            &design_raw,
            &datafit_raw,
            move |lam| Box::new(Mcp::with_weights(lam, 1e6, pen1.clone())),
            &path_cfg,
        );
        let beta_raw = b_raw.row(0).to_owned();

        // Path 2: standardize (no centering, scale only ⇒ no intercept),
        // rescaled weights, destandardize.
        let cfg = StandardizeConfig {
            center_x: false,
            scale_x: true,
            fit_intercept: false,
        };
        let (xs, ys, stats) = standardize(x.view(), y.view(), &cfg);
        let w_std = rescale_weights_for_standardize(w_orig.view(), &stats);
        let design_std = DenseMatrix::new(xs);
        let datafit_std = LeastSquares::new(ys);
        let pen2 = w_std.clone();
        let (b_std, _) = solve_path(
            &design_std,
            &datafit_std,
            move |lam| Box::new(Mcp::with_weights(lam, 1e6, pen2.clone())),
            &path_cfg,
        );
        let (beta_destd, _) = destandardize_path(b_std.view(), &stats);
        let beta_destd = beta_destd.row(0).to_owned();

        for j in 0..p {
            assert_abs_diff_eq!(beta_raw[j], beta_destd[j], epsilon = 1e-4);
        }
    }
}

/// Randomized bijection coverage for `standardize` / `destandardize` (H3).
///
/// The standardized solver fits `β̃` in transformed space and the caller
/// recovers original-scale `(β, α)` via `destandardize`. The full
/// round-trip identity is:
///
/// ```text
///     β̃_j = β_orig_j · s_j                   (definition of β̃)
///     destandardize(β̃, stats) = (β_orig, α)  (bijection)
///     α       = ȳ − Σ_j β_orig_j · x̄_j         (when centered + fit_intercept)
/// ```
///
/// The properties cover every flag combination (`center_x × scale_x ×
/// fit_intercept`) so that destandardize stays consistent if the
/// branching ever changes shape.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::collection::vec as prop_vec;
    use proptest::prelude::*;

    const N_SAMPLES: usize = 8;
    const N_FEATURES: usize = 4;

    fn arb_x_flat() -> impl Strategy<Value = Vec<f64>> {
        prop_vec(-3.0_f64..3.0, N_SAMPLES * N_FEATURES)
    }
    fn arb_y() -> impl Strategy<Value = Vec<f64>> {
        prop_vec(-3.0_f64..3.0, N_SAMPLES)
    }
    fn arb_beta_orig() -> impl Strategy<Value = Vec<f64>> {
        prop_vec(-2.0_f64..2.0, N_FEATURES)
    }
    fn arb_config() -> impl Strategy<Value = StandardizeConfig> {
        (any::<bool>(), any::<bool>(), any::<bool>()).prop_map(|(c, s, fi)| StandardizeConfig {
            center_x: c,
            scale_x: s,
            fit_intercept: fi,
        })
    }

    fn build_x(flat: &[f64]) -> Array2<f64> {
        Array2::from_shape_vec((N_SAMPLES, N_FEATURES), flat.to_vec()).expect("shape ok")
    }

    proptest! {
        /// `destandardize(standardize_β(β_orig)) == β_orig`, for every
        /// (center_x, scale_x, fit_intercept) combination.
        ///
        /// Zero-variance columns break the bijection by construction
        /// (`β̃_j = β_orig_j · 0` loses the original-scale value), so
        /// drop those draws. Continuous random inputs hit them with
        /// probability zero — the assume just makes the failure mode
        /// explicit.
        #[test]
        fn destandardize_inverts_standardize_beta(
            x_flat in arb_x_flat(),
            y_vec in arb_y(),
            beta_orig_vec in arb_beta_orig(),
            cfg in arb_config(),
        ) {
            let x = build_x(&x_flat);
            let y = Array1::from(y_vec);
            let beta_orig = Array1::from(beta_orig_vec);

            let (_, _, stats) = standardize(x.view(), y.view(), &cfg);

            // Skip degenerate column draws.
            if let Some(scales) = &stats.x_scales {
                prop_assume!(scales.iter().all(|&s| s > 1e-6));
            }

            // Form β̃ from β_orig according to the standardization stats.
            let beta_std: Array1<f64> = match &stats.x_scales {
                Some(s) => Array1::from_iter((0..N_FEATURES).map(|j| beta_orig[j] * s[j])),
                None => beta_orig.clone(),
            };

            let (beta_back, alpha) = destandardize(beta_std.view(), &stats);

            // β recovered to round-off.
            for j in 0..N_FEATURES {
                prop_assert!(
                    (beta_back[j] - beta_orig[j]).abs() < 1e-10 * (1.0 + beta_orig[j].abs()),
                    "j={}, expected={}, got={}", j, beta_orig[j], beta_back[j],
                );
            }

            // Intercept matches the documented `α = ȳ − Σ β_j x̄_j` form,
            // collapsing as the flags toggle off.
            let n_f = N_SAMPLES as f64;
            let y_mean = y.sum() / n_f;
            let expected_alpha = match (cfg.fit_intercept, cfg.center_x) {
                (false, _) => 0.0,
                (true, false) => y_mean,
                (true, true) => {
                    let means = stats.x_means.as_ref().expect("center_x ⇒ means");
                    let dot: f64 = (0..N_FEATURES).map(|j| beta_orig[j] * means[j]).sum();
                    y_mean - dot
                }
            };
            prop_assert!(
                (alpha - expected_alpha).abs() < 1e-10 * (1.0 + expected_alpha.abs()),
                "alpha: expected={}, got={}", expected_alpha, alpha,
            );
        }

        /// `destandardize_path` is the row-wise lift of `destandardize`
        /// and must agree on every row.
        #[test]
        fn destandardize_path_matches_per_row(
            x_flat in arb_x_flat(),
            y_vec in arb_y(),
            betas_orig_flat in prop_vec(-2.0_f64..2.0, 3 * N_FEATURES),
            cfg in arb_config(),
        ) {
            let x = build_x(&x_flat);
            let y = Array1::from(y_vec);
            let (_, _, stats) = standardize(x.view(), y.view(), &cfg);
            if let Some(scales) = &stats.x_scales {
                prop_assume!(scales.iter().all(|&s| s > 1e-6));
            }

            let betas_orig = Array2::from_shape_vec((3, N_FEATURES), betas_orig_flat).unwrap();
            let mut betas_std = betas_orig.clone();
            if let Some(s) = &stats.x_scales {
                for k in 0..3 {
                    for j in 0..N_FEATURES {
                        betas_std[[k, j]] *= s[j];
                    }
                }
            }

            let (path_betas, path_alphas) = destandardize_path(betas_std.view(), &stats);
            for k in 0..3 {
                let (b_row, a_row) = destandardize(betas_std.row(k), &stats);
                for j in 0..N_FEATURES {
                    prop_assert!((path_betas[[k, j]] - b_row[j]).abs() < 1e-12);
                }
                prop_assert!((path_alphas[k] - a_row).abs() < 1e-12);
            }
        }

        /// `rescale_weights_for_standardize` lifts the per-feature
        /// L1 penalty `λ Σ w_j |β_j|` into standardized space as
        /// `λ Σ (w_j / s_j) |β̃_j|`. The composed identity:
        /// `(w_j / s_j) · |β̃_j| = (w_j / s_j) · |β_orig_j · s_j|
        ///                       = w_j · |β_orig_j|`.
        #[test]
        fn rescale_weights_preserves_penalty_value(
            x_flat in arb_x_flat(),
            y_vec in arb_y(),
            beta_orig_vec in arb_beta_orig(),
            w_vec in prop_vec(0.0_f64..3.0, N_FEATURES),
            cfg in arb_config(),
        ) {
            let x = build_x(&x_flat);
            let y = Array1::from(y_vec);
            let beta_orig = Array1::from(beta_orig_vec);
            let weights = Array1::from(w_vec);

            let (_, _, stats) = standardize(x.view(), y.view(), &cfg);
            if let Some(scales) = &stats.x_scales {
                prop_assume!(scales.iter().all(|&s| s > 1e-6));
            }

            let w_rescaled = rescale_weights_for_standardize(weights.view(), &stats);
            let beta_std: Array1<f64> = match &stats.x_scales {
                Some(s) => Array1::from_iter((0..N_FEATURES).map(|j| beta_orig[j] * s[j])),
                None => beta_orig.clone(),
            };

            let pen_orig: f64 = (0..N_FEATURES).map(|j| weights[j] * beta_orig[j].abs()).sum();
            let pen_std: f64 = (0..N_FEATURES).map(|j| w_rescaled[j] * beta_std[j].abs()).sum();
            prop_assert!(
                (pen_orig - pen_std).abs() < 1e-10 * (1.0 + pen_orig.abs()),
                "penalty mismatch: orig={}, std={}", pen_orig, pen_std,
            );
        }
    }
}
