//! Multinomial (softmax) logistic regression via proximal-Newton.
//!
//! Penalized K-class softmax with coefficient matrix `B ∈ ℝ^{p×K}` laid out
//! row-major so the M2 row-group penalty `λ Σ_j w_j ‖B[j, :]‖_2` becomes a
//! contiguous-block group penalty on `bvec[jK + k] = B[j, k]`. Combined with
//! the task-outer-stacked response `z̃[kn + i] = z_{i, k}` and the M7
//! `MultiTaskDesign<X>` wrapper, the whole problem is a multi-task LS that
//! the M2 block-CD machinery handles unchanged.
//!
//! Symmetric (no reference class) parameterization, matching glmnet:
//! ```text
//!     η_{i, k} = X[i, :] · B[:, k]
//!     p_{i, k} = exp(η_{i, k}) / Σ_l exp(η_{i, l})
//! ```
//!
//! Surrogate uses Böhning's diagonal majorization
//! `diag(p_i) − p_i p_iᵀ ⪯ (1/2)(I − 11ᵀ/K)` — equivalently a constant
//! per-(i, k) Hessian diagonal of `1/2`. This decouples the K classes into
//! independent weighted-LS subproblems sharing the same design X but with
//! different working responses, and is the recipe `glmnet` uses (Friedman
//! / Hastie / Tibshirani 2010, §4.4):
//! ```text
//!     g_{i, k} = p_{i, k} − Y_{i, k}
//!     z_{i, k} = η_{i, k} − 2 g_{i, k}
//!     w_{i, k} = 1/2                         (× sample_weights[i] if set)
//! ```
//!
//! Loss is the cross-entropy through a numerically stable logsumexp:
//! ```text
//!     L(β) = (1/n) Σ_i (logsumexp(η_i) − Σ_k Y_{i, k} η_{i, k})
//! ```
//!
//! At β = 0 every `p_{i, k} = 1/K`, so `z_{i, k} = 2 (Y_{i, k} − 1/K)` and
//! the loss equals `log K` — the uniform-prior baseline.

use super::{GlmDatafit, LeastSquares};
use crate::design::DesignMatrix;
use ndarray::{Array1, Array2, ArrayView1};

pub struct MultinomialLogit {
    /// `(n, K)` one-hot or soft-label response. Rows summing to 1 is the
    /// canonical case but not enforced — soft labels (probabilities)
    /// compose cleanly through the same gradient/Hessian.
    y: Array2<f64>,
    n_classes: usize,
    sample_weights: Option<Array1<f64>>,
}

impl MultinomialLogit {
    pub fn new(y: Array2<f64>) -> Self {
        let n_classes = y.ncols();
        assert!(n_classes >= 2, "MultinomialLogit: n_classes must be ≥ 2");
        Self {
            y,
            n_classes,
            sample_weights: None,
        }
    }

    pub fn with_sample_weights(y: Array2<f64>, w: Array1<f64>) -> Self {
        assert_eq!(
            y.nrows(),
            w.len(),
            "sample_weights length must equal n_samples (Y.nrows())"
        );
        let n_classes = y.ncols();
        assert!(n_classes >= 2, "MultinomialLogit: n_classes must be ≥ 2");
        Self {
            y,
            n_classes,
            sample_weights: Some(w),
        }
    }

    /// Build from a length-n vector of integer class labels in
    /// `{0, ..., K-1}` by one-hot encoding.
    pub fn from_labels(labels: ArrayView1<'_, f64>, n_classes: usize) -> Self {
        assert!(n_classes >= 2, "MultinomialLogit: n_classes must be ≥ 2");
        let n = labels.len();
        let mut y = Array2::<f64>::zeros((n, n_classes));
        for i in 0..n {
            let k = labels[i] as usize;
            assert!(
                (labels[i] - k as f64).abs() < 1e-12 && k < n_classes,
                "label {} at row {} out of range for n_classes = {}",
                labels[i],
                i,
                n_classes
            );
            y[[i, k]] = 1.0;
        }
        Self {
            y,
            n_classes,
            sample_weights: None,
        }
    }

    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    pub fn y(&self) -> &Array2<f64> {
        &self.y
    }

    /// Original multinomial cross-entropy loss
    /// `(1/n) Σ_i (logsumexp(η_i) − Σ_k Y_{i,k} η_{i,k})` evaluated at `β`.
    /// `design` must be the K-task multi-task wrapping of the user matrix
    /// (so `design.n_samples() = n · K`).
    pub fn loss(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> f64 {
        let k = self.n_classes;
        let n = design.n_samples() / k;
        debug_assert_eq!(
            design.n_samples(),
            n * k,
            "design.n_samples must equal n_base * n_classes for multinomial logit"
        );
        let eta_in = design.matvec(beta);
        let (eta, lse, _p) = reshape_and_softmax(&eta_in, n, k);

        let mut total = 0.0_f64;
        for i in 0..n {
            let mut inner = 0.0_f64;
            for kk in 0..k {
                inner += self.y[[i, kk]] * eta[[i, kk]];
            }
            let term = lse[i] - inner;
            let w = self.sample_weights.as_ref().map(|w| w[i]).unwrap_or(1.0);
            total += w * term;
        }
        total / (n as f64)
    }

    /// Build the local quadratic surrogate at `β` as a task-outer-stacked
    /// `LeastSquares` of length `nK`. Per-(i,k) weight is `1/2`
    /// (Böhning bound), per-(i,k) working response is
    /// `z_{i,k} = η_{i,k} − 2(p_{i,k} − Y_{i,k})`. Per-sample multipliers
    /// from `with_sample_weights` (if set) compose by replicating across
    /// all K tasks for each base sample.
    pub fn surrogate_at(
        &self,
        design: &dyn DesignMatrix,
        beta: ArrayView1<'_, f64>,
    ) -> LeastSquares {
        let k = self.n_classes;
        let n = design.n_samples() / k;
        debug_assert_eq!(
            design.n_samples(),
            n * k,
            "design.n_samples must equal n_base * n_classes for multinomial logit"
        );
        let eta_in = design.matvec(beta);
        let (eta, _lse, p) = reshape_and_softmax(&eta_in, n, k);

        let mut z = Array1::<f64>::zeros(n * k);
        let mut w = Array1::<f64>::zeros(n * k);
        for task in 0..k {
            for i in 0..n {
                let g = p[[i, task]] - self.y[[i, task]];
                let zi = eta[[i, task]] - 2.0 * g;
                let scale = self.sample_weights.as_ref().map(|sw| sw[i]).unwrap_or(1.0);
                let idx = task * n + i;
                z[idx] = zi;
                w[idx] = 0.5 * scale;
            }
        }
        LeastSquares::with_sample_weights(z, w)
    }
}

impl GlmDatafit for MultinomialLogit {
    fn surrogate_at(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> LeastSquares {
        MultinomialLogit::surrogate_at(self, design, beta)
    }

    fn loss(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> f64 {
        MultinomialLogit::loss(self, design, beta)
    }
}

/// Reshape a length-`nK` task-outer η (as produced by
/// `MultiTaskDesign::matvec`) into an `(n, K)` matrix and compute per-row
/// stable logsumexp + softmax probabilities.
fn reshape_and_softmax(
    eta_in: &Array1<f64>,
    n: usize,
    k: usize,
) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let mut eta = Array2::<f64>::zeros((n, k));
    for task in 0..k {
        for i in 0..n {
            eta[[i, task]] = eta_in[task * n + i];
        }
    }
    let mut lse = Array1::<f64>::zeros(n);
    let mut p = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        let mut m = f64::NEG_INFINITY;
        for kk in 0..k {
            if eta[[i, kk]] > m {
                m = eta[[i, kk]];
            }
        }
        let mut s = 0.0_f64;
        for kk in 0..k {
            s += (eta[[i, kk]] - m).exp();
        }
        lse[i] = m + s.ln();
        for kk in 0..k {
            p[[i, kk]] = (eta[[i, kk]] - lse[i]).exp();
        }
    }
    (eta, lse, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::{DenseMatrix, MultiTaskDesign};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn from_labels_one_hots_correctly() {
        let labels = array![0.0, 1.0, 2.0, 1.0, 0.0];
        let glm = MultinomialLogit::from_labels(labels.view(), 3);
        assert_eq!(glm.y().shape(), &[5, 3]);
        for (i, &lab) in labels.iter().enumerate() {
            for k in 0..3 {
                let expected = if k == lab as usize { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(glm.y()[[i, k]], expected, epsilon = 0.0);
            }
        }
    }

    #[test]
    #[should_panic(expected = "n_classes must be ≥ 2")]
    fn panics_on_one_class() {
        let _ = MultinomialLogit::new(Array2::<f64>::zeros((3, 1)));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn from_labels_panics_on_out_of_range() {
        let labels = array![0.0, 1.0, 5.0];
        let _ = MultinomialLogit::from_labels(labels.view(), 3);
    }

    #[test]
    fn loss_at_zero_is_log_k() {
        // β = 0 ⇒ η = 0 ⇒ p_{i,k} = 1/K, so per-sample term is
        // logsumexp(0..0) − Σ_k Y_{i,k}·0 = log K. Average over samples = log K.
        let x = array![[1.0, 0.5], [-0.4, 0.3], [0.2, 0.8]];
        let labels = array![0.0, 1.0, 2.0];
        let k = 3;
        let n = labels.len();
        let p_features = x.ncols();
        let glm = MultinomialLogit::from_labels(labels.view(), k);
        let design = MultiTaskDesign::new(DenseMatrix::new(x), k);
        let beta = Array1::<f64>::zeros(p_features * k);
        let loss = glm.loss(&design, beta.view());
        assert_abs_diff_eq!(loss, (k as f64).ln(), epsilon = 1e-12);
        let _ = n;
    }

    #[test]
    fn surrogate_at_zero_has_z_eq_two_y_minus_one_over_k_and_uniform_half_weights() {
        // β = 0 ⇒ η = 0 ⇒ p_{i,k} = 1/K
        //         g_{i,k} = 1/K − Y_{i,k}
        //         z_{i,k} = 0 − 2 g_{i,k} = 2 (Y_{i,k} − 1/K)
        //         w_{i,k} = 1/2
        // The surrogate's `init_residual` at β=0 returns Xβ − z = -z, and
        // its sample_weights view should be all 0.5.
        use crate::datafit::Datafit;
        let x = array![[1.0, 0.5], [-0.4, 0.3], [0.2, 0.8], [0.7, -0.1]];
        let labels = array![0.0, 1.0, 2.0, 1.0];
        let k = 3;
        let n = labels.len();
        let p_features = x.ncols();
        let glm = MultinomialLogit::from_labels(labels.view(), k);
        let design = MultiTaskDesign::new(DenseMatrix::new(x), k);
        let beta = Array1::<f64>::zeros(p_features * k);
        let surr = glm.surrogate_at(&design, beta.view());

        let r = surr.init_residual(&design, beta.view());
        for task in 0..k {
            for i in 0..n {
                let y_ik = if labels[i] as usize == task { 1.0 } else { 0.0 };
                let expected_z = 2.0 * (y_ik - 1.0 / (k as f64));
                let idx = task * n + i;
                assert_abs_diff_eq!(r[idx], -expected_z, epsilon = 1e-12);
            }
        }
        let w_view = surr.sample_weights().expect("surrogate weights set");
        for &w in w_view.iter() {
            assert_abs_diff_eq!(w, 0.5, epsilon = 1e-12);
        }
    }

    #[test]
    fn softmax_probabilities_sum_to_one_per_sample() {
        // Build a non-trivial η, verify reshape_and_softmax preserves the
        // simplex constraint per row.
        let n = 4;
        let k = 5;
        let eta_in = Array1::from_iter((0..n * k).map(|i| (i as f64) * 0.3 - 1.7));
        let (_eta, _lse, p) = reshape_and_softmax(&eta_in, n, k);
        for i in 0..n {
            let s: f64 = (0..k).map(|kk| p[[i, kk]]).sum();
            assert_abs_diff_eq!(s, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn softmax_handles_extreme_eta_without_overflow() {
        // No NaN with very large positive/negative η.
        let eta_in = Array1::from(vec![1e6, -1e6, 0.0, 1e6, -1e6, 0.0]);
        let (_eta, lse, p) = reshape_and_softmax(&eta_in, 3, 2);
        for i in 0..3 {
            assert!(lse[i].is_finite(), "logsumexp not finite at row {i}");
            for kk in 0..2 {
                assert!(p[[i, kk]].is_finite(), "p[{i},{kk}] not finite");
                assert!(p[[i, kk]] >= 0.0 && p[[i, kk]] <= 1.0);
            }
        }
    }

    /// Build a deterministic 3-class problem: `n` samples, `p` features,
    /// only features 0 and 2 carry signal. Class label = argmax of three
    /// linear scores; small label noise makes the problem non-trivially
    /// regularized. xorshift stream for reproducibility.
    fn multinomial_problem(
        seed: u64,
        n: usize,
        p: usize,
        k: usize,
    ) -> (DenseMatrix, Array1<f64>, ndarray::Array2<f64>) {
        use ndarray::Array2;
        let mut state = seed.max(1);
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        // True B: row-major (n_features × n_classes). Features 0, 2 active.
        let mut true_b = Array2::<f64>::zeros((p, k));
        // Feature 0 splits classes (positive → class 0, negative → class 1).
        true_b[[0, 0]] = 1.5;
        true_b[[0, 1]] = -1.5;
        true_b[[0, 2]] = 0.0;
        // Feature 2 splits class 2 from the rest.
        true_b[[2, 0]] = -0.7;
        true_b[[2, 1]] = -0.7;
        true_b[[2, 2]] = 1.4;
        // Compute η = X B per (i, k); pick argmax with light noise.
        let mut labels = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut best_k = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for kk in 0..k {
                let mut s = 0.0_f64;
                for j in 0..p {
                    s += x[[i, j]] * true_b[[j, kk]];
                }
                s += 0.05 * sample();
                if s > best_v {
                    best_v = s;
                    best_k = kk;
                }
            }
            labels[i] = best_k as f64;
        }
        (DenseMatrix::new(x), labels, true_b)
    }

    /// Row-group L2 norm of a row-major-flattened B at feature j.
    fn row_norm(beta: &Array1<f64>, j: usize, k: usize) -> f64 {
        (0..k)
            .map(|kk| beta[j * k + kk].powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn multinomial_lasso_path_at_lambda_max_returns_zero() {
        use crate::penalty::{GroupLasso, GroupPenalty};
        use crate::solver::{block_lambda_max, prox_newton_block_solve_path, CdConfig};
        let n = 80;
        let p = 5;
        let k = 3;
        let (base, labels, _) = multinomial_problem(11, n, p, k);
        let glm = MultinomialLogit::from_labels(labels.view(), k);
        let design = MultiTaskDesign::new(base, k);
        let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(p, k);
        let base_w = Array1::<f64>::ones(groups.n_groups());

        let beta_zero = Array1::<f64>::zeros(p * k);
        let surr0 = glm.surrogate_at(&design, beta_zero.view());
        let lam_max = block_lambda_max(&design, &surr0, base_w.view(), &groups);

        let base_for_closure = base_w.clone();
        let make_inner = move |_beta: ndarray::ArrayView1<'_, f64>,
                               _groups: &crate::groups::Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            Box::new(GroupLasso::with_weights(lam, base_for_closure.clone()))
                as Box<dyn GroupPenalty>
        };

        let (betas, _) = prox_newton_block_solve_path(
            &design,
            &glm,
            base_w.clone(),
            make_inner,
            &groups,
            1,
            1.0,
            Some(vec![lam_max]),
            &CdConfig {
                max_iter: 200,
                tol: 1e-10,
                acceleration: None,
            },
            10,
            1e-8,
        );
        for j in 0..p * k {
            assert_abs_diff_eq!(betas[[0, j]], 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn multinomial_lasso_recovers_active_features() {
        use crate::penalty::{GroupLasso, GroupPenalty};
        use crate::solver::{prox_newton_block_solve_path, CdConfig};
        let n = 120;
        let p = 6;
        let k = 3;
        let (base_design, labels, _) = multinomial_problem(21, n, p, k);
        let glm = MultinomialLogit::from_labels(labels.view(), k);
        let design = MultiTaskDesign::new(base_design, k);
        let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(p, k);
        let base_w = Array1::<f64>::ones(groups.n_groups());

        let base_for_closure = base_w.clone();
        let make_inner = move |_beta: ndarray::ArrayView1<'_, f64>,
                               _groups: &crate::groups::Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            Box::new(GroupLasso::with_weights(lam, base_for_closure.clone()))
                as Box<dyn GroupPenalty>
        };

        let (betas, report) = prox_newton_block_solve_path(
            &design,
            &glm,
            base_w.clone(),
            make_inner,
            &groups,
            25,
            1e-2,
            None,
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            30,
            1e-7,
        );
        let last = report.lambdas.len() - 1;
        let last_beta = betas.row(last).to_owned();
        // Active rows 0 and 2 should have meaningful row-norms; rows 1, 3, 4, 5 should be tiny.
        assert!(row_norm(&last_beta, 0, k) > 0.3);
        assert!(row_norm(&last_beta, 2, k) > 0.3);
        for j in [1usize, 3, 4, 5] {
            let nrm = row_norm(&last_beta, j, k);
            assert!(
                nrm < row_norm(&last_beta, 0, k),
                "noise row {j} norm {nrm} exceeds active row 0 norm",
            );
        }
    }

    #[test]
    fn multinomial_mcp_via_lla_recovers_active_features() {
        use crate::penalty::{GroupLasso, GroupPenalty};
        use crate::solver::{prox_newton_block_solve_path, surrogate_weights_group_mcp, CdConfig};
        let n = 120;
        let p = 6;
        let k = 3;
        let (base_design, labels, _) = multinomial_problem(22, n, p, k);
        let glm = MultinomialLogit::from_labels(labels.view(), k);
        let design = MultiTaskDesign::new(base_design, k);
        let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(p, k);
        let base_w = Array1::<f64>::ones(groups.n_groups());
        let gamma = 3.0;

        let base_for_closure = base_w.clone();
        let make_inner = move |beta: ndarray::ArrayView1<'_, f64>,
                               g: &crate::groups::Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, base_for_closure.view());
            Box::new(GroupLasso::with_weights(lam, w)) as Box<dyn GroupPenalty>
        };

        let (betas, report) = prox_newton_block_solve_path(
            &design,
            &glm,
            base_w.clone(),
            make_inner,
            &groups,
            25,
            1e-2,
            None,
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            30,
            1e-7,
        );
        let last = report.lambdas.len() - 1;
        let last_beta = betas.row(last).to_owned();
        assert!(row_norm(&last_beta, 0, k) > 0.3);
        assert!(row_norm(&last_beta, 2, k) > 0.3);
    }

    #[test]
    fn multinomial_through_standardized_matches_pre_scaled() {
        use crate::design::Standardized;
        use crate::penalty::{GroupLasso, GroupPenalty};
        use crate::solver::{prox_newton_block_solve_path, CdConfig};
        let n = 80;
        let p = 5;
        let k = 3;
        let (base_design, labels, _) = multinomial_problem(31, n, p, k);
        let x = base_design.view().to_owned();
        let scales = Array1::from(vec![1.5, 0.7, 2.3, 0.9, 1.1]);

        // Pre-scale the user matrix, then wrap in MultiTaskDesign.
        let mut x_scaled = x.clone();
        for j in 0..p {
            for i in 0..n {
                x_scaled[[i, j]] /= scales[j];
            }
        }
        let dense_ref = MultiTaskDesign::new(DenseMatrix::new(x_scaled), k);
        // Lazy: Standardized wraps the raw matrix, then MultiTaskDesign on top.
        // Standardized changes the per-feature view, MultiTaskDesign replicates
        // that across K virtual tasks.
        let std_inner = Standardized::new(DenseMatrix::new(x), scales);
        let std_design = MultiTaskDesign::new(std_inner, k);

        let glm_a = MultinomialLogit::from_labels(labels.view(), k);
        let glm_b = MultinomialLogit::from_labels(labels.view(), k);
        let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(p, k);
        let base_w = Array1::<f64>::ones(groups.n_groups());

        let base_for_closure = base_w.clone();
        let make_inner = move |_beta: ndarray::ArrayView1<'_, f64>,
                               _g: &crate::groups::Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            Box::new(GroupLasso::with_weights(lam, base_for_closure.clone()))
                as Box<dyn GroupPenalty>
        };

        let cd_cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };

        let (betas_ref, _) = prox_newton_block_solve_path(
            &dense_ref,
            &glm_a,
            base_w.clone(),
            &make_inner,
            &groups,
            10,
            1e-2,
            None,
            &cd_cfg,
            20,
            1e-8,
        );
        let (betas_std, _) = prox_newton_block_solve_path(
            &std_design,
            &glm_b,
            base_w.clone(),
            &make_inner,
            &groups,
            10,
            1e-2,
            None,
            &cd_cfg,
            20,
            1e-8,
        );
        assert_eq!(betas_ref.shape(), betas_std.shape());
        for kk in 0..betas_ref.nrows() {
            for j in 0..p * k {
                assert_abs_diff_eq!(betas_ref[[kk, j]], betas_std[[kk, j]], epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn multinomial_with_sparse_inner_matches_dense() {
        use crate::design::SparseCSC;
        use crate::penalty::{GroupLasso, GroupPenalty};
        use crate::solver::{prox_newton_block_solve_path, CdConfig};
        use ndarray::Array2;

        let n = 60;
        let p = 5;
        let k = 3;
        let mut state: u64 = 411;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        // Sparse-ish X: ~50% nonzero entries.
        let mut x_dense = Array2::<f64>::zeros((n, p));
        let mut data: Vec<f64> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        let mut indptr: Vec<usize> = vec![0];
        for j in 0..p {
            for i in 0..n {
                let v = sample();
                if v.abs() > 0.5 {
                    x_dense[[i, j]] = v;
                    data.push(v);
                    indices.push(i);
                }
            }
            indptr.push(data.len());
        }
        // Random-ish labels but deterministic.
        let labels = Array1::from_shape_fn(n, |i| (i % k) as f64);

        let glm_d = MultinomialLogit::from_labels(labels.view(), k);
        let glm_s = MultinomialLogit::from_labels(labels.view(), k);
        let dense = MultiTaskDesign::new(DenseMatrix::new(x_dense), k);
        let csc = SparseCSC::new(
            n,
            Array1::from(data),
            Array1::from(indices),
            Array1::from(indptr),
        );
        let sparse = MultiTaskDesign::new(csc, k);
        let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(p, k);
        let base_w = Array1::<f64>::ones(groups.n_groups());

        let base_for_closure = base_w.clone();
        let make_inner = move |_beta: ndarray::ArrayView1<'_, f64>,
                               _g: &crate::groups::Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            Box::new(GroupLasso::with_weights(lam, base_for_closure.clone()))
                as Box<dyn GroupPenalty>
        };
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };

        let (betas_d, _) = prox_newton_block_solve_path(
            &dense,
            &glm_d,
            base_w.clone(),
            &make_inner,
            &groups,
            6,
            1e-2,
            None,
            &cfg,
            20,
            1e-7,
        );
        let (betas_s, _) = prox_newton_block_solve_path(
            &sparse,
            &glm_s,
            base_w.clone(),
            &make_inner,
            &groups,
            6,
            1e-2,
            None,
            &cfg,
            20,
            1e-7,
        );
        assert_eq!(betas_d.shape(), betas_s.shape());
        for k_lam in 0..betas_d.nrows() {
            for j in 0..p * k {
                assert_abs_diff_eq!(betas_d[[k_lam, j]], betas_s[[k_lam, j]], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn loss_decreases_with_proper_signal() {
        // Loss at the true β should be strictly less than log K (the β=0
        // baseline) on a well-separated 3-class problem.
        let n = 12;
        let p = 2;
        let k = 3;
        let x = array![
            [1.0, 0.0],
            [1.0, 0.1],
            [0.9, -0.1],
            [0.8, 0.2],
            [-0.5, 1.0],
            [-0.6, 0.9],
            [-0.4, 1.1],
            [-0.5, 0.8],
            [-0.5, -1.0],
            [-0.6, -0.9],
            [-0.4, -1.1],
            [-0.5, -0.8],
        ];
        // Class 0: x in upper-right; class 1: x in upper-left; class 2: lower-left.
        let labels = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let glm = MultinomialLogit::from_labels(labels.view(), k);
        let design = MultiTaskDesign::new(DenseMatrix::new(x), k);
        // Strong "true" β: row 0 picks the x-axis, row 1 picks the y-axis,
        // each pulling toward its class.
        // Row-major B[j*K + class] layout.
        // j=0 (x-axis): class 0 +5, class 1 -5, class 2 -5.
        // j=1 (y-axis): class 0  0, class 1 +5, class 2 -5.
        let beta = Array1::from(vec![5.0, -5.0, -5.0, 0.0, 5.0, -5.0]);
        let loss_zero = glm.loss(&design, Array1::<f64>::zeros(p * k).view());
        let loss_true = glm.loss(&design, beta.view());
        assert_abs_diff_eq!(loss_zero, (k as f64).ln(), epsilon = 1e-12);
        assert!(
            loss_true < 0.1 * loss_zero,
            "loss_true={} should be much smaller than loss_zero={}",
            loss_true,
            loss_zero
        );
        let _ = n;
    }
}
