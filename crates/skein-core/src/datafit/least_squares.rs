use super::Datafit;
use crate::design::DesignMatrix;
use ndarray::{Array1, ArrayView1};

/// Least-squares loss `(1/2n) Σ w_i (Xβ_i − y_i)²` (uniform `w_i = 1` by
/// default; per-sample `w` honored throughout the trait when present —
/// `value`, `coord_grad`, `full_grad`, and `coord_lipschitz` all carry
/// the weight through).
pub struct LeastSquares {
    y: Array1<f64>,
    sample_weights: Option<Array1<f64>>,
}

impl LeastSquares {
    pub fn new(y: Array1<f64>) -> Self {
        Self {
            y,
            sample_weights: None,
        }
    }

    pub fn with_sample_weights(y: Array1<f64>, w: Array1<f64>) -> Self {
        assert_eq!(
            y.len(),
            w.len(),
            "sample_weights length must equal y length"
        );
        Self {
            y,
            sample_weights: Some(w),
        }
    }

    pub fn y(&self) -> ArrayView1<'_, f64> {
        self.y.view()
    }
}

impl Datafit for LeastSquares {
    fn value(&self, residual: ArrayView1<'_, f64>) -> f64 {
        let n = residual.len() as f64;
        match &self.sample_weights {
            None => 0.5 * residual.dot(&residual) / n,
            Some(w) => {
                let mut s = 0.0_f64;
                for i in 0..residual.len() {
                    s += w[i] * residual[i] * residual[i];
                }
                0.5 * s / n
            }
        }
    }

    fn init_residual(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut r = design.matvec(beta);
        r -= &self.y;
        r
    }

    fn coord_grad(
        &self,
        design: &dyn DesignMatrix,
        j: usize,
        residual: ArrayView1<'_, f64>,
    ) -> f64 {
        let n = design.n_samples() as f64;
        match &self.sample_weights {
            None => design.col_dot(j, residual) / n,
            Some(w) => {
                // (1/n) Σ w_i x_ij r_i — express as a column dot with a
                // weighted residual so we still ride the design's
                // `col_dot` fast path.
                let weighted: Array1<f64> =
                    (0..residual.len()).map(|i| w[i] * residual[i]).collect();
                design.col_dot(j, weighted.view()) / n
            }
        }
    }

    fn full_grad(&self, design: &dyn DesignMatrix, residual: ArrayView1<'_, f64>) -> Array1<f64> {
        let n = design.n_samples() as f64;
        match &self.sample_weights {
            None => &design.rmatvec(residual) / n,
            Some(w) => {
                let weighted: Array1<f64> =
                    (0..residual.len()).map(|i| w[i] * residual[i]).collect();
                &design.rmatvec(weighted.view()) / n
            }
        }
    }

    fn coord_lipschitz(&self, design: &dyn DesignMatrix, j: usize) -> f64 {
        let n = design.n_samples() as f64;
        match &self.sample_weights {
            None => design.col_sq_norm(j) / n,
            Some(w) => {
                // (1/n) Σ w_i x_ij² — read column j explicitly since the
                // `DesignMatrix` trait doesn't expose a weighted-norm
                // helper.
                let mut s = 0.0_f64;
                let col = design.columns(&[j]);
                for i in 0..design.n_samples() {
                    let v = col[[i, 0]];
                    s += w[i] * v * v;
                }
                s / n
            }
        }
    }

    fn sample_weights(&self) -> Option<ArrayView1<'_, f64>> {
        self.sample_weights.as_ref().map(|w| w.view())
    }

    fn lasso_dual_obj(
        &self,
        design: &dyn DesignMatrix,
        beta: ArrayView1<'_, f64>,
        residual: ArrayView1<'_, f64>,
        grad: ArrayView1<'_, f64>,
        scale: f64,
    ) -> Option<f64> {
        // Same formula for unweighted and weighted LS — only `r_sq` (the
        // residual energy term) picks up the diagonal weight. Derivation:
        // for `f(z) = (1/2n) zᵀ W z` the Fenchel conjugate is
        // `f*(θ) = (n/2) Σ θᵢ²/wᵢ` (W = diag(w), wᵢ > 0). The natural
        // dual point is `θ_naive = −(1/n) W r`, at which `f*(θ_naive) =
        // (1/2n) Σ wᵢ rᵢ²` and the rest collapses after eliminating `y`
        // via `Xβ = r + y` exactly as in the unweighted case. So the
        // closed form is the unweighted formula with `‖r‖²` replaced by
        // `Σ wᵢ rᵢ²`. The supplied `grad` must be the matching weighted
        // gradient (`(1/n) Σ wᵢ xᵢⱼ rᵢ`) — `full_grad` already returns
        // that.
        let n = design.n_samples() as f64;
        let r_sq: f64 = match &self.sample_weights {
            None => residual.dot(&residual),
            Some(w) => {
                let mut s = 0.0_f64;
                for i in 0..residual.len() {
                    s += w[i] * residual[i] * residual[i];
                }
                s
            }
        };
        let beta_dot_grad: f64 = beta.dot(&grad);
        Some((r_sq / n) * scale * (1.0 - 0.5 * scale) - scale * beta_dot_grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::DenseMatrix;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn small_design() -> DenseMatrix {
        // 3 × 2 design with non-trivial column norms.
        DenseMatrix::new(array![[1.0_f64, -1.0], [2.0, 0.0], [-1.0, 3.0],])
    }

    #[test]
    fn value_unweighted_is_half_mean_squared_residual() {
        let y = array![1.0_f64, 2.0, 3.0];
        let df = LeastSquares::new(y.clone());
        // r = (1, -1, 2); ‖r‖² = 6; (1/(2n))·6 = 1.0
        let r = array![1.0_f64, -1.0, 2.0];
        assert_abs_diff_eq!(df.value(r.view()), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn value_weighted_uses_supplied_weights() {
        let y = array![0.0_f64, 0.0, 0.0];
        let w = array![2.0_f64, 1.0, 0.0];
        let df = LeastSquares::with_sample_weights(y, w);
        let r = array![1.0_f64, 1.0, 100.0];
        // Σ w_i r_i² = 2·1 + 1·1 + 0·10000 = 3
        // value = (1/(2·3))·3 = 0.5
        assert_abs_diff_eq!(df.value(r.view()), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn init_residual_returns_x_beta_minus_y() {
        let design = small_design();
        let y = array![0.5_f64, -1.0, 2.0];
        let df = LeastSquares::new(y);
        let beta = array![1.0_f64, 1.0];
        // Xβ = (0, 2, 2); residual = Xβ − y = (-0.5, 3, 0)
        let r = df.init_residual(&design, beta.view());
        assert_abs_diff_eq!(r[0], -0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(r[1], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn coord_grad_unweighted_matches_xj_dot_r_over_n() {
        let design = small_design();
        let y = array![0.0_f64, 0.0, 0.0];
        let df = LeastSquares::new(y);
        let r = array![1.0_f64, 2.0, -1.0];
        // Column 0: (1, 2, -1) · (1, 2, -1) = 1 + 4 + 1 = 6 / 3 = 2
        // Column 1: (-1, 0, 3) · (1, 2, -1) = -1 + 0 − 3 = -4 / 3
        assert_abs_diff_eq!(df.coord_grad(&design, 0, r.view()), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            df.coord_grad(&design, 1, r.view()),
            -4.0 / 3.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn coord_grad_weighted_matches_weighted_inner_product() {
        let design = small_design();
        let y = array![0.0_f64, 0.0, 0.0];
        let w = array![1.0_f64, 0.0, 2.0];
        let df = LeastSquares::with_sample_weights(y, w);
        let r = array![1.0_f64, 2.0, -1.0];
        // Column 0: w·r = (1, 0, -2); (1, 2, -1) · (1, 0, -2) = 1 + 0 + 2 = 3 / 3 = 1
        // Column 1: (-1, 0, 3) · (1, 0, -2) = -1 + 0 − 6 = -7 / 3
        assert_abs_diff_eq!(df.coord_grad(&design, 0, r.view()), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            df.coord_grad(&design, 1, r.view()),
            -7.0 / 3.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn full_grad_matches_per_coord_grad_loop_unweighted() {
        let design = small_design();
        let y = array![0.5_f64, -1.0, 2.0];
        let df = LeastSquares::new(y);
        let r = array![1.0_f64, 2.0, -1.0];
        let g = df.full_grad(&design, r.view());
        for j in 0..design.n_features() {
            assert_abs_diff_eq!(g[j], df.coord_grad(&design, j, r.view()), epsilon = 1e-12);
        }
    }

    #[test]
    fn full_grad_matches_per_coord_grad_loop_weighted() {
        let design = small_design();
        let y = array![0.5_f64, -1.0, 2.0];
        let w = array![1.5_f64, 0.5, 1.0];
        let df = LeastSquares::with_sample_weights(y, w);
        let r = array![1.0_f64, 2.0, -1.0];
        let g = df.full_grad(&design, r.view());
        for j in 0..design.n_features() {
            assert_abs_diff_eq!(g[j], df.coord_grad(&design, j, r.view()), epsilon = 1e-12);
        }
    }

    #[test]
    fn coord_lipschitz_unweighted_is_col_sq_norm_over_n() {
        let design = small_design();
        let y = array![0.0_f64, 0.0, 0.0];
        let df = LeastSquares::new(y);
        // ‖col_0‖² = 1+4+1 = 6 → 2; ‖col_1‖² = 1+0+9 = 10 → 10/3
        assert_abs_diff_eq!(df.coord_lipschitz(&design, 0), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(df.coord_lipschitz(&design, 1), 10.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn coord_lipschitz_weighted_is_weighted_col_sq_norm_over_n() {
        let design = small_design();
        let y = array![0.0_f64, 0.0, 0.0];
        let w = array![1.0_f64, 0.0, 2.0];
        let df = LeastSquares::with_sample_weights(y, w);
        // col 0: 1·1 + 0·4 + 2·1 = 3 → 1.0
        // col 1: 1·1 + 0·0 + 2·9 = 19 → 19/3
        assert_abs_diff_eq!(df.coord_lipschitz(&design, 0), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(df.coord_lipschitz(&design, 1), 19.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn sample_weights_accessor_round_trips() {
        let y = array![0.0_f64, 0.0, 0.0];
        let df_none = LeastSquares::new(y.clone());
        assert!(df_none.sample_weights().is_none());

        let w = array![0.5_f64, 1.5, 2.0];
        let df_w = LeastSquares::with_sample_weights(y, w.clone());
        let view = df_w.sample_weights().expect("weights must be present");
        for i in 0..3 {
            assert_abs_diff_eq!(view[i], w[i]);
        }
    }

    #[test]
    fn lasso_dual_obj_unweighted_matches_closed_form() {
        let design = small_design();
        let y = array![0.5_f64, -1.0, 2.0];
        let df = LeastSquares::new(y);
        let beta = array![0.3_f64, -0.2];
        let r = df.init_residual(&design, beta.view());
        let g = df.full_grad(&design, r.view());
        let scale = 0.7;
        let n = design.n_samples() as f64;
        let r_sq = r.dot(&r);
        let bg = beta.dot(&g);
        let expected = (r_sq / n) * scale * (1.0 - 0.5 * scale) - scale * bg;
        let actual = df
            .lasso_dual_obj(&design, beta.view(), r.view(), g.view(), scale)
            .expect("unweighted LS must return a closed-form dual");
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
    }

    #[test]
    fn lasso_dual_obj_weighted_matches_closed_form() {
        // Derivation: D(θ_scaled) = (Σwᵢrᵢ²/n)·scale·(1−scale/2) − scale·βᵀg,
        // with `grad` = (1/n) Σ wᵢ xᵢⱼ rᵢ (which is what `full_grad`
        // returns under `with_sample_weights`).
        let design = small_design();
        let y = array![0.5_f64, -1.0, 2.0];
        let w = array![1.5_f64, 0.5, 1.0];
        let df = LeastSquares::with_sample_weights(y, w.clone());
        let beta = array![0.3_f64, -0.2];
        let r = df.init_residual(&design, beta.view());
        let g = df.full_grad(&design, r.view());
        let scale = 0.7;
        let n = design.n_samples() as f64;
        let wr_sq: f64 = (0..r.len()).map(|i| w[i] * r[i] * r[i]).sum();
        let bg = beta.dot(&g);
        let expected = (wr_sq / n) * scale * (1.0 - 0.5 * scale) - scale * bg;
        let actual = df
            .lasso_dual_obj(&design, beta.view(), r.view(), g.view(), scale)
            .expect("weighted LS must return a closed-form dual");
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
    }

    #[test]
    fn lasso_dual_obj_weighted_gap_nonnegative_at_lambda_max() {
        // Pin the dual ≤ primal contract: at β = 0 (so r = −y) and
        // λ = max_j |grad_j|/w_j, the gap should be ≥ 0 and small (it
        // collapses to exactly the L1 envelope's slack — but for weighted
        // LS we don't have a closed-form gap = 0, so just assert
        // non-negativity, which is the dual-feasibility contract).
        let design = small_design();
        let y = array![1.0_f64, -2.0, 0.5];
        let w = array![2.0_f64, 1.0, 0.5];
        let df = LeastSquares::with_sample_weights(y, w.clone());
        let beta = array![0.0_f64, 0.0];
        let r = df.init_residual(&design, beta.view());
        let g = df.full_grad(&design, r.view());
        let lam = g.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let lambda_bound = g.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let scale = if lambda_bound > lam {
            lam / lambda_bound
        } else {
            1.0
        };
        let dual = df
            .lasso_dual_obj(&design, beta.view(), r.view(), g.view(), scale)
            .expect("weighted LS must return a closed-form dual");
        let primal = df.value(r.view()); // R(0) = 0
        assert!(
            primal - dual >= -1e-12,
            "gap must be non-negative (primal={} dual={})",
            primal,
            dual
        );
    }

    #[test]
    fn weights_length_mismatch_panics() {
        let y = array![0.0_f64, 0.0, 0.0];
        let w = array![1.0_f64, 1.0]; // wrong length
        let result = std::panic::catch_unwind(|| LeastSquares::with_sample_weights(y, w));
        assert!(
            result.is_err(),
            "constructor must reject mismatched weights"
        );
    }

    #[test]
    fn y_accessor_returns_supplied_target() {
        let y = array![1.0_f64, 2.0, 3.0];
        let df = LeastSquares::new(y.clone());
        let view = df.y();
        assert_eq!(view.len(), 3);
        for i in 0..3 {
            assert_abs_diff_eq!(view[i], y[i]);
        }
    }
}
