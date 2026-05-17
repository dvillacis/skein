//! Cox proportional-hazards regression via diagonal-IRLS prox-Newton
//! (Breslow or Efron tie handling).
//!
//! `CoxPH` holds right-censored survival outcomes `(time, event)` where
//! `event ∈ {0, 1}` (1 = observed event, 0 = right-censored) and
//! produces the local quadratic surrogate at any β as a
//! [`LeastSquares`] datafit. The Breslow log partial likelihood is
//!
//! ```text
//!     log L(β) = Σ_{k: δ_k=1} [ η_k − log S(t_k) ]
//!     S(t)    = Σ_{j ∈ R(t)} exp(η_j)            R(t) = {j : t_j ≥ t}
//! ```
//!
//! and the per-sample diagonal score / Hessian (dropping the Hessian's
//! off-diagonal terms — this is the classical diagonal-IRLS Cox step,
//! matching what `glmnet` / `ncvreg` use):
//!
//! ```text
//!     CumH(t_i)   = Σ_{k: δ_k=1, t_k ≤ t_i} 1 / S(t_k)
//!     CumH2(t_i)  = Σ_{k: δ_k=1, t_k ≤ t_i} 1 / S(t_k)²
//!     g_i         = −δ_i + exp(η_i) · CumH(t_i)            [-∂(log L)/∂η_i]
//!     w_i         = exp(η_i) · CumH(t_i) − exp(2η_i) · CumH2(t_i)
//!     z_i         = η_i − g_i / w_i
//! ```
//!
//! ## Ties handling
//!
//! Two methods are supported — pick via [`CoxPH::with_ties`] (default
//! [`TieHandling::Breslow`]):
//!
//! - **Breslow.** All `k` events tied at time `t` share the same risk
//!   set `S(t)` and contribute `k/S(t)` to `CumH` and `k/S(t)²` to
//!   `CumH2`. Cheap; mildly biased when ties are heavy.
//! - **Efron.** The `i`-th tied event (0-indexed) sees a reduced risk
//!   set `S_eff_i(t) = S(t) − (i/k) · S_D(t)`, where
//!   `S_D(t) = Σ_{j: tied event} exp(η_j)`. More accurate under heavy
//!   ties; matches R's `survival::coxph(..., ties="efron")` (the R
//!   default) and `glmnet(..., ties="efron")`. Reduces to Breslow
//!   when `k = 1` per block.
//!
//! `η` is clamped to `[-30, 30]` before `exp()` for overflow safety;
//! `w_i` is floored at `1e-6` so the working response stays finite when
//! the diagonal Hessian collapses.
//!
//! No per-sample weights yet (weighted Cox has subtle conventions —
//! frequency vs. probability weighting). No `fit_intercept`: the
//! baseline hazard absorbs any constant, and `S(t)` is invariant to a
//! uniform shift of `η`.

use super::{GlmDatafit, LeastSquares};
use crate::design::DesignMatrix;
use crate::numerics::{ETA_CLAMP, W_FLOOR};
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Tie-handling method for `CoxPH`. See module docs for the math.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TieHandling {
    /// Breslow approximation. All tied events share the same risk set.
    /// Cheaper; matches `glmnet`'s default and the older Cox literature.
    #[default]
    Breslow,
    /// Efron's exact-method approximation. More accurate when ties are
    /// heavy; matches R `survival::coxph`'s default.
    Efron,
}

/// Cox proportional-hazards regression with right-censored outcomes.
///
/// Construct with `(time, event)` — both length `n`, `time ≥ 0` finite,
/// `event ∈ {0, 1}`, and at least one event observed. The constructor
/// precomputes the time-sort permutation once; subsequent
/// `surrogate_at(β)` calls reuse it.
pub struct CoxPH {
    time: Array1<f64>,
    event: Array1<f64>,
    /// Permutation of `0..n` putting samples in ascending time order.
    sort_order: Vec<usize>,
    ties: TieHandling,
}

impl CoxPH {
    /// Construct with the default Breslow tie handling.
    pub fn new(time: Array1<f64>, event: Array1<f64>) -> Self {
        Self::with_ties(time, event, TieHandling::Breslow)
    }

    pub fn with_ties(time: Array1<f64>, event: Array1<f64>, ties: TieHandling) -> Self {
        assert_eq!(
            time.len(),
            event.len(),
            "time and event must have the same length"
        );
        let n = time.len();
        assert!(n > 0, "CoxPH requires at least one sample");
        for i in 0..n {
            let t = time[i];
            assert!(
                t.is_finite() && t >= 0.0,
                "CoxPH requires time ≥ 0 (finite); got {} at index {}",
                t,
                i
            );
            let d = event[i];
            assert!(
                d == 0.0 || d == 1.0,
                "CoxPH requires event ∈ {{0, 1}}; got {} at index {}",
                d,
                i
            );
        }
        let n_events: usize = event.iter().map(|&v| (v > 0.5) as usize).sum();
        assert!(
            n_events >= 1,
            "CoxPH requires at least one event (sample with event = 1)"
        );

        let mut sort_order: Vec<usize> = (0..n).collect();
        sort_order.sort_by(|&a, &b| {
            time[a]
                .partial_cmp(&time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Self {
            time,
            event,
            sort_order,
            ties,
        }
    }

    pub fn time(&self) -> ArrayView1<'_, f64> {
        self.time.view()
    }

    pub fn event(&self) -> ArrayView1<'_, f64> {
        self.event.view()
    }

    pub fn ties(&self) -> TieHandling {
        self.ties
    }

    /// Negative Cox log partial likelihood divided by `n`. Under Breslow
    /// ties this is `(1/n) Σ_{k: δ_k=1} [ log S(t_k) − η_k ]`; under
    /// Efron the per-tie-block log-S contribution becomes
    /// `Σ_{i=0}^{k−1} log(S(t) − (i/k)·S_D(t))` where `S_D(t)` sums
    /// `exp(η)` over tied events.
    pub fn loss(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> f64 {
        let n_f = design.n_samples() as f64;
        let n = design.n_samples();
        let eta = design.matvec(beta);

        let exp_eta_sorted = self.exp_eta_in_sort_order(&eta);
        let s = self.compute_s_per_sample(&exp_eta_sorted);

        let mut total = 0.0_f64;
        let mut i = 0;
        while i < n {
            let block_t = self.time[self.sort_order[i]];
            let block_start = i;
            let mut block_end = i + 1;
            while block_end < n && self.time[self.sort_order[block_end]] == block_t {
                block_end += 1;
            }

            // Sum exp(η) over events in this block, and accumulate −η
            // per event into the NLL.
            let mut s_d = 0.0_f64;
            let mut events_in_block = 0_usize;
            // Index `k` is used into three parallel arrays (sort_order,
            // exp_eta_sorted, event-via-sort_order); zip would obscure that.
            #[allow(clippy::needless_range_loop)]
            for k in block_start..block_end {
                let orig = self.sort_order[k];
                if self.event[orig] > 0.5 {
                    events_in_block += 1;
                    s_d += exp_eta_sorted[k];
                    let eta_k = eta[orig].clamp(-ETA_CLAMP, ETA_CLAMP);
                    total -= eta_k;
                }
            }

            let s_block = s[block_start].max(1e-300);
            match self.ties {
                TieHandling::Breslow => {
                    total += (events_in_block as f64) * s_block.ln();
                }
                TieHandling::Efron => {
                    let k_events = events_in_block as f64;
                    for ev_idx in 0..events_in_block {
                        let frac = ev_idx as f64 / k_events;
                        let s_eff = (s_block - frac * s_d).max(1e-300);
                        total += s_eff.ln();
                    }
                }
            }
            i = block_end;
        }
        total / n_f
    }

    /// Build the diagonal-IRLS weighted-LS surrogate at `β`.
    pub fn surrogate_at(
        &self,
        design: &dyn DesignMatrix,
        beta: ArrayView1<'_, f64>,
    ) -> LeastSquares {
        let n = design.n_samples();
        let eta = design.matvec(beta);
        let exp_eta_sorted = self.exp_eta_in_sort_order(&eta);
        let s = self.compute_s_per_sample(&exp_eta_sorted);
        let (cum_h, cum_h2) = self.compute_cum_h(&s, &exp_eta_sorted);

        let mut w = Array1::<f64>::zeros(n);
        let mut z = Array1::<f64>::zeros(n);
        for (k, &orig) in self.sort_order.iter().enumerate() {
            let eta_k = eta[orig].clamp(-ETA_CLAMP, ETA_CLAMP);
            let exp_eta_k = exp_eta_sorted[k];
            let cum_h_k = cum_h[k];
            let cum_h2_k = cum_h2[k];
            let event_k = self.event[orig];

            let w_raw = exp_eta_k * cum_h_k - exp_eta_k * exp_eta_k * cum_h2_k;
            let w_floored = w_raw.max(W_FLOOR);
            let g_raw = -event_k + exp_eta_k * cum_h_k;

            w[orig] = w_floored;
            z[orig] = eta_k - g_raw / w_floored;
        }
        LeastSquares::with_sample_weights(z, w)
    }

    /// `exp(η_i)` for each i, returned in time-sorted order.
    fn exp_eta_in_sort_order(&self, eta: &Array1<f64>) -> Vec<f64> {
        self.sort_order
            .iter()
            .map(|&orig| eta[orig].clamp(-ETA_CLAMP, ETA_CLAMP).exp())
            .collect()
    }

    /// `S(t_i) = Σ_{j ∈ R(t_i)} exp(η_j)` per sample, in time-sorted
    /// order. Tied times share the same `S` value (Breslow risk set).
    fn compute_s_per_sample(&self, exp_eta_sorted: &[f64]) -> Vec<f64> {
        let n = exp_eta_sorted.len();
        let mut s = vec![0.0_f64; n];
        let mut cum = 0.0_f64;

        // Walk backward through tie-blocks; for each block, add this
        // block's contributions to `cum`, then assign that running sum
        // as `S` for every sample in the block.
        let mut i = n;
        while i > 0 {
            let block_end = i; // exclusive
            let block_t = self.time[self.sort_order[i - 1]];
            let mut block_start = i - 1;
            while block_start > 0 && self.time[self.sort_order[block_start - 1]] == block_t {
                block_start -= 1;
            }
            cum += exp_eta_sorted[block_start..block_end].iter().sum::<f64>();
            s[block_start..block_end].fill(cum);
            i = block_start;
        }
        s
    }

    /// `CumH(t_i)` and `CumH2(t_i)` per sample, in time-sorted order.
    /// Walks forward through tie-blocks. Under Breslow each block's `k`
    /// events all share `S(t)` and contribute `k/S(t)` and `k/S(t)²`;
    /// under Efron the `i`-th event sees a reduced risk set
    /// `S_eff_i = S(t) − (i/k) · S_D(t)`, where
    /// `S_D(t) = Σ_{j ∈ block, δ_j=1} exp(η_j)`.
    fn compute_cum_h(&self, s_per_sample: &[f64], exp_eta_sorted: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = s_per_sample.len();
        let mut cum_h = vec![0.0_f64; n];
        let mut cum_h2 = vec![0.0_f64; n];
        let mut running_h = 0.0_f64;
        let mut running_h2 = 0.0_f64;

        let mut i = 0;
        while i < n {
            let block_t = self.time[self.sort_order[i]];
            let block_start = i;
            let mut block_end = i + 1;
            while block_end < n && self.time[self.sort_order[block_end]] == block_t {
                block_end += 1;
            }

            let mut events_in_block = 0_usize;
            let mut s_d = 0.0_f64;
            // See the matching loop in `loss` — `k` indexes both sort_order
            // (then event) and exp_eta_sorted in parallel.
            #[allow(clippy::needless_range_loop)]
            for k in block_start..block_end {
                if self.event[self.sort_order[k]] > 0.5 {
                    events_in_block += 1;
                    s_d += exp_eta_sorted[k];
                }
            }
            let s_block = s_per_sample[block_start].max(1e-300);
            match self.ties {
                TieHandling::Breslow => {
                    running_h += events_in_block as f64 / s_block;
                    running_h2 += events_in_block as f64 / (s_block * s_block);
                }
                TieHandling::Efron => {
                    let k_events = events_in_block as f64;
                    for ev_idx in 0..events_in_block {
                        let frac = ev_idx as f64 / k_events;
                        let s_eff = (s_block - frac * s_d).max(1e-300);
                        running_h += 1.0 / s_eff;
                        running_h2 += 1.0 / (s_eff * s_eff);
                    }
                }
            }

            for k in block_start..block_end {
                cum_h[k] = running_h;
                cum_h2[k] = running_h2;
            }
            i = block_end;
        }
        (cum_h, cum_h2)
    }
}

impl GlmDatafit for CoxPH {
    fn surrogate_at(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> LeastSquares {
        CoxPH::surrogate_at(self, design, beta)
    }

    fn loss(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> f64 {
        CoxPH::loss(self, design, beta)
    }

    // Intentionally inherit the `None` defaults for `glm_per_sample_loss_grad`
    // and `glm_dual_obj`. Cox's partial-likelihood dual under Breslow/Efron
    // tie handling has no closed form analogous to logistic/Poisson — the
    // risk-set structure couples samples, so the conjugate doesn't decouple
    // per-sample. Wu & Lange (2008) sketch a constrained dual but it's not
    // a single-shot evaluation. Gap-safe screening for Cox is therefore
    // deferred; the prox-Newton outer loop falls back to KKT-only
    // termination for this GLM.

    fn refresh_surrogate_components(
        &self,
        eta: ArrayView1<'_, f64>,
        mut w_out: ArrayViewMut1<'_, f64>,
        mut r_out: ArrayViewMut1<'_, f64>,
    ) {
        // Mirrors `CoxPH::surrogate_at` minus the matvec; the caller's
        // fused solver maintains `eta = X·β` incrementally. Cox doesn't
        // expose user `sample_weights`, so `w_out[i] = w_floored` and
        // `r_out[i] = -g_raw_i / w_floored_i` (the surrogate's
        // `z_i − η_i` working residual).
        let n = eta.len();
        debug_assert_eq!(w_out.len(), n);
        debug_assert_eq!(r_out.len(), n);

        // Snapshot eta into an owned Array so the helpers (which take
        // `&Array1<f64>`) can call it.
        let eta_owned = eta.to_owned();
        let exp_eta_sorted = self.exp_eta_in_sort_order(&eta_owned);
        let s = self.compute_s_per_sample(&exp_eta_sorted);
        let (cum_h, cum_h2) = self.compute_cum_h(&s, &exp_eta_sorted);

        for (k, &orig) in self.sort_order.iter().enumerate() {
            let exp_eta_k = exp_eta_sorted[k];
            let cum_h_k = cum_h[k];
            let cum_h2_k = cum_h2[k];
            let event_k = self.event[orig];

            let w_raw = exp_eta_k * cum_h_k - exp_eta_k * exp_eta_k * cum_h2_k;
            let w_floored = w_raw.max(W_FLOOR);
            let g_raw = -event_k + exp_eta_k * cum_h_k;

            w_out[orig] = w_floored;
            r_out[orig] = -g_raw / w_floored;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::Datafit;
    use crate::design::DenseMatrix;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Tiny 3-sample example with ascending unique times — values
    /// hand-derivable from the Breslow formulas:
    ///
    /// - Sample 0: t=1, δ=1
    /// - Sample 1: t=2, δ=1
    /// - Sample 2: t=3, δ=0
    ///
    /// At β=0, exp(η)=1 everywhere, so S(1)=3, S(2)=2, S(3)=1.
    fn tiny_problem() -> (DenseMatrix, CoxPH) {
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]];
        let time = array![1.0, 2.0, 3.0];
        let event = array![1.0, 1.0, 0.0];
        (DenseMatrix::new(x), CoxPH::new(time, event))
    }

    #[test]
    fn cox_loss_at_zero_matches_log_six_over_three() {
        // log L = δ_0 (0 - log 3) + δ_1 (0 - log 2) + δ_2 (0 - log 1)
        //       = -log 3 - log 2 = -log 6
        // NLL = log 6 ; divided by n=3.
        let (design, glm) = tiny_problem();
        let beta = Array1::<f64>::zeros(2);
        let loss = glm.loss(&design, beta.view());
        let expected = (6.0_f64).ln() / 3.0;
        assert_abs_diff_eq!(loss, expected, epsilon = 1e-12);
    }

    #[test]
    fn cox_surrogate_at_zero_diagonal_weights_and_residual() {
        // CumH:  sample 0 = 1/3
        //        sample 1 = 1/3 + 1/2 = 5/6
        //        sample 2 = 5/6
        // CumH2: sample 0 = 1/9
        //        sample 1 = 1/9 + 1/4 = 13/36
        //        sample 2 = 13/36
        // w_i = exp(η)·CumH − exp(2η)·CumH2  (η=0)
        //      sample 0: 1/3 - 1/9 = 2/9
        //      sample 1: 5/6 - 13/36 = 17/36
        //      sample 2: 5/6 - 13/36 = 17/36
        // g_i = -δ + exp(η)·CumH
        //      sample 0: -1 + 1/3 = -2/3
        //      sample 1: -1 + 5/6 = -1/6
        //      sample 2:  0 + 5/6 =  5/6
        // z_i = η − g/w  (η=0)
        //      sample 0: 0 - (-2/3) / (2/9) = 3
        //      sample 1: 0 - (-1/6) / (17/36) = 36/(6·17) = 6/17
        //      sample 2: 0 - (5/6) / (17/36) = -30/17
        let (design, glm) = tiny_problem();
        let beta = Array1::<f64>::zeros(2);
        let surr = glm.surrogate_at(&design, beta.view());
        // The surrogate stores `z`. init_residual at β=0 = Xβ − z = -z.
        let r = surr.init_residual(&design, beta.view());
        assert_abs_diff_eq!(r[0], -3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[1], -6.0 / 17.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[2], 30.0 / 17.0, epsilon = 1e-12);
    }

    #[test]
    fn cox_surrogate_with_ties_shares_risk_set() {
        // Two tied uncensored samples + one censored ⇒ Breslow risk set
        // is the same for both events at the tied time.
        // - Sample 0: t=1, δ=1
        // - Sample 1: t=1, δ=1   (tied with sample 0)
        // - Sample 2: t=2, δ=0
        // At β=0: S(1) = 3 (all three at risk), S(2) = 1.
        // Two events at time 1 share S=3, so CumH at t=1 = 2/3,
        // CumH2 at t=1 = 2/9. Sample 2 (no event) has CumH(2)=2/3, CumH2=2/9.
        // (No event at t=2.)
        // w_0 = w_1 = w_2 = 1·(2/3) − 1·(2/9) = 6/9 − 2/9 = 4/9.
        // g_0 = g_1 = -1 + 2/3 = -1/3; g_2 = 0 + 2/3 = 2/3.
        // z_0 = z_1 = -(-1/3)/(4/9) = (1/3)·(9/4) = 3/4.
        // z_2 = -(2/3)/(4/9) = -(2/3)·(9/4) = -3/2.
        let x = array![[1.0], [1.0], [1.0]];
        let time = array![1.0, 1.0, 2.0];
        let event = array![1.0, 1.0, 0.0];
        let design = DenseMatrix::new(x);
        let glm = CoxPH::new(time, event);
        let beta = Array1::<f64>::zeros(1);
        let surr = glm.surrogate_at(&design, beta.view());
        let r = surr.init_residual(&design, beta.view());
        assert_abs_diff_eq!(r[0], -3.0 / 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[1], -3.0 / 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[2], 3.0 / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn cox_handles_extreme_eta_without_overflow() {
        let (design, glm) = tiny_problem();
        let beta = array![100.0, 100.0]; // η huge before clamp
        let loss = glm.loss(&design, beta.view());
        assert!(loss.is_finite(), "loss should be finite, got {}", loss);
        let surr = glm.surrogate_at(&design, beta.view());
        let r = surr.init_residual(&design, beta.view());
        for i in 0..3 {
            assert!(r[i].is_finite(), "residual finite at i={}", i);
        }
    }

    #[test]
    #[should_panic(expected = "at least one event")]
    fn cox_panics_on_all_censored() {
        let _ = CoxPH::new(array![1.0, 2.0], array![0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "event ∈")]
    fn cox_panics_on_non_binary_event() {
        let _ = CoxPH::new(array![1.0, 2.0], array![1.0, 0.5]);
    }

    #[test]
    #[should_panic(expected = "time ≥ 0")]
    fn cox_panics_on_negative_time() {
        let _ = CoxPH::new(array![1.0, -1.0], array![1.0, 1.0]);
    }

    #[test]
    fn cox_efron_with_unique_times_matches_breslow() {
        // No ties ⇒ Efron's per-event reduced risk set has only one
        // term per block (i=0 ⇒ s_eff = S − 0 = S), reducing to Breslow.
        // Loss + surrogate must match exactly.
        let (design, _glm_b) = tiny_problem();
        let breslow = CoxPH::with_ties(
            array![1.0, 2.0, 3.0],
            array![1.0, 1.0, 0.0],
            TieHandling::Breslow,
        );
        let efron = CoxPH::with_ties(
            array![1.0, 2.0, 3.0],
            array![1.0, 1.0, 0.0],
            TieHandling::Efron,
        );
        let beta = array![0.3, -0.2];
        assert_abs_diff_eq!(
            breslow.loss(&design, beta.view()),
            efron.loss(&design, beta.view()),
            epsilon = 1e-12
        );
        let s_b = breslow.surrogate_at(&design, beta.view());
        let s_e = efron.surrogate_at(&design, beta.view());
        let r_b = s_b.init_residual(&design, beta.view());
        let r_e = s_e.init_residual(&design, beta.view());
        for i in 0..3 {
            assert_abs_diff_eq!(r_b[i], r_e[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn cox_efron_with_two_tied_events_uses_reduced_risk_set() {
        // Two tied events at t=1, one censored at t=2. At β=0:
        // S(1) = 3, S_D(1) = 2 (two tied events with exp(η)=1 each).
        // Efron contributions to running_h at t=1:
        //   i=0: 1/(3 − 0·2/2) = 1/3
        //   i=1: 1/(3 − 1·2/2) = 1/(3−1) = 1/2
        // Running CumH at t=1 = 1/3 + 1/2 = 5/6
        // Running CumH2 at t=1 = 1/9 + 1/4 = 13/36
        // (Breslow would give 2/3 and 2/9; Efron > Breslow because the
        // i=1 event sees a smaller risk set.)
        // Censored sample at t=2 inherits the same running CumH/CumH2.
        // w_i(η=0) = CumH − CumH2:
        //   sample 0: 5/6 − 13/36 = 30/36 − 13/36 = 17/36
        //   sample 1: 17/36
        //   sample 2: 17/36
        // g_i = -δ + CumH:
        //   sample 0: -1 + 5/6 = -1/6
        //   sample 1: -1/6
        //   sample 2:  0 + 5/6 = 5/6
        // z_i = -g/w:
        //   sample 0: -(-1/6)/(17/36) = 6/17
        //   sample 1: 6/17
        //   sample 2: -(5/6)/(17/36) = -30/17
        let x = array![[1.0], [1.0], [1.0]];
        let time = array![1.0, 1.0, 2.0];
        let event = array![1.0, 1.0, 0.0];
        let design = DenseMatrix::new(x);
        let glm = CoxPH::with_ties(time, event, TieHandling::Efron);
        let beta = Array1::<f64>::zeros(1);
        let surr = glm.surrogate_at(&design, beta.view());
        let r = surr.init_residual(&design, beta.view());
        assert_abs_diff_eq!(r[0], -6.0 / 17.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[1], -6.0 / 17.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[2], 30.0 / 17.0, epsilon = 1e-12);
    }

    #[test]
    fn cox_efron_loss_with_two_tied_events_matches_hand_derivation() {
        // At β=0: NLL = -η_0 - η_1 + log(3) + log(3 − 1)
        //              = 0 + 0 + log 3 + log 2 = log 6.
        // Wait — that's the same as Breslow at β=0 since events_in_block
        // · log S_block = 2 log 3 = log 9 vs Efron's log 3 + log 2 = log 6.
        // So NLL = log 6 / n. (Compared to Breslow's log 9 / n.)
        let x = array![[1.0], [1.0], [1.0]];
        let time = array![1.0, 1.0, 2.0];
        let event = array![1.0, 1.0, 0.0];
        let design = DenseMatrix::new(x);

        let breslow = CoxPH::with_ties(time.clone(), event.clone(), TieHandling::Breslow);
        let efron = CoxPH::with_ties(time, event, TieHandling::Efron);
        let beta = Array1::<f64>::zeros(1);
        let n_f = 3.0_f64;
        assert_abs_diff_eq!(
            breslow.loss(&design, beta.view()),
            (9.0_f64).ln() / n_f,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            efron.loss(&design, beta.view()),
            (6.0_f64).ln() / n_f,
            epsilon = 1e-12
        );
    }

    #[test]
    fn cox_default_constructor_uses_breslow() {
        let (design, glm) = tiny_problem();
        assert_eq!(glm.ties(), TieHandling::Breslow);
        let beta = array![0.3, -0.2];
        let glm_b = CoxPH::with_ties(
            array![1.0, 2.0, 3.0],
            array![1.0, 1.0, 0.0],
            TieHandling::Breslow,
        );
        assert_abs_diff_eq!(
            glm.loss(&design, beta.view()),
            glm_b.loss(&design, beta.view()),
            epsilon = 1e-12
        );
    }
}
