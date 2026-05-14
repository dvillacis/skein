//! Group structure for structured sparsity.
//!
//! Stored as a CSR-like (ptr, idx) pair so that overlapping and unequal-size
//! groups are both representable. `ptr` has length `n_groups + 1`.

use crate::{Result, SkeinError};

#[derive(Debug, Clone)]
pub struct Groups {
    ptr: Vec<usize>,
    idx: Vec<usize>,
}

impl Groups {
    pub fn from_csr(ptr: Vec<usize>, idx: Vec<usize>) -> Result<Self> {
        if ptr.is_empty() {
            return Err(SkeinError::InvalidParameter(
                "ptr must contain at least one element".into(),
            ));
        }
        if *ptr.last().unwrap() != idx.len() {
            return Err(SkeinError::InvalidParameter(format!(
                "ptr.last() = {} does not match idx.len() = {}",
                ptr.last().unwrap(),
                idx.len()
            )));
        }
        Ok(Self { ptr, idx })
    }

    /// Each feature is its own group (recovers element-wise sparsity).
    pub fn singletons(n_features: usize) -> Self {
        let ptr = (0..=n_features).collect();
        let idx = (0..n_features).collect();
        Self { ptr, idx }
    }

    /// Equal-size contiguous groups of size `g`. Last group may be smaller.
    pub fn contiguous_blocks(n_features: usize, group_size: usize) -> Self {
        let mut ptr = vec![0];
        let mut idx = Vec::with_capacity(n_features);
        let mut start = 0;
        while start < n_features {
            let end = (start + group_size).min(n_features);
            for j in start..end {
                idx.push(j);
            }
            ptr.push(idx.len());
            start = end;
        }
        Self { ptr, idx }
    }

    pub fn n_groups(&self) -> usize {
        self.ptr.len() - 1
    }

    pub fn group(&self, g: usize) -> &[usize] {
        &self.idx[self.ptr[g]..self.ptr[g + 1]]
    }

    /// `true` if any feature index appears in more than one group.
    ///
    /// Disjoint groups are the assumption baked into the per-group
    /// operator-norm Lipschitz used by both serial and Jacobi-parallel
    /// block-CD; overlapping groups break that analysis (and silently
    /// corrupt Jacobi updates, where two threads compute against the
    /// same snapshot β and then both write to a shared coordinate).
    /// The parallel block-CD entry point (`block_cd_solve_subset_parallel`)
    /// uses this check to fall back to serial Gauss-Seidel — see the
    /// note there.
    ///
    /// O(`idx.len()` + `max_idx`) — single pass with a bitset sized to
    /// the largest column index referenced.
    pub fn has_overlap(&self) -> bool {
        if self.idx.is_empty() {
            return false;
        }
        let max_idx = *self.idx.iter().max().expect("non-empty checked above");
        let mut seen = vec![false; max_idx + 1];
        for &j in &self.idx {
            if seen[j] {
                return true;
            }
            seen[j] = true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_overlap_singletons_is_false() {
        let g = Groups::singletons(10);
        assert!(!g.has_overlap());
    }

    #[test]
    fn has_overlap_contiguous_blocks_is_false() {
        let g = Groups::contiguous_blocks(13, 4);
        assert!(!g.has_overlap());
    }

    #[test]
    fn has_overlap_disjoint_csr_is_false() {
        // Two groups: {0,2,4} and {1,3}. No shared index.
        let g = Groups::from_csr(vec![0, 3, 5], vec![0, 2, 4, 1, 3]).unwrap();
        assert!(!g.has_overlap());
    }

    #[test]
    fn has_overlap_shared_index_is_true() {
        // Two groups: {0,1,2} and {2,3}. Index 2 is in both.
        let g = Groups::from_csr(vec![0, 3, 5], vec![0, 1, 2, 2, 3]).unwrap();
        assert!(g.has_overlap());
    }

    #[test]
    fn has_overlap_repeated_within_single_group_is_true() {
        // A pathological group {0, 0, 1} — same column listed twice in
        // one group is also overlap by the bitset check.
        let g = Groups::from_csr(vec![0, 3], vec![0, 0, 1]).unwrap();
        assert!(g.has_overlap());
    }

    #[test]
    fn has_overlap_empty_groups_is_false() {
        // Zero groups, empty idx.
        let g = Groups::from_csr(vec![0], vec![]).unwrap();
        assert!(!g.has_overlap());
    }
}
