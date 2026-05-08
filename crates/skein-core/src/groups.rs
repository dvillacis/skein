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
}
