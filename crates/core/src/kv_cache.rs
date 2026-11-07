use crate::error::CoreError;
use crate::types::GgmlType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheConfig {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ctx_len: usize,
    pub quant_type: GgmlType,
}

impl KvCacheConfig {
    #[must_use]
    pub const fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        ctx_len: usize,
        quant_type: GgmlType,
    ) -> Self {
        Self {
            n_layers,
            n_kv_heads,
            head_dim,
            ctx_len,
            quant_type,
        }
    }

    #[must_use]
    pub const fn elements_per_token(&self) -> usize {
        2 * self.n_layers * self.n_kv_heads * self.head_dim
    }

    pub fn bytes_per_token(&self) -> Result<usize, CoreError> {
        let elems = self.elements_per_token();
        self.quant_type.bytes_for_elements(elems)
    }

    pub fn total_bytes(&self) -> Result<usize, CoreError> {
        let bpt = self.bytes_per_token()?;
        Ok(bpt.saturating_mul(self.ctx_len))
    }

    pub fn max_tokens_for_budget(&self, budget_bytes: usize) -> Result<usize, CoreError> {
        let bpt = self.bytes_per_token()?;
        if bpt == 0 {
            return Ok(0);
        }
        Ok(budget_bytes / bpt)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_llama3_8b_kv_cache_f16() {
        let cfg = KvCacheConfig::new(32, 8, 128, 4096, GgmlType::F16);
        let elems = cfg.elements_per_token();
        assert_eq!(elems, 2 * 32 * 8 * 128);
        assert_eq!(elems, 65_536);

        let bpt = cfg.bytes_per_token().unwrap();
        assert_eq!(bpt, 65_536 * 2);
        assert_eq!(bpt, 131_072);

        let total = cfg.total_bytes().unwrap();
        assert_eq!(total, 131_072 * 4096);
        assert_eq!(total, 536_870_912);
    }

    #[test]
    fn test_llama3_8b_kv_cache_q8_0() {
        let cfg = KvCacheConfig::new(32, 8, 128, 4096, GgmlType::Q8_0);
        let bpt = cfg.bytes_per_token().unwrap();
        assert_eq!(bpt, (65_536 / 32) * 34);
        assert_eq!(bpt, 69_632);

        let total = cfg.total_bytes().unwrap();
        assert_eq!(total, 69_632 * 4096);
    }

    #[test]
    fn test_llama3_8b_kv_cache_q4_0() {
        let cfg = KvCacheConfig::new(32, 8, 128, 4096, GgmlType::Q4_0);
        let bpt = cfg.bytes_per_token().unwrap();
        assert_eq!(bpt, (65_536 / 32) * 18);
        assert_eq!(bpt, 36_864);

        let total = cfg.total_bytes().unwrap();
        assert_eq!(total, 36_864 * 4096);
    }
}
