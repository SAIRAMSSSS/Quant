use crate::error::CoreError;
use crate::types::GgmlType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorRole {
    Embedding,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionOutput,
    MlpGate,
    MlpUp,
    MlpDown,
    LayerNorm,
    LmHead,
    Bias,
    Other,
}

impl TensorRole {
    pub fn sensitivity(&self) -> f64 {
        match self {
            Self::LmHead => 2.2,
            Self::Embedding => 1.8,
            Self::AttentionQ | Self::AttentionK | Self::AttentionV | Self::AttentionOutput => 1.3,
            Self::MlpDown => 1.25,
            Self::MlpGate | Self::MlpUp => 1.0,
            Self::LayerNorm | Self::Bias => 100.0,
            Self::Other => 1.0,
        }
    }

    pub fn allowed_types(&self) -> &'static [GgmlType] {
        match self {
            Self::LayerNorm | Self::Bias => &[GgmlType::F32],
            Self::LmHead => &[
                GgmlType::Q8_0,
                GgmlType::Q6_K,
                GgmlType::Q5_K,
                GgmlType::Q4_K,
                GgmlType::F16,
            ],
            Self::Embedding => &[
                GgmlType::Q8_0,
                GgmlType::Q6_K,
                GgmlType::Q5_K,
                GgmlType::Q4_K,
                GgmlType::F16,
            ],
            Self::AttentionQ | Self::AttentionK | Self::AttentionV | Self::AttentionOutput => &[
                GgmlType::Q8_0,
                GgmlType::Q6_K,
                GgmlType::Q5_K,
                GgmlType::Q4_K,
                GgmlType::Q4_0,
            ],
            Self::MlpDown => &[
                GgmlType::Q8_0,
                GgmlType::Q6_K,
                GgmlType::Q5_K,
                GgmlType::Q4_K,
                GgmlType::Q4_0,
                GgmlType::Q3_K,
            ],
            Self::MlpGate | Self::MlpUp => &[
                GgmlType::Q8_0,
                GgmlType::Q6_K,
                GgmlType::Q5_K,
                GgmlType::Q4_K,
                GgmlType::Q4_0,
                GgmlType::Q3_K,
                GgmlType::Q2_K,
            ],
            Self::Other => &[
                GgmlType::Q8_0,
                GgmlType::Q6_K,
                GgmlType::Q5_K,
                GgmlType::Q4_K,
                GgmlType::Q4_0,
                GgmlType::F16,
            ],
        }
    }
}

pub fn classify_tensor_role(name: &str) -> TensorRole {
    let lower = name.to_ascii_lowercase();
    if lower.contains("bias") {
        TensorRole::Bias
    } else if lower.contains("norm") || lower.contains("ln_") || lower.contains("layernorm") {
        TensorRole::LayerNorm
    } else if lower.contains("lm_head") || lower.contains("output.weight") {
        TensorRole::LmHead
    } else if lower.contains("embed") || lower.contains("token_embd") || lower.contains("wte") {
        TensorRole::Embedding
    } else if lower.contains("q_proj") || lower.contains("attn_q") || lower.contains("wq") {
        TensorRole::AttentionQ
    } else if lower.contains("k_proj") || lower.contains("attn_k") || lower.contains("wk") {
        TensorRole::AttentionK
    } else if lower.contains("v_proj") || lower.contains("attn_v") || lower.contains("wv") {
        TensorRole::AttentionV
    } else if lower.contains("o_proj")
        || lower.contains("attn_output")
        || lower.contains("wo")
        || lower.contains("dense")
    {
        TensorRole::AttentionOutput
    } else if lower.contains("gate_proj") || lower.contains("ffn_gate") || lower.contains("w1") {
        TensorRole::MlpGate
    } else if lower.contains("up_proj") || lower.contains("ffn_up") || lower.contains("w3") {
        TensorRole::MlpUp
    } else if lower.contains("down_proj") || lower.contains("ffn_down") || lower.contains("w2") {
        TensorRole::MlpDown
    } else {
        TensorRole::Other
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateOption {
    pub quant_type: GgmlType,
    pub size_bytes: usize,
    pub distortion: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorGroup {
    pub name: String,
    pub role: TensorRole,
    pub shape: Vec<usize>,
    pub num_elements: usize,
    pub candidates: Vec<CandidateOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorPlanDetail {
    pub name: String,
    pub role: TensorRole,
    pub shape: Vec<usize>,
    pub num_elements: usize,
    pub selected_type: GgmlType,
    pub size_bytes: usize,
    pub distortion: f64,
    pub bpw: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationPlan {
    pub tensor_plans: Vec<TensorPlanDetail>,
    pub total_weight_bytes: usize,
    pub kv_cache_bytes: usize,
    pub overhead_bytes: usize,
    pub total_vram_bytes: usize,
    pub target_vram_bytes: usize,
    pub total_distortion: f64,
    pub margin_bytes: i64,
}

impl QuantizationPlan {
    pub fn get_type_for_tensor(&self, name: &str) -> Option<GgmlType> {
        self.tensor_plans
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.selected_type)
    }

    pub fn type_histogram(&self) -> HashMap<GgmlType, usize> {
        let mut hist = HashMap::new();
        for plan in &self.tensor_plans {
            *hist.entry(plan.selected_type).or_insert(0) += 1;
        }
        hist
    }
}

pub struct MckpSolver;

impl MckpSolver {
    pub fn build_tensor_groups(
        tensors: &HashMap<String, (Vec<usize>, usize)>,
    ) -> Result<Vec<TensorGroup>, CoreError> {
        let mut groups = Vec::with_capacity(tensors.len());

        for (name, (shape, num_elements)) in tensors {
            let is_1d = shape.len() <= 1;
            let role = classify_tensor_role(name);
            let allowed: &[GgmlType] = if is_1d {
                &[GgmlType::F32]
            } else {
                role.allowed_types()
            };
            let sensitivity = role.sensitivity();
            let row_len = shape.last().copied().unwrap_or(0);

            let mut candidates = Vec::new();
            for &qtype in allowed {
                if row_len > 0 && !row_len.is_multiple_of(qtype.block_size()) {
                    continue;
                }

                if let Ok(size_bytes) = qtype.bytes_for_elements(*num_elements) {
                    let base_err = qtype.base_distortion();
                    let distortion = sensitivity * base_err * (*num_elements as f64);
                    candidates.push(CandidateOption {
                        quant_type: qtype,
                        size_bytes,
                        distortion,
                    });
                }
            }

            if candidates.is_empty() {
                let fallbacks: &[GgmlType] = if is_1d {
                    &[GgmlType::F32]
                } else {
                    &[GgmlType::Q8_0, GgmlType::Q4_0, GgmlType::F16]
                };
                for &fallback in fallbacks {
                    if row_len == 0 || row_len.is_multiple_of(fallback.block_size()) {
                        if let Ok(size_bytes) = fallback.bytes_for_elements(*num_elements) {
                            let base_err = fallback.base_distortion();
                            candidates.push(CandidateOption {
                                quant_type: fallback,
                                size_bytes,
                                distortion: sensitivity * base_err * (*num_elements as f64),
                            });
                        }
                    }
                }
            }

            if candidates.is_empty() {
                return Err(CoreError::MissingCandidates(name.clone()));
            }

            candidates.sort_by_key(|a| a.size_bytes);
            groups.push(TensorGroup {
                name: name.clone(),
                role,
                shape: shape.clone(),
                num_elements: *num_elements,
                candidates,
            });
        }

        groups.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(groups)
    }

    pub fn solve(
        groups: &[TensorGroup],
        target_vram_bytes: usize,
        kv_cache_bytes: usize,
        overhead_bytes: usize,
    ) -> Result<QuantizationPlan, CoreError> {
        let total_non_weight = kv_cache_bytes.saturating_add(overhead_bytes);
        if target_vram_bytes <= total_non_weight {
            return Err(CoreError::KnapsackInfeasible {
                budget_bytes: target_vram_bytes,
                minimum_bytes: total_non_weight,
            });
        }

        let weight_budget = target_vram_bytes.saturating_sub(total_non_weight);

        let min_weight_sum: usize = groups
            .iter()
            .map(|g| g.candidates.first().map_or(0, |c| c.size_bytes))
            .sum();

        if min_weight_sum > weight_budget {
            return Err(CoreError::KnapsackInfeasible {
                budget_bytes: weight_budget,
                minimum_bytes: min_weight_sum,
            });
        }

        let max_budget_granules = 4096usize;
        let unit_size = (weight_budget / max_budget_granules).max(64 * 1024);
        let budget_units = weight_budget / unit_size;

        let num_items = groups.len();
        let inf = f64::INFINITY;

        let mut dp = vec![inf; budget_units + 1];
        let mut choices = vec![vec![0usize; budget_units + 1]; num_items];

        dp[0] = 0.0;

        for (i, group) in groups.iter().enumerate() {
            let mut next_dp = vec![inf; budget_units + 1];
            for (c_idx, cand) in group.candidates.iter().enumerate() {
                let w_units = cand.size_bytes.div_ceil(unit_size);
                let dist = cand.distortion;

                for b in w_units..=budget_units {
                    let prev = dp[b - w_units];
                    if prev != inf {
                        let candidate_total = prev + dist;
                        if candidate_total < next_dp[b] {
                            next_dp[b] = candidate_total;
                            choices[i][b] = c_idx;
                        }
                    }
                }
            }
            dp = next_dp;
        }

        let mut best_b = 0;
        let mut min_total_dist = inf;
        for (b, &dist) in dp.iter().enumerate() {
            if dist < min_total_dist {
                min_total_dist = dist;
                best_b = b;
            }
        }

        let mut selected_indices = vec![0usize; num_items];
        if min_total_dist < inf {
            let mut curr_b = best_b;
            for i in (0..num_items).rev() {
                let c_idx = choices[i][curr_b];
                selected_indices[i] = c_idx;
                let w_units = groups[i].candidates[c_idx].size_bytes.div_ceil(unit_size);
                curr_b = curr_b.saturating_sub(w_units);
            }
        } else {
            for (i, group) in groups.iter().enumerate() {
                let mut best_c = 0;
                let mut best_score = inf;
                for (c_idx, c) in group.candidates.iter().enumerate() {
                    if c.distortion < best_score {
                        best_score = c.distortion;
                        best_c = c_idx;
                    }
                }
                selected_indices[i] = best_c;
            }
        }

        let mut total_weights: usize = groups
            .iter()
            .enumerate()
            .map(|(i, g)| g.candidates[selected_indices[i]].size_bytes)
            .sum();

        while total_weights > weight_budget {
            let mut downgrade_item = None;
            let mut lowest_penalty = inf;

            for (i, g) in groups.iter().enumerate() {
                let cur_idx = selected_indices[i];
                if cur_idx > 0 {
                    let cur_cand = &g.candidates[cur_idx];
                    let lower_cand = &g.candidates[cur_idx - 1];
                    let delta_weight = cur_cand.size_bytes.saturating_sub(lower_cand.size_bytes);
                    if delta_weight > 0 {
                        let delta_dist = lower_cand.distortion - cur_cand.distortion;
                        let penalty_ratio = delta_dist / (delta_weight as f64);
                        if penalty_ratio < lowest_penalty {
                            lowest_penalty = penalty_ratio;
                            downgrade_item = Some(i);
                        }
                    }
                }
            }

            match downgrade_item {
                Some(idx) => {
                    let old_idx = selected_indices[idx];
                    let new_idx = old_idx - 1;
                    selected_indices[idx] = new_idx;
                    total_weights = total_weights.saturating_sub(
                        groups[idx].candidates[old_idx].size_bytes
                            - groups[idx].candidates[new_idx].size_bytes,
                    );
                }
                None => {
                    return Err(CoreError::KnapsackInfeasible {
                        budget_bytes: weight_budget,
                        minimum_bytes: total_weights,
                    });
                }
            }
        }

        let mut tensor_plans = Vec::with_capacity(num_items);
        let mut final_total_distortion = 0.0;

        for (i, group) in groups.iter().enumerate() {
            let cand = &group.candidates[selected_indices[i]];
            final_total_distortion += cand.distortion;
            tensor_plans.push(TensorPlanDetail {
                name: group.name.clone(),
                role: group.role,
                shape: group.shape.clone(),
                num_elements: group.num_elements,
                selected_type: cand.quant_type,
                size_bytes: cand.size_bytes,
                distortion: cand.distortion,
                bpw: cand.quant_type.effective_bpw(),
            });
        }

        let total_vram = total_weights
            .saturating_add(kv_cache_bytes)
            .saturating_add(overhead_bytes);
        let margin = (target_vram_bytes as i64) - (total_vram as i64);

        Ok(QuantizationPlan {
            tensor_plans,
            total_weight_bytes: total_weights,
            kv_cache_bytes,
            overhead_bytes,
            total_vram_bytes: total_vram,
            target_vram_bytes,
            total_distortion: final_total_distortion,
            margin_bytes: margin,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_tensor_roles() {
        assert_eq!(
            classify_tensor_role("model.layers.0.self_attn.q_proj.weight"),
            TensorRole::AttentionQ
        );
        assert_eq!(
            classify_tensor_role("model.layers.0.self_attn.k_proj.bias"),
            TensorRole::Bias
        );
        assert_eq!(
            classify_tensor_role("model.layers.0.mlp.down_proj.weight"),
            TensorRole::MlpDown
        );
        assert_eq!(
            classify_tensor_role("model.layers.0.input_layernorm.weight"),
            TensorRole::LayerNorm
        );
        assert_eq!(classify_tensor_role("lm_head.weight"), TensorRole::LmHead);
        assert_eq!(
            classify_tensor_role("model.embed_tokens.weight"),
            TensorRole::Embedding
        );
    }

    #[test]
    fn test_mckp_solver_feasibility() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![1024, 256], 1024 * 256),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![256, 256], 256 * 256),
        );
        tensors.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            (vec![256, 1024], 256 * 1024),
        );
        tensors.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            (vec![256], 256),
        );

        let groups = MckpSolver::build_tensor_groups(&tensors).unwrap();
        let target_vram = 2 * 1024 * 1024;
        let kv_cache = 256 * 1024;
        let overhead = 256 * 1024;

        let plan = MckpSolver::solve(&groups, target_vram, kv_cache, overhead).unwrap();
        assert!(plan.total_vram_bytes <= target_vram);
        assert!(plan.margin_bytes >= 0);
        assert_eq!(
            plan.get_type_for_tensor("model.layers.0.input_layernorm.weight")
                .unwrap(),
            GgmlType::F32
        );
    }

    #[test]
    fn test_mckp_infeasible_rejection() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![4096, 4096], 4096 * 4096),
        );

        let groups = MckpSolver::build_tensor_groups(&tensors).unwrap();
        let target_vram = 1024;
        let kv_cache = 512;
        let overhead = 512;

        let res = MckpSolver::solve(&groups, target_vram, kv_cache, overhead);
        assert!(res.is_err());
    }
}
