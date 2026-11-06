use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use quant_core::mckp::{MckpSolver, QuantizationPlan};
use quant_core::safetensors::parse_from_file;
use quant_core::types::GgmlType;
use quant_gguf_bridge::streamer::StreamQuantizer;
use quant_hw_probe::probe_hardware as probe_hw_inner;
use std::collections::HashMap;
use std::path::Path;

#[pyclass(name = "HardwareProfile")]
#[derive(Clone)]
pub struct PyHardwareProfile {
    #[pyo3(get)]
    pub total_system_ram: u64,
    #[pyo3(get)]
    pub available_system_ram: u64,
    #[pyo3(get)]
    pub primary_vram: Option<u64>,
    #[pyo3(get)]
    pub gpus: Vec<(String, String, u64, u64)>,
}

#[pymethods]
impl PyHardwareProfile {
    pub fn recommend_vram(&self, headroom_ratio: f64) -> u64 {
        let valid_ratio = headroom_ratio.clamp(0.05, 0.50);
        if let Some(vram) = self.primary_vram {
            let usable = (vram as f64) * (1.0 - valid_ratio);
            usable as u64
        } else {
            let usable = (self.available_system_ram as f64) * 0.70;
            usable as u64
        }
    }
}

#[pyclass(name = "QuantizationPlan")]
#[derive(Clone)]
pub struct PyQuantizationPlan {
    #[pyo3(get)]
    pub target_vram_bytes: usize,
    #[pyo3(get)]
    pub total_weight_bytes: usize,
    #[pyo3(get)]
    pub kv_cache_bytes: usize,
    #[pyo3(get)]
    pub overhead_bytes: usize,
    #[pyo3(get)]
    pub total_vram_bytes: usize,
    #[pyo3(get)]
    pub margin_bytes: i64,
    #[pyo3(get)]
    pub total_distortion: f64,
    #[pyo3(get)]
    pub assignments: HashMap<String, String>,
    pub(crate) inner_plan: QuantizationPlan,
}

#[pymethods]
impl PyQuantizationPlan {
    pub fn get_type(&self, tensor_name: &str) -> Option<String> {
        self.assignments.get(tensor_name).cloned()
    }

    pub fn quantization_mixture(&self) -> HashMap<String, usize> {
        let hist = self.inner_plan.type_histogram();
        let mut map = HashMap::new();
        for (k, v) in hist {
            map.insert(k.as_str().to_string(), v);
        }
        map
    }
}

#[pyclass(name = "ModelSummary")]
#[derive(Clone)]
pub struct PyModelSummary {
    #[pyo3(get)]
    pub header_size: u64,
    #[pyo3(get)]
    pub tensor_count: usize,
    #[pyo3(get)]
    pub total_elements: usize,
    #[pyo3(get)]
    pub total_bytes: usize,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
}

#[pyfunction]
pub fn probe_hardware() -> PyResult<PyHardwareProfile> {
    let hw = probe_hw_inner();
    let primary_vram = hw.primary_vram_bytes();
    let total_system_ram = hw.system_ram.total_bytes;
    let available_system_ram = hw.system_ram.available_bytes;

    let gpus = hw
        .gpus
        .into_iter()
        .map(|g| {
            let backend_str = match g.backend {
                quant_hw_probe::GpuBackend::Cuda => "CUDA".to_string(),
                quant_hw_probe::GpuBackend::Rocm => "ROCm".to_string(),
                quant_hw_probe::GpuBackend::Metal => "Metal".to_string(),
            };
            (g.name, backend_str, g.total_vram_bytes, g.free_vram_bytes)
        })
        .collect();

    Ok(PyHardwareProfile {
        total_system_ram,
        available_system_ram,
        primary_vram,
        gpus,
    })
}

#[pyfunction]
#[pyo3(signature = (model_path, target_vram_bytes=0, ctx_size=4096, kv_type="f16", overhead_mb=512))]
pub fn plan_quantization(
    model_path: &str,
    target_vram_bytes: u64,
    ctx_size: usize,
    kv_type: &str,
    overhead_mb: usize,
) -> PyResult<PyQuantizationPlan> {
    let (tensors, _metadata) = if model_path.starts_with("http://")
        || model_path.starts_with("https://")
    {
        let header = quant_core::safetensors::fetch_remote_header(model_path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        (header.tensors, header.metadata)
    } else {
        let path = Path::new(model_path);
        if !path.exists() {
            return Err(PyIOError::new_err(format!(
                "model file not found: {model_path}"
            )));
        }

        if path.is_dir()
            || path.file_name().and_then(|s| s.to_str()) == Some("model.safetensors.index.json")
        {
            let multi = quant_core::load_multi_shard_headers(path)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            (multi.combined_tensors, multi.combined_metadata)
        } else {
            let header = parse_from_file(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
            (header.tensors, header.metadata)
        }
    };

    let mut tensor_map = HashMap::with_capacity(tensors.len());
    for (k, v) in &tensors {
        tensor_map.insert(k.clone(), (v.shape.clone(), v.element_count()));
    }

    let kv_enum = GgmlType::from_str_case_insensitive(kv_type).unwrap_or(GgmlType::F16);
    let overhead_bytes = overhead_mb * 1024 * 1024;

    let target_bytes = if target_vram_bytes == 0 {
        let hw = probe_hw_inner();
        hw.recommend_vram_target(0.10) as usize
    } else {
        target_vram_bytes as usize
    };

    let mut n_layers = 0usize;
    let n_kv_heads = 8usize;
    let mut head_dim = 128usize;

    for (name, meta) in &tensors {
        if name.contains("layers.") {
            let parts: Vec<&str> = name.split('.').collect();
            for (idx, &part) in parts.iter().enumerate() {
                if part == "layers" && idx + 1 < parts.len() {
                    if let Ok(num) = parts[idx + 1].parse::<usize>() {
                        if num + 1 > n_layers {
                            n_layers = num + 1;
                        }
                    }
                }
            }
        }
        if name.contains("k_proj") && meta.shape.len() >= 2 {
            head_dim = meta.shape[1].min(128);
        }
    }

    if n_layers == 0 {
        n_layers = 32;
    }

    let kv_cfg = quant_core::KvCacheConfig::new(n_layers, n_kv_heads, head_dim, ctx_size, kv_enum);
    let kv_bytes = kv_cfg
        .total_bytes()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let groups = MckpSolver::build_tensor_groups(&tensor_map)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let plan = MckpSolver::solve(&groups, target_bytes, kv_bytes, overhead_bytes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut assignments = HashMap::new();
    for p in &plan.tensor_plans {
        assignments.insert(p.name.clone(), p.selected_type.as_str().to_string());
    }

    Ok(PyQuantizationPlan {
        target_vram_bytes: plan.target_vram_bytes,
        total_weight_bytes: plan.total_weight_bytes,
        kv_cache_bytes: plan.kv_cache_bytes,
        overhead_bytes: plan.overhead_bytes,
        total_vram_bytes: plan.total_vram_bytes,
        margin_bytes: plan.margin_bytes,
        total_distortion: plan.total_distortion,
        assignments,
        inner_plan: plan,
    })
}

#[pyfunction]
pub fn inspect_model(model_path: &str) -> PyResult<PyModelSummary> {
    if model_path.starts_with("http://") || model_path.starts_with("https://") {
        let header = quant_core::safetensors::fetch_remote_header(model_path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut total_elements = 0usize;
        let mut total_bytes = 0usize;
        for meta in header.tensors.values() {
            total_elements += meta.element_count();
            total_bytes += meta.byte_size();
        }

        return Ok(PyModelSummary {
            header_size: header.header_size,
            tensor_count: header.tensors.len(),
            total_elements,
            total_bytes,
            metadata: header.metadata,
        });
    }

    let path = Path::new(model_path);
    if !path.exists() {
        return Err(PyIOError::new_err(format!(
            "model file not found: {model_path}"
        )));
    }

    if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
        let gguf = quant_gguf_bridge::reader::inspect_gguf_file(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut total_elements = 0usize;
        let mut total_bytes = 0usize;
        for t in &gguf.tensors {
            let elems: usize = t.dimensions.iter().map(|&d| d as usize).product();
            total_elements += elems;
            if let Ok(bytes) = t.ggml_type.bytes_for_elements(elems) {
                total_bytes += bytes;
            }
        }

        let mut meta_str_map = HashMap::new();
        for (k, v) in &gguf.metadata {
            meta_str_map.insert(k.clone(), format!("{v:?}"));
        }

        return Ok(PyModelSummary {
            header_size: gguf.data_offset,
            tensor_count: gguf.tensors.len(),
            total_elements,
            total_bytes,
            metadata: meta_str_map,
        });
    }

    let (tensors, metadata, header_size) = if path.is_dir()
        || path.file_name().and_then(|s| s.to_str()) == Some("model.safetensors.index.json")
    {
        let multi = quant_core::load_multi_shard_headers(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        (multi.combined_tensors, multi.combined_metadata, 0u64)
    } else {
        let header = parse_from_file(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        (header.tensors, header.metadata, header.header_size)
    };

    let mut total_elements = 0usize;
    let mut total_bytes = 0usize;

    for meta in tensors.values() {
        total_elements += meta.element_count();
        total_bytes += meta.byte_size();
    }

    Ok(PyModelSummary {
        header_size,
        tensor_count: tensors.len(),
        total_elements,
        total_bytes,
        metadata,
    })
}

#[pyfunction]
pub fn execute_quantization(
    model_path: &str,
    output_path: &str,
    plan: &PyQuantizationPlan,
) -> PyResult<u64> {
    let written =
        StreamQuantizer::stream_safetensors_to_gguf(model_path, output_path, &plan.inner_plan)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(written)
}

#[pymodule]
fn quant_engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHardwareProfile>()?;
    m.add_class::<PyQuantizationPlan>()?;
    m.add_class::<PyModelSummary>()?;

    m.add_function(wrap_pyfunction!(probe_hardware, m)?)?;
    m.add_function(wrap_pyfunction!(plan_quantization, m)?)?;
    m.add_function(wrap_pyfunction!(inspect_model, m)?)?;
    m.add_function(wrap_pyfunction!(execute_quantization, m)?)?;

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_py_hardware_profile_recommend() {
        let profile = PyHardwareProfile {
            total_system_ram: 16 * 1024 * 1024 * 1024,
            available_system_ram: 8 * 1024 * 1024 * 1024,
            primary_vram: Some(4 * 1024 * 1024 * 1024),
            gpus: vec![],
        };
        let rec = profile.recommend_vram(0.10);
        assert!(rec > 0);
        assert!(rec < 4 * 1024 * 1024 * 1024);
    }
}
