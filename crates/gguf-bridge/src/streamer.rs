use crate::constants::GGUF_DEFAULT_ALIGNMENT;
use crate::error::BridgeError;
use crate::metadata::GgufValue;
use crate::quantize::quantize_raw_tensor;
use crate::tensor_info::GgufTensorInfo;
use crate::writer::GgufWriter;
use quant_core::mckp::QuantizationPlan;
use quant_core::page_aligned::align_to_page;
use quant_core::safetensors::parse_header_bytes;
use quant_core::types::GgmlType;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::Path;

pub struct StreamQuantizer;

impl StreamQuantizer {
    pub fn stream_safetensors_to_gguf<P1: AsRef<Path>, P2: AsRef<Path>>(
        input_path: P1,
        output_path: P2,
        plan: &QuantizationPlan,
    ) -> Result<u64, BridgeError> {
        let p = input_path.as_ref();
        if p.is_dir()
            || p.file_name().and_then(|s| s.to_str()) == Some("model.safetensors.index.json")
        {
            return Self::stream_multi_shard_to_gguf(p, output_path, plan);
        }

        let input_file = File::open(p)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&input_file)? };

        let header = parse_header_bytes(&mmap)?;
        let data_start_offset = 8usize.saturating_add(header.header_size as usize);

        let mut tensor_names: Vec<String> = header.tensors.keys().cloned().collect();
        tensor_names.sort();

        let mut tensor_infos = Vec::with_capacity(tensor_names.len());
        let mut running_data_offset = 0u64;

        for name in &tensor_names {
            let meta = match header.tensors.get(name) {
                Some(m) => m,
                None => continue,
            };

            let target_type = plan.get_type_for_tensor(name).unwrap_or(GgmlType::F16);
            let elem_count = meta.element_count();
            let size_bytes = target_type.bytes_for_elements(elem_count)? as u64;

            let dims: Vec<u64> = meta.shape.iter().rev().map(|&d| d as u64).collect();

            tensor_infos.push(GgufTensorInfo {
                name: name.clone(),
                dimensions: dims,
                ggml_type: target_type,
                offset: running_data_offset,
            });

            running_data_offset += size_bytes;
            let rem = (running_data_offset as usize) % GGUF_DEFAULT_ALIGNMENT;
            if rem != 0 {
                running_data_offset += (GGUF_DEFAULT_ALIGNMENT - rem) as u64;
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert(
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        );
        metadata.insert(
            "general.alignment".to_string(),
            GgufValue::Uint32(GGUF_DEFAULT_ALIGNMENT as u32),
        );

        let out_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(output_path)?;

        let mut writer = GgufWriter::new(out_file, GGUF_DEFAULT_ALIGNMENT);
        writer.write_header_and_metadata(&metadata, &tensor_infos)?;

        let mut quant_buffer = Vec::new();

        for name in &tensor_names {
            let meta = match header.tensors.get(name) {
                Some(m) => m,
                None => continue,
            };

            let target_type = plan.get_type_for_tensor(name).unwrap_or(GgmlType::F16);

            let src_start = data_start_offset.saturating_add(meta.data_offsets.0);
            let src_end = data_start_offset.saturating_add(meta.data_offsets.1);

            if src_end > mmap.len() {
                return Err(BridgeError::BufferSizeMismatch {
                    expected: src_end,
                    found: mmap.len(),
                });
            }

            let raw_src = &mmap[src_start..src_end];
            quantize_raw_tensor(&meta.dtype, raw_src, target_type, &mut quant_buffer)?;

            writer.write_tensor_data(&quant_buffer)?;
            quant_buffer.clear();

            #[cfg(unix)]
            {
                let src_len = meta.byte_size();
                let (page_start, page_len) = align_to_page(src_start, src_len);
                if page_len > 0 && page_start.saturating_add(page_len) <= mmap.len() {
                    // SAFETY: mmap is read-only file mapping, page boundaries are aligned to PAGE_SIZE.
                    unsafe {
                        let _ = mmap.unchecked_advise_range(
                            memmap2::UncheckedAdvice::DontNeed,
                            page_start,
                            page_len,
                        );
                    }
                }
            }
        }

        Ok(running_data_offset)
    }

    pub fn stream_multi_shard_to_gguf<P1: AsRef<Path>, P2: AsRef<Path>>(
        input_path: P1,
        output_path: P2,
        plan: &QuantizationPlan,
    ) -> Result<u64, BridgeError> {
        let p = input_path.as_ref();
        let base_dir = if p.is_dir() {
            p.to_path_buf()
        } else {
            p.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
        };

        let multi = quant_core::load_multi_shard_headers(p)?;
        let mut tensor_names: Vec<String> = multi.combined_tensors.keys().cloned().collect();
        tensor_names.sort();

        let mut tensor_infos = Vec::with_capacity(tensor_names.len());
        let mut running_data_offset = 0u64;

        for name in &tensor_names {
            let meta = match multi.combined_tensors.get(name) {
                Some(m) => m,
                None => continue,
            };

            let target_type = plan.get_type_for_tensor(name).unwrap_or(GgmlType::F16);
            let elem_count = meta.element_count();
            let size_bytes = target_type.bytes_for_elements(elem_count)? as u64;

            let dims: Vec<u64> = meta.shape.iter().rev().map(|&d| d as u64).collect();

            tensor_infos.push(GgufTensorInfo {
                name: name.clone(),
                dimensions: dims,
                ggml_type: target_type,
                offset: running_data_offset,
            });

            running_data_offset += size_bytes;
            let rem = (running_data_offset as usize) % GGUF_DEFAULT_ALIGNMENT;
            if rem != 0 {
                running_data_offset += (GGUF_DEFAULT_ALIGNMENT - rem) as u64;
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert(
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        );
        metadata.insert(
            "general.alignment".to_string(),
            GgufValue::Uint32(GGUF_DEFAULT_ALIGNMENT as u32),
        );

        let out_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(output_path)?;

        let mut writer = GgufWriter::new(out_file, GGUF_DEFAULT_ALIGNMENT);
        writer.write_header_and_metadata(&metadata, &tensor_infos)?;

        let mut quant_buffer = Vec::new();
        let mut current_shard: Option<(String, memmap2::Mmap, usize)> = None;

        for name in &tensor_names {
            let meta = match multi.combined_tensors.get(name) {
                Some(m) => m,
                None => continue,
            };
            let shard_name = match multi.tensor_to_shard.get(name) {
                Some(s) => s,
                None => continue,
            };

            let need_reload = match &current_shard {
                Some((cur, _, _)) => cur != shard_name,
                None => true,
            };

            if need_reload {
                let shard_path = base_dir.join(shard_name);
                let sf = File::open(&shard_path)?;
                let smmap = unsafe { memmap2::MmapOptions::new().map(&sf)? };
                let sh_hdr = match multi.shards.get(shard_name) {
                    Some(h) => h,
                    None => continue,
                };
                let d_start = 8usize.saturating_add(sh_hdr.header_size as usize);
                current_shard = Some((shard_name.clone(), smmap, d_start));
            }

            let (_, ref smmap, data_start_offset) = match &current_shard {
                Some(tup) => tup,
                None => continue,
            };

            let target_type = plan.get_type_for_tensor(name).unwrap_or(GgmlType::F16);
            let src_start = data_start_offset.saturating_add(meta.data_offsets.0);
            let src_end = data_start_offset.saturating_add(meta.data_offsets.1);

            if src_end > smmap.len() {
                return Err(BridgeError::BufferSizeMismatch {
                    expected: src_end,
                    found: smmap.len(),
                });
            }

            let raw_src = &smmap[src_start..src_end];
            quantize_raw_tensor(&meta.dtype, raw_src, target_type, &mut quant_buffer)?;

            writer.write_tensor_data(&quant_buffer)?;
            quant_buffer.clear();

            #[cfg(unix)]
            {
                let src_len = meta.byte_size();
                let (page_start, page_len) = align_to_page(src_start, src_len);
                if page_len > 0 && page_start.saturating_add(page_len) <= smmap.len() {
                    unsafe {
                        let _ = smmap.unchecked_advise_range(
                            memmap2::UncheckedAdvice::DontNeed,
                            page_start,
                            page_len,
                        );
                    }
                }
            }
        }

        Ok(running_data_offset)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use quant_core::mckp::MckpSolver;
    use tempfile::NamedTempFile;

    #[test]
    fn test_stream_quantizer_end_to_end() {
        let f16_data: Vec<u8> = vec![0u8; 128 * 2];
        let json = format!(
            r#"{{"__metadata__":{{"format":"pt"}},"model.layers.0.self_attn.q_proj.weight":{{"dtype":"F16","shape":[16,8],"data_offsets":[0,{}]}}}}"#,
            f16_data.len()
        );

        let json_bytes = json.as_bytes();
        let header_len = json_bytes.len() as u64;

        let mut safetensors_bytes = Vec::new();
        safetensors_bytes.extend_from_slice(&header_len.to_le_bytes());
        safetensors_bytes.extend_from_slice(json_bytes);
        safetensors_bytes.extend_from_slice(&f16_data);

        let mut in_temp = NamedTempFile::new().unwrap();
        use std::io::Write;
        in_temp.write_all(&safetensors_bytes).unwrap();
        in_temp.flush().unwrap();

        let out_temp = NamedTempFile::new().unwrap();

        let mut tensors_map = HashMap::new();
        tensors_map.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![16, 8], 128),
        );
        let groups = MckpSolver::build_tensor_groups(&tensors_map).unwrap();
        let plan = MckpSolver::solve(&groups, 1024 * 1024, 0, 0).unwrap();

        let written =
            StreamQuantizer::stream_safetensors_to_gguf(in_temp.path(), out_temp.path(), &plan)
                .unwrap();

        assert!(written > 0);

        let gguf_hdr = crate::reader::inspect_gguf_file(out_temp.path()).unwrap();
        assert_eq!(gguf_hdr.version, 3);
        assert_eq!(gguf_hdr.tensor_count, 1);
        assert_eq!(
            gguf_hdr.tensors[0].name,
            "model.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn test_stream_multi_shard_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let s1_path = tmp.path().join("model-00001-of-00002.safetensors");
        let s2_path = tmp.path().join("model-00002-of-00002.safetensors");
        let idx_path = tmp.path().join("model.safetensors.index.json");
        let out_gguf = tmp.path().join("output.gguf");

        let s1_json = r#"{"model.embed_tokens.weight":{"dtype":"F16","shape":[16,16],"data_offsets":[0,512]}}"#;
        let mut s1_buf = Vec::new();
        s1_buf.extend_from_slice(&(s1_json.len() as u64).to_le_bytes());
        s1_buf.extend_from_slice(s1_json.as_bytes());
        s1_buf.extend_from_slice(&vec![0u8; 512]);
        std::fs::write(&s1_path, s1_buf).unwrap();

        let s2_json =
            r#"{"lm_head.weight":{"dtype":"F16","shape":[16,16],"data_offsets":[0,512]}}"#;
        let mut s2_buf = Vec::new();
        s2_buf.extend_from_slice(&(s2_json.len() as u64).to_le_bytes());
        s2_buf.extend_from_slice(s2_json.as_bytes());
        s2_buf.extend_from_slice(&vec![0u8; 512]);
        std::fs::write(&s2_path, s2_buf).unwrap();

        let idx_json = r#"{
            "metadata": {"total_size": 1024},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        std::fs::write(&idx_path, idx_json).unwrap();

        let mut tensors_map = HashMap::new();
        tensors_map.insert("model.embed_tokens.weight".to_string(), (vec![16, 16], 256));
        tensors_map.insert("lm_head.weight".to_string(), (vec![16, 16], 256));

        let groups = MckpSolver::build_tensor_groups(&tensors_map).unwrap();
        let plan = MckpSolver::solve(&groups, 1024 * 1024, 0, 0).unwrap();

        let written =
            StreamQuantizer::stream_safetensors_to_gguf(tmp.path(), &out_gguf, &plan).unwrap();
        assert!(written > 0);

        let gguf_hdr = crate::reader::inspect_gguf_file(&out_gguf).unwrap();
        assert_eq!(gguf_hdr.tensor_count, 2);
    }
}
