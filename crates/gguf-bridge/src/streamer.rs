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
        let input_file = File::open(input_path)?;
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
}
