use crate::constants::GGUF_DEFAULT_ALIGNMENT;
use crate::error::BridgeError;
use crate::metadata::GgufValue;
use crate::quantize::{quantize_f32_slice, quantize_raw_tensor_with_permute};
use crate::tensor_info::GgufTensorInfo;
use crate::writer::GgufWriter;
use byteorder::{ByteOrder, LittleEndian};
use half::f16;
use quant_core::mckp::QuantizationPlan;
use quant_core::page_aligned::align_to_page;
use quant_core::safetensors::parse_header_bytes;
use quant_core::types::GgmlType;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::path::Path;

#[derive(Debug, Clone)]
struct EmitTensorItem {
    source_name: String,
    gguf_name: String,
}

pub fn map_hf_to_gguf_tensor_name(hf_name: &str) -> String {
    if hf_name == "model.embed_tokens.weight" {
        return "token_embd.weight".to_string();
    }
    if hf_name == "model.norm.weight" {
        return "output_norm.weight".to_string();
    }
    if hf_name == "lm_head.weight" {
        return "output.weight".to_string();
    }

    if let Some(rest) = hf_name.strip_prefix("model.layers.") {
        if let Some((layer_idx_str, sub)) = rest.split_once('.') {
            if let Ok(layer_idx) = layer_idx_str.parse::<usize>() {
                let mapped_sub = match sub {
                    "input_layernorm.weight" => "attn_norm.weight",
                    "post_attention_layernorm.weight" => "ffn_norm.weight",
                    "self_attn.q_proj.weight" => "attn_q.weight",
                    "self_attn.q_proj.bias" => "attn_q.bias",
                    "self_attn.k_proj.weight" => "attn_k.weight",
                    "self_attn.k_proj.bias" => "attn_k.bias",
                    "self_attn.v_proj.weight" => "attn_v.weight",
                    "self_attn.v_proj.bias" => "attn_v.bias",
                    "self_attn.o_proj.weight" => "attn_output.weight",
                    "mlp.gate_proj.weight" => "ffn_gate.weight",
                    "mlp.up_proj.weight" => "ffn_up.weight",
                    "mlp.down_proj.weight" => "ffn_down.weight",
                    other => other,
                };
                return format!("blk.{layer_idx}.{mapped_sub}");
            }
        }
    }

    hf_name.to_string()
}

#[derive(Debug, Default)]
struct ConfigMeta {
    metadata: HashMap<String, GgufValue>,
    vocab_size: Option<usize>,
    n_attn_heads: Option<usize>,
    n_kv_heads: Option<usize>,
    rope_is_neox: bool,
}

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

        let mut sorted_sources: Vec<String> = header.tensors.keys().cloned().collect();
        sorted_sources.sort();

        let mut emit_items = Vec::with_capacity(sorted_sources.len() + 1);
        if !header.tensors.contains_key("lm_head.weight")
            && header.tensors.contains_key("model.embed_tokens.weight")
        {
            emit_items.push(EmitTensorItem {
                source_name: "model.embed_tokens.weight".to_string(),
                gguf_name: "output.weight".to_string(),
            });
        }

        for src in sorted_sources {
            let gguf_name = map_hf_to_gguf_tensor_name(&src);
            emit_items.push(EmitTensorItem {
                source_name: src,
                gguf_name,
            });
        }

        let mut tensor_infos = Vec::with_capacity(emit_items.len());
        let mut running_data_offset = 0u64;

        for item in &emit_items {
            let meta = match header.tensors.get(&item.source_name) {
                Some(m) => m,
                None => continue,
            };

            let target_type = plan
                .get_type_for_tensor(&item.gguf_name)
                .or_else(|| plan.get_type_for_tensor(&item.source_name))
                .unwrap_or(GgmlType::F16);
            let elem_count = meta.element_count();
            let size_bytes = target_type.bytes_for_elements(elem_count)? as u64;

            let dims: Vec<u64> = meta.shape.iter().rev().map(|&d| d as u64).collect();

            tensor_infos.push(GgufTensorInfo {
                name: item.gguf_name.clone(),
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
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        );
        metadata.insert(
            "general.alignment".to_string(),
            GgufValue::Uint32(GGUF_DEFAULT_ALIGNMENT as u32),
        );

        let parent_dir = p.parent().unwrap_or(Path::new("."));
        let config_path = parent_dir.join("config.json");
        let mut target_vocab_size = None;
        let mut n_attn_heads = None;
        let mut n_kv_heads = None;
        let mut rope_is_neox = true;
        if config_path.exists() {
            let cfg = Self::load_config_json_metadata(&config_path);
            metadata.extend(cfg.metadata);
            target_vocab_size = cfg.vocab_size;
            n_attn_heads = cfg.n_attn_heads;
            n_kv_heads = cfg.n_kv_heads;
            rope_is_neox = cfg.rope_is_neox;
        }

        let tokenizer_path = parent_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            let tok_cfg_path = parent_dir.join("tokenizer_config.json");
            let tok_cfg = if tok_cfg_path.exists() {
                Some(tok_cfg_path.as_path())
            } else {
                None
            };
            metadata.extend(Self::load_tokenizer_metadata(
                &tokenizer_path,
                tok_cfg,
                target_vocab_size,
            ));
        }

        if !metadata.contains_key("general.architecture") {
            metadata.insert(
                "general.architecture".to_string(),
                GgufValue::String("llama".to_string()),
            );
        }

        let out_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(output_path)?;

        let buf_writer = BufWriter::with_capacity(4 * 1024 * 1024, out_file);
        let mut writer = GgufWriter::new(buf_writer, GGUF_DEFAULT_ALIGNMENT);
        writer.write_header_and_metadata(&metadata, &tensor_infos)?;

        let mut f32_buffer = Vec::new();
        let mut quant_buffer = Vec::new();

        for item in &emit_items {
            let meta = match header.tensors.get(&item.source_name) {
                Some(m) => m,
                None => continue,
            };

            let target_type = plan
                .get_type_for_tensor(&item.gguf_name)
                .or_else(|| plan.get_type_for_tensor(&item.source_name))
                .unwrap_or(GgmlType::F16);

            let src_start = data_start_offset.saturating_add(meta.data_offsets.0);
            let src_end = data_start_offset.saturating_add(meta.data_offsets.1);

            if src_end > mmap.len() {
                return Err(BridgeError::BufferSizeMismatch {
                    expected: src_end,
                    found: mmap.len(),
                });
            }

            let permute_heads = if !rope_is_neox {
                if item.source_name.ends_with("q_proj.weight")
                    || item.source_name.ends_with("q_proj.bias")
                    || item.gguf_name.ends_with("attn_q.weight")
                    || item.gguf_name.ends_with("attn_q.bias")
                {
                    n_attn_heads
                } else if item.source_name.ends_with("k_proj.weight")
                    || item.source_name.ends_with("k_proj.bias")
                    || item.gguf_name.ends_with("attn_k.weight")
                    || item.gguf_name.ends_with("attn_k.bias")
                {
                    n_kv_heads
                } else {
                    None
                }
            } else {
                None
            };

            let raw_src = &mmap[src_start..src_end];
            stream_write_tensor(
                &mut writer,
                &meta.dtype,
                raw_src,
                &meta.shape,
                permute_heads,
                target_type,
                &mut f32_buffer,
                &mut quant_buffer,
            )?;

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
        let mut sorted_sources: Vec<String> = multi.combined_tensors.keys().cloned().collect();
        sorted_sources.sort();

        let mut emit_items = Vec::with_capacity(sorted_sources.len() + 1);
        if !multi.combined_tensors.contains_key("lm_head.weight")
            && multi
                .combined_tensors
                .contains_key("model.embed_tokens.weight")
        {
            emit_items.push(EmitTensorItem {
                source_name: "model.embed_tokens.weight".to_string(),
                gguf_name: "output.weight".to_string(),
            });
        }

        for src in sorted_sources {
            let gguf_name = map_hf_to_gguf_tensor_name(&src);
            emit_items.push(EmitTensorItem {
                source_name: src,
                gguf_name,
            });
        }

        let mut tensor_infos = Vec::with_capacity(emit_items.len());
        let mut running_data_offset = 0u64;

        for item in &emit_items {
            let meta = match multi.combined_tensors.get(&item.source_name) {
                Some(m) => m,
                None => continue,
            };

            let target_type = plan
                .get_type_for_tensor(&item.gguf_name)
                .or_else(|| plan.get_type_for_tensor(&item.source_name))
                .unwrap_or(GgmlType::F16);
            let elem_count = meta.element_count();
            let size_bytes = target_type.bytes_for_elements(elem_count)? as u64;

            let dims: Vec<u64> = meta.shape.iter().rev().map(|&d| d as u64).collect();

            tensor_infos.push(GgufTensorInfo {
                name: item.gguf_name.clone(),
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
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        );
        metadata.insert(
            "general.alignment".to_string(),
            GgufValue::Uint32(GGUF_DEFAULT_ALIGNMENT as u32),
        );

        let parent_dir = p.parent().unwrap_or(Path::new("."));
        let config_path = parent_dir.join("config.json");
        let mut target_vocab_size = None;
        let mut n_attn_heads = None;
        let mut n_kv_heads = None;
        let mut rope_is_neox = true;
        if config_path.exists() {
            let cfg = Self::load_config_json_metadata(&config_path);
            metadata.extend(cfg.metadata);
            target_vocab_size = cfg.vocab_size;
            n_attn_heads = cfg.n_attn_heads;
            n_kv_heads = cfg.n_kv_heads;
            rope_is_neox = cfg.rope_is_neox;
        }

        let tokenizer_path = parent_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            let tok_cfg_path = parent_dir.join("tokenizer_config.json");
            let tok_cfg = if tok_cfg_path.exists() {
                Some(tok_cfg_path.as_path())
            } else {
                None
            };
            metadata.extend(Self::load_tokenizer_metadata(
                &tokenizer_path,
                tok_cfg,
                target_vocab_size,
            ));
        }

        if !metadata.contains_key("general.architecture") {
            metadata.insert(
                "general.architecture".to_string(),
                GgufValue::String("llama".to_string()),
            );
        }

        let out_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(output_path)?;

        let buf_writer = BufWriter::with_capacity(4 * 1024 * 1024, out_file);
        let mut writer = GgufWriter::new(buf_writer, GGUF_DEFAULT_ALIGNMENT);
        writer.write_header_and_metadata(&metadata, &tensor_infos)?;

        let mut f32_buffer = Vec::new();
        let mut quant_buffer = Vec::new();
        let mut current_shard: Option<(String, memmap2::Mmap, usize)> = None;

        for item in &emit_items {
            let meta = match multi.combined_tensors.get(&item.source_name) {
                Some(m) => m,
                None => continue,
            };
            let shard_name = match multi.tensor_to_shard.get(&item.source_name) {
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

            let target_type = plan
                .get_type_for_tensor(&item.gguf_name)
                .or_else(|| plan.get_type_for_tensor(&item.source_name))
                .unwrap_or(GgmlType::F16);
            let src_start = data_start_offset.saturating_add(meta.data_offsets.0);
            let src_end = data_start_offset.saturating_add(meta.data_offsets.1);

            if src_end > smmap.len() {
                return Err(BridgeError::BufferSizeMismatch {
                    expected: src_end,
                    found: smmap.len(),
                });
            }

            let permute_heads = if !rope_is_neox {
                if item.source_name.ends_with("q_proj.weight")
                    || item.source_name.ends_with("q_proj.bias")
                    || item.gguf_name.ends_with("attn_q.weight")
                    || item.gguf_name.ends_with("attn_q.bias")
                {
                    n_attn_heads
                } else if item.source_name.ends_with("k_proj.weight")
                    || item.source_name.ends_with("k_proj.bias")
                    || item.gguf_name.ends_with("attn_k.weight")
                    || item.gguf_name.ends_with("attn_k.bias")
                {
                    n_kv_heads
                } else {
                    None
                }
            } else {
                None
            };

            let raw_src = &smmap[src_start..src_end];
            stream_write_tensor(
                &mut writer,
                &meta.dtype,
                raw_src,
                &meta.shape,
                permute_heads,
                target_type,
                &mut f32_buffer,
                &mut quant_buffer,
            )?;

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

    fn load_config_json_metadata(config_path: &Path) -> ConfigMeta {
        let mut map = HashMap::new();
        let content = match std::fs::read_to_string(config_path) {
            Ok(c) => c,
            Err(_) => return ConfigMeta::default(),
        };

        let val: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => return ConfigMeta::default(),
        };

        let arch = match val.get("model_type").and_then(|v| v.as_str()) {
            Some(t) => t.to_string(),
            None => "llama".to_string(),
        };

        map.insert(
            "general.architecture".to_string(),
            GgufValue::String(arch.clone()),
        );

        if let Some(n) = val.get("num_hidden_layers").and_then(|v| v.as_u64()) {
            map.insert(format!("{arch}.block_count"), GgufValue::Uint32(n as u32));
        }
        if let Some(n) = val.get("max_position_embeddings").and_then(|v| v.as_u64()) {
            map.insert(
                format!("{arch}.context_length"),
                GgufValue::Uint32(n as u32),
            );
        }
        if let Some(n) = val.get("hidden_size").and_then(|v| v.as_u64()) {
            map.insert(
                format!("{arch}.embedding_length"),
                GgufValue::Uint32(n as u32),
            );
        }
        if let Some(n) = val.get("intermediate_size").and_then(|v| v.as_u64()) {
            map.insert(
                format!("{arch}.feed_forward_length"),
                GgufValue::Uint32(n as u32),
            );
        }

        let n_attn_heads = val
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        if let Some(n) = n_attn_heads {
            map.insert(
                format!("{arch}.attention.head_count"),
                GgufValue::Uint32(n as u32),
            );
        }

        let n_kv_heads = val
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        if let Some(n) = n_kv_heads {
            map.insert(
                format!("{arch}.attention.head_count_kv"),
                GgufValue::Uint32(n as u32),
            );
        }

        if let (Some(h), Some(heads)) = (
            val.get("hidden_size").and_then(|v| v.as_u64()),
            val.get("num_attention_heads").and_then(|v| v.as_u64()),
        ) {
            if let Some(dim) = h.checked_div(heads) {
                map.insert(
                    format!("{arch}.rope.dimension_count"),
                    GgufValue::Uint32(dim as u32),
                );
            }
        }

        if let Some(f) = val.get("rope_theta").and_then(|v| v.as_f64()) {
            map.insert(
                format!("{arch}.rope.freq_base"),
                GgufValue::Float32(f as f32),
            );
        }
        if let Some(f) = val.get("rms_norm_eps").and_then(|v| v.as_f64()) {
            map.insert(
                format!("{arch}.attention.layer_norm_rms_epsilon"),
                GgufValue::Float32(f as f32),
            );
        }

        let vocab_size = val
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let rope_is_neox = val
            .get("rope_is_neox_style")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        ConfigMeta {
            metadata: map,
            vocab_size,
            n_attn_heads,
            n_kv_heads,
            rope_is_neox,
        }
    }

    fn load_tokenizer_metadata(
        tok_path: &Path,
        tok_cfg_path: Option<&Path>,
        target_vocab_size: Option<usize>,
    ) -> HashMap<String, GgufValue> {
        let mut map = HashMap::new();
        let content = match std::fs::read_to_string(tok_path) {
            Ok(c) => c,
            Err(_) => return map,
        };

        let val: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => return map,
        };

        let model_type = val
            .get("model")
            .and_then(|m| m.get("type"))
            .and_then(|t| t.as_str())
            .unwrap_or("BPE");

        let (ggml_model, ggml_pre) = if model_type.eq_ignore_ascii_case("bpe") {
            ("gpt2", "qwen2")
        } else {
            ("llama", "default")
        };

        map.insert(
            "tokenizer.ggml.model".to_string(),
            GgufValue::String(ggml_model.to_string()),
        );
        map.insert(
            "tokenizer.ggml.pre".to_string(),
            GgufValue::String(ggml_pre.to_string()),
        );

        let mut max_id: usize = 0;
        let mut token_map: HashMap<usize, (String, bool)> = HashMap::new();

        if let Some(vocab_obj) = val
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| v.as_object())
        {
            for (tok, id_val) in vocab_obj {
                if let Some(id) = id_val.as_u64().map(|x| x as usize) {
                    if id > max_id {
                        max_id = id;
                    }
                    token_map.insert(id, (tok.clone(), false));
                }
            }
        }

        if let Some(added_arr) = val.get("added_tokens").and_then(|a| a.as_array()) {
            for item in added_arr {
                let id = item.get("id").and_then(|i| i.as_u64()).map(|x| x as usize);
                let content = item.get("content").and_then(|c| c.as_str());
                let special = item
                    .get("special")
                    .and_then(|s| s.as_bool())
                    .unwrap_or(false);
                if let (Some(id), Some(content)) = (id, content) {
                    if id > max_id {
                        max_id = id;
                    }
                    token_map.insert(id, (content.to_string(), special));
                }
            }
        }

        if !token_map.is_empty() {
            let total = match target_vocab_size {
                Some(v) => (max_id + 1).max(v),
                None => max_id + 1,
            };

            let mut tokens = Vec::with_capacity(total);
            let mut token_types = Vec::with_capacity(total);

            for id in 0..total {
                if let Some((tok_str, is_special)) = token_map.get(&id) {
                    tokens.push(GgufValue::String(tok_str.clone()));
                    let t_type = if *is_special { 3i32 } else { 1i32 };
                    token_types.push(GgufValue::Int32(t_type));
                } else {
                    tokens.push(GgufValue::String(format!("[PAD{id}]")));
                    token_types.push(GgufValue::Int32(5i32));
                }
            }

            map.insert(
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array(tokens),
            );
            map.insert(
                "tokenizer.ggml.token_type".to_string(),
                GgufValue::Array(token_types),
            );
        }

        if let Some(merges_arr) = val
            .get("model")
            .and_then(|m| m.get("merges"))
            .and_then(|m| m.as_array())
        {
            let mut merges = Vec::with_capacity(merges_arr.len());
            for item in merges_arr {
                if let Some(s) = item.as_str() {
                    merges.push(GgufValue::String(s.to_string()));
                }
            }
            if !merges.is_empty() {
                map.insert(
                    "tokenizer.ggml.merges".to_string(),
                    GgufValue::Array(merges),
                );
            }
        }

        map.insert(
            "tokenizer.ggml.add_bos_token".to_string(),
            GgufValue::Bool(false),
        );

        if let Some(cfg_path) = tok_cfg_path {
            if let Ok(cfg_content) = std::fs::read_to_string(cfg_path) {
                if let Ok(cfg_val) = serde_json::from_str::<serde_json::Value>(&cfg_content) {
                    if let Some(chat_tmpl) = cfg_val.get("chat_template").and_then(|t| t.as_str()) {
                        map.insert(
                            "tokenizer.chat_template".to_string(),
                            GgufValue::String(chat_tmpl.to_string()),
                        );
                    }
                    if let Some(add_bos) = cfg_val.get("add_bos_token").and_then(|b| b.as_bool()) {
                        map.insert(
                            "tokenizer.ggml.add_bos_token".to_string(),
                            GgufValue::Bool(add_bos),
                        );
                    }
                }
            }
        }

        for (id, (tok_str, _)) in &token_map {
            if tok_str == "<|im_end|>" {
                map.insert(
                    "tokenizer.ggml.eos_token_id".to_string(),
                    GgufValue::Uint32(*id as u32),
                );
            } else if tok_str == "<|endoftext|>" {
                map.insert(
                    "tokenizer.ggml.padding_token_id".to_string(),
                    GgufValue::Uint32(*id as u32),
                );
                map.insert(
                    "tokenizer.ggml.bos_token_id".to_string(),
                    GgufValue::Uint32(*id as u32),
                );
            }
        }

        map
    }
}

#[allow(clippy::too_many_arguments)]
fn stream_write_tensor<W: std::io::Write + std::io::Seek>(
    writer: &mut GgufWriter<W>,
    src_dtype: &str,
    raw_src: &[u8],
    shape: &[usize],
    permute_heads: Option<usize>,
    target_type: GgmlType,
    f32_buf: &mut Vec<f32>,
    quant_buf: &mut Vec<u8>,
) -> Result<(), BridgeError> {
    let dtype_upper = src_dtype.to_ascii_uppercase();

    if permute_heads.is_some() {
        quantize_raw_tensor_with_permute(
            src_dtype,
            raw_src,
            shape,
            permute_heads,
            target_type,
            quant_buf,
        )?;
        writer.write_tensor_data(quant_buf)?;
        quant_buf.clear();
        quant_buf.shrink_to(64 * 1024);
        return Ok(());
    }

    let elem_count = match dtype_upper.as_str() {
        "F16" | "BF16" => raw_src.len() / 2,
        "F32" => raw_src.len() / 4,
        other => return Err(BridgeError::UnsupportedSourceDtype(other.to_string())),
    };

    let src_elem_bytes = match dtype_upper.as_str() {
        "F16" | "BF16" => 2,
        "F32" => 4,
        _ => 4,
    };

    if (dtype_upper == "F16" && target_type == GgmlType::F16)
        || (dtype_upper == "F32" && target_type == GgmlType::F32)
    {
        writer.write_tensor_data(raw_src)?;
        return Ok(());
    }

    const CHUNK_ELEMS: usize = 65536;
    let mut offset_elems = 0;

    while offset_elems < elem_count {
        let n = (elem_count - offset_elems).min(CHUNK_ELEMS);
        let src_start = offset_elems * src_elem_bytes;
        let src_end = src_start + n * src_elem_bytes;
        let raw_chunk = &raw_src[src_start..src_end];

        f32_buf.resize(n, 0.0);
        match dtype_upper.as_str() {
            "F16" => {
                for (i, c) in raw_chunk.as_chunks::<2>().0.iter().enumerate() {
                    f32_buf[i] = f16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
            "BF16" => {
                for (i, c) in raw_chunk.as_chunks::<2>().0.iter().enumerate() {
                    let u = LittleEndian::read_u16(c);
                    f32_buf[i] = f32::from_bits((u as u32) << 16);
                }
            }
            _ => {
                for (i, c) in raw_chunk.as_chunks::<4>().0.iter().enumerate() {
                    f32_buf[i] = LittleEndian::read_f32(c);
                }
            }
        }

        quantize_f32_slice(f32_buf, target_type, quant_buf)?;
        writer.write_tensor_data_chunk(quant_buf)?;
        offset_elems += n;
    }

    writer.finish_tensor_data()?;
    quant_buf.clear();
    quant_buf.shrink_to(64 * 1024);
    f32_buf.clear();
    f32_buf.shrink_to(64 * 1024);
    Ok(())
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
        assert_eq!(gguf_hdr.tensors[0].name, "blk.0.attn_q.weight");
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
