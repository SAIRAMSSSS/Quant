use crate::error::CoreError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

impl TensorMetadata {
    pub fn element_count(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape
                .iter()
                .copied()
                .fold(1usize, |acc, dim| acc.saturating_mul(dim))
        }
    }

    pub fn byte_size(&self) -> usize {
        self.data_offsets.1.saturating_sub(self.data_offsets.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsHeader {
    pub tensors: HashMap<String, TensorMetadata>,
    pub metadata: HashMap<String, String>,
    pub header_size: u64,
}

#[derive(Deserialize)]
struct RawTensorEntry {
    dtype: Option<String>,
    shape: Option<Vec<usize>>,
    data_offsets: Option<Vec<usize>>,
}

pub fn parse_header_bytes(bytes: &[u8]) -> Result<SafetensorsHeader, CoreError> {
    if bytes.len() < 8 {
        return Err(CoreError::SafetensorsHeaderTooSmall(bytes.len()));
    }

    let header_len_bytes: [u8; 8] = bytes[0..8]
        .try_into()
        .map_err(|_| CoreError::SafetensorsHeaderTooSmall(bytes.len()))?;
    let header_len = u64::from_le_bytes(header_len_bytes);

    if header_len == 0 || header_len > 100 * 1024 * 1024 {
        return Err(CoreError::SafetensorsInvalidLength(header_len));
    }

    let total_required = 8usize.saturating_add(header_len as usize);
    if bytes.len() < total_required {
        return Err(CoreError::SafetensorsHeaderTooSmall(bytes.len()));
    }

    let json_slice = &bytes[8..total_required];
    let raw_map: HashMap<String, serde_json::Value> = serde_json::from_slice(json_slice)?;

    let mut tensors = HashMap::new();
    let mut metadata = HashMap::new();

    for (key, value) in raw_map {
        if key == "__metadata__" {
            if let Ok(meta_map) =
                serde_json::from_value::<HashMap<String, serde_json::Value>>(value)
            {
                for (mk, mv) in meta_map {
                    if let Some(s) = mv.as_str() {
                        metadata.insert(mk, s.to_string());
                    } else {
                        metadata.insert(mk, mv.to_string());
                    }
                }
            }
            continue;
        }

        let entry: RawTensorEntry = serde_json::from_value(value)?;
        let dtype = entry.dtype.ok_or_else(|| CoreError::SafetensorsCorrupt {
            detail: format!("tensor '{key}' missing dtype"),
        })?;
        let shape = entry.shape.ok_or_else(|| CoreError::SafetensorsCorrupt {
            detail: format!("tensor '{key}' missing shape"),
        })?;
        let offsets = entry
            .data_offsets
            .ok_or_else(|| CoreError::SafetensorsCorrupt {
                detail: format!("tensor '{key}' missing data_offsets"),
            })?;

        if offsets.len() != 2 {
            return Err(CoreError::SafetensorsCorrupt {
                detail: format!(
                    "tensor '{key}' invalid data_offsets length: {}",
                    offsets.len()
                ),
            });
        }

        if offsets[0] > offsets[1] {
            return Err(CoreError::SafetensorsCorrupt {
                detail: format!(
                    "tensor '{key}' start offset {} exceeds end offset {}",
                    offsets[0], offsets[1]
                ),
            });
        }

        tensors.insert(
            key,
            TensorMetadata {
                dtype,
                shape,
                data_offsets: (offsets[0], offsets[1]),
            },
        );
    }

    Ok(SafetensorsHeader {
        tensors,
        metadata,
        header_size: header_len,
    })
}

pub fn parse_from_file<P: AsRef<Path>>(path: P) -> Result<SafetensorsHeader, CoreError> {
    let file = File::open(path)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    parse_header_bytes(&mmap)
}

pub fn fetch_remote_header(url: &str) -> Result<SafetensorsHeader, CoreError> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("quant-engine/0.1.0")
        .build()?;

    let res_len = client
        .get(url)
        .header(reqwest::header::RANGE, "bytes=0-7")
        .send()?;

    let status = res_len.status();
    if !status.is_success() && status.as_u16() != 206 {
        return Err(CoreError::HttpRangeRejected(status.as_u16()));
    }

    let len_bytes = res_len.bytes()?;
    if len_bytes.len() < 8 {
        return Err(CoreError::SafetensorsHeaderTooSmall(len_bytes.len()));
    }

    let header_len = u64::from_le_bytes([
        len_bytes[0],
        len_bytes[1],
        len_bytes[2],
        len_bytes[3],
        len_bytes[4],
        len_bytes[5],
        len_bytes[6],
        len_bytes[7],
    ]);

    if header_len == 0 || header_len > 100 * 1024 * 1024 {
        return Err(CoreError::SafetensorsInvalidLength(header_len));
    }

    let end_byte = 8u64.saturating_add(header_len).saturating_sub(1);
    let range_hdr = format!("bytes=8-{end_byte}");

    let res_json = client
        .get(url)
        .header(reqwest::header::RANGE, range_hdr)
        .send()?;

    let status_json = res_json.status();
    if !status_json.is_success() && status_json.as_u16() != 206 {
        return Err(CoreError::HttpRangeRejected(status_json.as_u16()));
    }

    let json_bytes = res_json.bytes()?;
    let mut combined = Vec::with_capacity(8 + json_bytes.len());
    combined.extend_from_slice(&header_len.to_le_bytes());
    combined.extend_from_slice(&json_bytes);

    parse_header_bytes(&combined)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsIndex {
    pub metadata: HashMap<String, serde_json::Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn parse_index_bytes(bytes: &[u8]) -> Result<SafetensorsIndex, CoreError> {
    #[derive(Deserialize)]
    struct RawIndex {
        metadata: Option<HashMap<String, serde_json::Value>>,
        weight_map: Option<HashMap<String, String>>,
    }
    let raw: RawIndex = serde_json::from_slice(bytes)?;
    let weight_map = raw.weight_map.ok_or_else(|| {
        CoreError::SafetensorsIndexError("missing weight_map in safetensors index".to_string())
    })?;
    Ok(SafetensorsIndex {
        metadata: raw.metadata.unwrap_or_default(),
        weight_map,
    })
}

pub fn parse_index_file<P: AsRef<Path>>(path: P) -> Result<SafetensorsIndex, CoreError> {
    let bytes = std::fs::read(path)?;
    parse_index_bytes(&bytes)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiShardHeader {
    pub shards: HashMap<String, SafetensorsHeader>,
    pub tensor_to_shard: HashMap<String, String>,
    pub combined_tensors: HashMap<String, TensorMetadata>,
    pub combined_metadata: HashMap<String, String>,
}

pub fn load_multi_shard_headers<P: AsRef<Path>>(
    index_or_dir_path: P,
) -> Result<MultiShardHeader, CoreError> {
    let path = index_or_dir_path.as_ref();
    let (base_dir, index_file) = if path.is_dir() {
        let candidate = path.join("model.safetensors.index.json");
        if candidate.exists() {
            (path.to_path_buf(), candidate)
        } else {
            let mut entries = Vec::new();
            for entry in std::fs::read_dir(path)?.flatten() {
                let p = entry.path();
                if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                    entries.push(p);
                }
            }
            if entries.is_empty() {
                return Err(CoreError::SafetensorsShardNotFound(format!(
                    "no .safetensors shards found in directory {}",
                    path.display()
                )));
            }

            let mut shards = HashMap::new();
            let mut tensor_to_shard = HashMap::new();
            let mut combined_tensors = HashMap::new();
            let mut combined_metadata = HashMap::new();

            for shard_path in entries {
                let filename = shard_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let hdr = parse_from_file(&shard_path)?;
                for (tname, tmeta) in &hdr.tensors {
                    tensor_to_shard.insert(tname.clone(), filename.clone());
                    combined_tensors.insert(tname.clone(), tmeta.clone());
                }
                for (mk, mv) in &hdr.metadata {
                    combined_metadata.insert(mk.clone(), mv.clone());
                }
                shards.insert(filename, hdr);
            }

            return Ok(MultiShardHeader {
                shards,
                tensor_to_shard,
                combined_tensors,
                combined_metadata,
            });
        }
    } else {
        let parent = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        (parent, path.to_path_buf())
    };

    let index = parse_index_file(&index_file)?;
    let mut unique_shards = Vec::new();
    for shard_name in index.weight_map.values() {
        if !unique_shards.contains(shard_name) {
            unique_shards.push(shard_name.clone());
        }
    }

    let mut shards = HashMap::new();
    let mut tensor_to_shard = HashMap::new();
    let mut combined_tensors = HashMap::new();
    let mut combined_metadata = HashMap::new();

    for shard_name in unique_shards {
        let shard_path = base_dir.join(&shard_name);
        if !shard_path.exists() {
            return Err(CoreError::SafetensorsShardNotFound(format!(
                "shard file '{}' not found at {}",
                shard_name,
                shard_path.display()
            )));
        }
        let hdr = parse_from_file(&shard_path)?;
        for (mk, mv) in &hdr.metadata {
            combined_metadata.insert(mk.clone(), mv.clone());
        }
        shards.insert(shard_name.clone(), hdr);
    }

    for (tname, shard_name) in index.weight_map {
        tensor_to_shard.insert(tname.clone(), shard_name.clone());
        if let Some(shard_hdr) = shards.get(&shard_name) {
            if let Some(tmeta) = shard_hdr.tensors.get(&tname) {
                combined_tensors.insert(tname, tmeta.clone());
            }
        }
    }

    Ok(MultiShardHeader {
        shards,
        tensor_to_shard,
        combined_tensors,
        combined_metadata,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_safetensors_header() {
        let json = r#"{"__metadata__":{"format":"pt"},"model.embed_tokens.weight":{"dtype":"F16","shape":[32000,4096],"data_offsets":[0,262144000]}}"#;
        let json_bytes = json.as_bytes();
        let header_len = json_bytes.len() as u64;

        let mut buf = Vec::new();
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(json_bytes);

        let parsed = parse_header_bytes(&buf).unwrap();
        assert_eq!(parsed.header_size, header_len);
        assert_eq!(
            parsed.metadata.get("format").map(|s| s.as_str()),
            Some("pt")
        );

        let meta = parsed.tensors.get("model.embed_tokens.weight").unwrap();
        assert_eq!(meta.dtype, "F16");
        assert_eq!(meta.shape, vec![32000, 4096]);
        assert_eq!(meta.data_offsets, (0, 262_144_000));
        assert_eq!(meta.element_count(), 32000 * 4096);
        assert_eq!(meta.byte_size(), 262_144_000);
    }

    #[test]
    fn test_corrupt_offset_rejection() {
        let json = r#"{"bad.tensor":{"dtype":"F16","shape":[10],"data_offsets":[500,100]}}"#;
        let json_bytes = json.as_bytes();
        let header_len = json_bytes.len() as u64;

        let mut buf = Vec::new();
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(json_bytes);

        let res = parse_header_bytes(&buf);
        assert!(res.is_err());
    }

    #[test]
    fn test_parse_safetensors_index() {
        let index_json = r#"{
            "metadata": {"total_size": 1048576},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        let index = parse_index_bytes(index_json.as_bytes()).unwrap();
        assert_eq!(index.weight_map.len(), 2);
        assert_eq!(
            index
                .weight_map
                .get("model.embed_tokens.weight")
                .map(|s| s.as_str()),
            Some("model-00001-of-00002.safetensors")
        );
    }

    #[test]
    fn test_load_multi_shard_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let shard1_path = tmp.path().join("model-00001-of-00002.safetensors");
        let shard2_path = tmp.path().join("model-00002-of-00002.safetensors");
        let index_path = tmp.path().join("model.safetensors.index.json");

        let s1_json = r#"{"model.embed_tokens.weight":{"dtype":"F16","shape":[16,16],"data_offsets":[0,512]}}"#;
        let mut s1_buf = Vec::new();
        s1_buf.extend_from_slice(&(s1_json.len() as u64).to_le_bytes());
        s1_buf.extend_from_slice(s1_json.as_bytes());
        s1_buf.extend_from_slice(&vec![0u8; 512]);
        std::fs::write(&shard1_path, s1_buf).unwrap();

        let s2_json =
            r#"{"lm_head.weight":{"dtype":"F16","shape":[16,16],"data_offsets":[0,512]}}"#;
        let mut s2_buf = Vec::new();
        s2_buf.extend_from_slice(&(s2_json.len() as u64).to_le_bytes());
        s2_buf.extend_from_slice(s2_json.as_bytes());
        s2_buf.extend_from_slice(&vec![0u8; 512]);
        std::fs::write(&shard2_path, s2_buf).unwrap();

        let idx_content = r#"{
            "metadata": {"total_size": 1024},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        std::fs::write(&index_path, idx_content).unwrap();

        let multi = load_multi_shard_headers(tmp.path()).unwrap();
        assert_eq!(multi.combined_tensors.len(), 2);
        assert_eq!(multi.shards.len(), 2);
        assert!(multi
            .combined_tensors
            .contains_key("model.embed_tokens.weight"));
        assert!(multi.combined_tensors.contains_key("lm_head.weight"));
    }
}
