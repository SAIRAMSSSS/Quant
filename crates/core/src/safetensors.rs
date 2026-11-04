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
        assert_eq!(meta.data_offsets, (0, 262144000));
        assert_eq!(meta.element_count(), 32000 * 4096);
        assert_eq!(meta.byte_size(), 262144000);
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
}
