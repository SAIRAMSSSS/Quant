use crate::constants::*;
use crate::error::BridgeError;
use crate::metadata::GgufValue;
use crate::tensor_info::GgufTensorInfo;
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub alignment: usize,
    pub data_offset: u64,
}

pub fn read_gguf_header<R: Read + Seek>(r: &mut R) -> Result<GgufHeader, BridgeError> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != GGUF_MAGIC {
        return Err(BridgeError::InvalidMagic(magic));
    }

    let version = r.read_u32::<LittleEndian>()?;
    if version != GGUF_VERSION && version != 2 {
        return Err(BridgeError::UnsupportedVersion(version));
    }

    let tensor_count = r.read_u64::<LittleEndian>()?;
    let metadata_kv_count = r.read_u64::<LittleEndian>()?;

    let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);

    for _ in 0..metadata_kv_count {
        let key_len = r.read_u64::<LittleEndian>()? as usize;
        let mut key_buf = vec![0u8; key_len];
        r.read_exact(&mut key_buf)?;
        let key = String::from_utf8(key_buf)
            .map_err(|_| BridgeError::QuantizeError("invalid utf-8 in metadata key".to_string()))?;

        let val_type = r.read_u32::<LittleEndian>()?;
        let value = GgufValue::read_from(r, val_type)?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        tensors.push(GgufTensorInfo::read_from(r)?);
    }

    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::Uint32(val)) => *val as usize,
        Some(GgufValue::Uint64(val)) => *val as usize,
        _ => GGUF_DEFAULT_ALIGNMENT,
    };

    let cur_pos = r.stream_position()?;
    let rem = (cur_pos as usize) % alignment;
    let data_offset = if rem == 0 {
        cur_pos
    } else {
        cur_pos + (alignment - rem) as u64
    };

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_kv_count,
        metadata,
        tensors,
        alignment,
        data_offset,
    })
}

pub fn inspect_gguf_file<P: AsRef<Path>>(path: P) -> Result<GgufHeader, BridgeError> {
    let mut file = File::open(path)?;
    read_gguf_header(&mut file)
}
