use crate::error::BridgeError;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use quant_core::types::GgmlType;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

impl GgufTensorInfo {
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), BridgeError> {
        let name_bytes = self.name.as_bytes();
        w.write_u64::<LittleEndian>(name_bytes.len() as u64)?;
        w.write_all(name_bytes)?;

        w.write_u32::<LittleEndian>(self.dimensions.len() as u32)?;
        for &dim in &self.dimensions {
            w.write_u64::<LittleEndian>(dim)?;
        }

        w.write_u32::<LittleEndian>(self.ggml_type as u32)?;
        w.write_u64::<LittleEndian>(self.offset)?;
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> Result<Self, BridgeError> {
        let name_len = r.read_u64::<LittleEndian>()? as usize;
        let mut name_buf = vec![0u8; name_len];
        r.read_exact(&mut name_buf)?;
        let name = String::from_utf8(name_buf)
            .map_err(|_| BridgeError::QuantizeError("invalid utf-8 in tensor name".to_string()))?;

        let n_dims = r.read_u32::<LittleEndian>()? as usize;
        let mut dimensions = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dimensions.push(r.read_u64::<LittleEndian>()?);
        }

        let type_raw = r.read_u32::<LittleEndian>()?;
        let ggml_type = GgmlType::from_u32(type_raw)?;
        let offset = r.read_u64::<LittleEndian>()?;

        Ok(Self {
            name,
            dimensions,
            ggml_type,
            offset,
        })
    }
}
