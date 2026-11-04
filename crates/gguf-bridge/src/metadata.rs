use crate::constants::*;
use crate::error::BridgeError;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn type_tag(&self) -> u32 {
        match self {
            Self::Uint8(_) => GGUF_TYPE_UINT8,
            Self::Int8(_) => GGUF_TYPE_INT8,
            Self::Uint16(_) => GGUF_TYPE_UINT16,
            Self::Int16(_) => GGUF_TYPE_INT16,
            Self::Uint32(_) => GGUF_TYPE_UINT32,
            Self::Int32(_) => GGUF_TYPE_INT32,
            Self::Float32(_) => GGUF_TYPE_FLOAT32,
            Self::Bool(_) => GGUF_TYPE_BOOL,
            Self::String(_) => GGUF_TYPE_STRING,
            Self::Array(_) => GGUF_TYPE_ARRAY,
            Self::Uint64(_) => GGUF_TYPE_UINT64,
            Self::Int64(_) => GGUF_TYPE_INT64,
            Self::Float64(_) => GGUF_TYPE_FLOAT64,
        }
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), BridgeError> {
        match self {
            Self::Uint8(v) => w.write_u8(*v)?,
            Self::Int8(v) => w.write_i8(*v)?,
            Self::Uint16(v) => w.write_u16::<LittleEndian>(*v)?,
            Self::Int16(v) => w.write_i16::<LittleEndian>(*v)?,
            Self::Uint32(v) => w.write_u32::<LittleEndian>(*v)?,
            Self::Int32(v) => w.write_i32::<LittleEndian>(*v)?,
            Self::Float32(v) => w.write_f32::<LittleEndian>(*v)?,
            Self::Bool(v) => w.write_u8(if *v { 1 } else { 0 })?,
            Self::String(s) => {
                let bytes = s.as_bytes();
                w.write_u64::<LittleEndian>(bytes.len() as u64)?;
                w.write_all(bytes)?;
            }
            Self::Array(arr) => {
                let elem_type = if let Some(first) = arr.first() {
                    first.type_tag()
                } else {
                    GGUF_TYPE_UINT32
                };
                w.write_u32::<LittleEndian>(elem_type)?;
                w.write_u64::<LittleEndian>(arr.len() as u64)?;
                for item in arr {
                    item.write_raw_element(w)?;
                }
            }
            Self::Uint64(v) => w.write_u64::<LittleEndian>(*v)?,
            Self::Int64(v) => w.write_i64::<LittleEndian>(*v)?,
            Self::Float64(v) => w.write_f64::<LittleEndian>(*v)?,
        }
        Ok(())
    }

    fn write_raw_element<W: Write>(&self, w: &mut W) -> Result<(), BridgeError> {
        match self {
            Self::Uint8(v) => w.write_u8(*v)?,
            Self::Int8(v) => w.write_i8(*v)?,
            Self::Uint16(v) => w.write_u16::<LittleEndian>(*v)?,
            Self::Int16(v) => w.write_i16::<LittleEndian>(*v)?,
            Self::Uint32(v) => w.write_u32::<LittleEndian>(*v)?,
            Self::Int32(v) => w.write_i32::<LittleEndian>(*v)?,
            Self::Float32(v) => w.write_f32::<LittleEndian>(*v)?,
            Self::Bool(v) => w.write_u8(if *v { 1 } else { 0 })?,
            Self::String(s) => {
                let bytes = s.as_bytes();
                w.write_u64::<LittleEndian>(bytes.len() as u64)?;
                w.write_all(bytes)?;
            }
            Self::Array(_) => {
                return Err(BridgeError::QuantizeError(
                    "nested arrays in gguf metadata are not supported".to_string(),
                ));
            }
            Self::Uint64(v) => w.write_u64::<LittleEndian>(*v)?,
            Self::Int64(v) => w.write_i64::<LittleEndian>(*v)?,
            Self::Float64(v) => w.write_f64::<LittleEndian>(*v)?,
        }
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R, type_tag: u32) -> Result<Self, BridgeError> {
        match type_tag {
            GGUF_TYPE_UINT8 => Ok(Self::Uint8(r.read_u8()?)),
            GGUF_TYPE_INT8 => Ok(Self::Int8(r.read_i8()?)),
            GGUF_TYPE_UINT16 => Ok(Self::Uint16(r.read_u16::<LittleEndian>()?)),
            GGUF_TYPE_INT16 => Ok(Self::Int16(r.read_i16::<LittleEndian>()?)),
            GGUF_TYPE_UINT32 => Ok(Self::Uint32(r.read_u32::<LittleEndian>()?)),
            GGUF_TYPE_INT32 => Ok(Self::Int32(r.read_i32::<LittleEndian>()?)),
            GGUF_TYPE_FLOAT32 => Ok(Self::Float32(r.read_f32::<LittleEndian>()?)),
            GGUF_TYPE_BOOL => Ok(Self::Bool(r.read_u8()? != 0)),
            GGUF_TYPE_STRING => {
                let len = r.read_u64::<LittleEndian>()? as usize;
                let mut buf = vec![0u8; len];
                r.read_exact(&mut buf)?;
                let s = String::from_utf8(buf).map_err(|_| {
                    BridgeError::QuantizeError("invalid utf-8 in gguf string".to_string())
                })?;
                Ok(Self::String(s))
            }
            GGUF_TYPE_ARRAY => {
                let elem_type = r.read_u32::<LittleEndian>()?;
                let count = r.read_u64::<LittleEndian>()? as usize;
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(Self::read_from(r, elem_type)?);
                }
                Ok(Self::Array(arr))
            }
            GGUF_TYPE_UINT64 => Ok(Self::Uint64(r.read_u64::<LittleEndian>()?)),
            GGUF_TYPE_INT64 => Ok(Self::Int64(r.read_i64::<LittleEndian>()?)),
            GGUF_TYPE_FLOAT64 => Ok(Self::Float64(r.read_f64::<LittleEndian>()?)),
            other => Err(BridgeError::InvalidMetadataType(other)),
        }
    }
}
