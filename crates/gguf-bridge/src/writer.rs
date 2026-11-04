use crate::constants::*;
use crate::error::BridgeError;
use crate::metadata::GgufValue;
use crate::tensor_info::GgufTensorInfo;
use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Seek, Write};

pub struct GgufWriter<W: Write + Seek> {
    writer: W,
    alignment: usize,
    tensor_infos: Vec<GgufTensorInfo>,
    current_data_offset: u64,
}

impl<W: Write + Seek> GgufWriter<W> {
    pub fn new(writer: W, alignment: usize) -> Self {
        Self {
            writer,
            alignment: if alignment == 0 {
                GGUF_DEFAULT_ALIGNMENT
            } else {
                alignment
            },
            tensor_infos: Vec::new(),
            current_data_offset: 0,
        }
    }

    pub fn write_header_and_metadata(
        &mut self,
        metadata: &HashMap<String, GgufValue>,
        tensor_infos: &[GgufTensorInfo],
    ) -> Result<u64, BridgeError> {
        self.writer.write_all(&GGUF_MAGIC)?;
        self.writer.write_u32::<LittleEndian>(GGUF_VERSION)?;
        self.writer
            .write_u64::<LittleEndian>(tensor_infos.len() as u64)?;
        self.writer
            .write_u64::<LittleEndian>(metadata.len() as u64)?;

        for (key, val) in metadata {
            let key_bytes = key.as_bytes();
            self.writer
                .write_u64::<LittleEndian>(key_bytes.len() as u64)?;
            self.writer.write_all(key_bytes)?;
            self.writer.write_u32::<LittleEndian>(val.type_tag())?;
            val.write_to(&mut self.writer)?;
        }

        for info in tensor_infos {
            info.write_to(&mut self.writer)?;
        }

        let cur_pos = self.writer.stream_position()?;
        let rem = (cur_pos as usize) % self.alignment;
        if rem != 0 {
            let pad = self.alignment - rem;
            let zeros = vec![0u8; pad];
            self.writer.write_all(&zeros)?;
        }

        let data_start = self.writer.stream_position()?;
        self.tensor_infos = tensor_infos.to_vec();
        self.current_data_offset = 0;
        Ok(data_start)
    }

    pub fn write_tensor_data(&mut self, data: &[u8]) -> Result<u64, BridgeError> {
        let offset = self.current_data_offset;
        self.writer.write_all(data)?;
        self.current_data_offset += data.len() as u64;

        let rem = (self.current_data_offset as usize) % self.alignment;
        if rem != 0 {
            let pad = self.alignment - rem;
            let zeros = vec![0u8; pad];
            self.writer.write_all(&zeros)?;
            self.current_data_offset += pad as u64;
        }

        self.writer.flush()?;
        Ok(offset)
    }
}
