use crate::error::BridgeError;
use byteorder::{ByteOrder, LittleEndian};
use half::f16;
use quant_core::types::GgmlType;

pub fn quantize_f32_to_f16(src: &[f32], dst: &mut [u8]) -> Result<usize, BridgeError> {
    let required = src.len().saturating_mul(2);
    if dst.len() < required {
        return Err(BridgeError::BufferSizeMismatch {
            expected: required,
            found: dst.len(),
        });
    }

    for (i, &val) in src.iter().enumerate() {
        let h = f16::from_f32(val);
        let out_idx = i * 2;
        let bytes = h.to_le_bytes();
        dst[out_idx] = bytes[0];
        dst[out_idx + 1] = bytes[1];
    }

    Ok(required)
}

pub fn quantize_f32_to_q8_0(src: &[f32], dst: &mut [u8]) -> Result<usize, BridgeError> {
    if !src.len().is_multiple_of(32) {
        return Err(BridgeError::QuantizeError(format!(
            "element count {} is not a multiple of 32 for Q8_0",
            src.len()
        )));
    }

    let nb = src.len() / 32;
    let required = nb.saturating_mul(34);
    if dst.len() < required {
        return Err(BridgeError::BufferSizeMismatch {
            expected: required,
            found: dst.len(),
        });
    }

    let mut dst_ptr = 0usize;

    for chunk in src.as_chunks::<32>().0 {
        let mut amax = 0.0f32;
        for &x in chunk {
            let ax = x.abs();
            if ax > amax {
                amax = ax;
            }
        }

        let d = amax / 127.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
        let d_f16 = f16::from_f32(d);

        let d_bytes = d_f16.to_le_bytes();
        dst[dst_ptr] = d_bytes[0];
        dst[dst_ptr + 1] = d_bytes[1];
        dst_ptr += 2;

        for &x in chunk {
            let v = (x * id).round();
            let q = v.clamp(-128.0, 127.0) as i8;
            dst[dst_ptr] = q as u8;
            dst_ptr += 1;
        }
    }

    Ok(required)
}

pub fn quantize_f32_to_q4_0(src: &[f32], dst: &mut [u8]) -> Result<usize, BridgeError> {
    if !src.len().is_multiple_of(32) {
        return Err(BridgeError::QuantizeError(format!(
            "element count {} is not a multiple of 32 for Q4_0",
            src.len()
        )));
    }

    let nb = src.len() / 32;
    let required = nb.saturating_mul(18);
    if dst.len() < required {
        return Err(BridgeError::BufferSizeMismatch {
            expected: required,
            found: dst.len(),
        });
    }

    let mut dst_ptr = 0usize;

    for chunk in src.as_chunks::<32>().0 {
        let mut amax = 0.0f32;
        for &x in chunk {
            let ax = x.abs();
            if ax > amax {
                amax = ax;
            }
        }

        let d = amax / -8.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
        let d_f16 = f16::from_f32(d);

        let d_bytes = d_f16.to_le_bytes();
        dst[dst_ptr] = d_bytes[0];
        dst[dst_ptr + 1] = d_bytes[1];
        dst_ptr += 2;

        for i in 0..16 {
            let v0 = (chunk[i] * id).round();
            let q0 = ((v0 + 8.5f32).clamp(0.0, 15.0) as u8) & 0x0F;

            let v1 = (chunk[i + 16] * id).round();
            let q1 = ((v1 + 8.5f32).clamp(0.0, 15.0) as u8) & 0x0F;

            dst[dst_ptr] = q0 | (q1 << 4);
            dst_ptr += 1;
        }
    }

    Ok(required)
}

pub fn quantize_raw_tensor(
    src_dtype: &str,
    raw_bytes: &[u8],
    target_type: GgmlType,
    dst: &mut Vec<u8>,
) -> Result<(), BridgeError> {
    let dtype_upper = src_dtype.to_ascii_uppercase();

    match (dtype_upper.as_str(), target_type) {
        ("F16", GgmlType::F16) => {
            dst.clear();
            dst.extend_from_slice(raw_bytes);
            Ok(())
        }
        ("BF16", GgmlType::F16) => {
            let elem_count = raw_bytes.len() / 2;
            dst.resize(elem_count * 2, 0);
            for i in 0..elem_count {
                let u = LittleEndian::read_u16(&raw_bytes[i * 2..(i + 1) * 2]);
                let f = f32::from_bits((u as u32) << 16);
                let h = f16::from_f32(f);
                let b = h.to_le_bytes();
                dst[i * 2] = b[0];
                dst[i * 2 + 1] = b[1];
            }
            Ok(())
        }
        ("F32", GgmlType::F16) => {
            let elem_count = raw_bytes.len() / 4;
            dst.resize(elem_count * 2, 0);
            let mut f32_buf = vec![0.0f32; elem_count];
            for i in 0..elem_count {
                f32_buf[i] = LittleEndian::read_f32(&raw_bytes[i * 4..(i + 1) * 4]);
            }
            quantize_f32_to_f16(&f32_buf, dst)?;
            Ok(())
        }
        ("F32", GgmlType::Q8_0) => {
            let elem_count = raw_bytes.len() / 4;
            let mut f32_buf = vec![0.0f32; elem_count];
            for i in 0..elem_count {
                f32_buf[i] = LittleEndian::read_f32(&raw_bytes[i * 4..(i + 1) * 4]);
            }
            let out_size = target_type.bytes_for_elements(elem_count)?;
            dst.resize(out_size, 0);
            quantize_f32_to_q8_0(&f32_buf, dst)?;
            Ok(())
        }
        ("F16", GgmlType::Q8_0) => {
            let elem_count = raw_bytes.len() / 2;
            let mut f32_buf = vec![0.0f32; elem_count];
            for i in 0..elem_count {
                let b = [raw_bytes[i * 2], raw_bytes[i * 2 + 1]];
                f32_buf[i] = f16::from_le_bytes(b).to_f32();
            }
            let out_size = target_type.bytes_for_elements(elem_count)?;
            dst.resize(out_size, 0);
            quantize_f32_to_q8_0(&f32_buf, dst)?;
            Ok(())
        }
        ("BF16", GgmlType::Q8_0) => {
            let elem_count = raw_bytes.len() / 2;
            let mut f32_buf = vec![0.0f32; elem_count];
            for i in 0..elem_count {
                let u = LittleEndian::read_u16(&raw_bytes[i * 2..(i + 1) * 2]);
                f32_buf[i] = f32::from_bits((u as u32) << 16);
            }
            let out_size = target_type.bytes_for_elements(elem_count)?;
            dst.resize(out_size, 0);
            quantize_f32_to_q8_0(&f32_buf, dst)?;
            Ok(())
        }
        ("F32", GgmlType::Q4_0) => {
            let elem_count = raw_bytes.len() / 4;
            let mut f32_buf = vec![0.0f32; elem_count];
            for i in 0..elem_count {
                f32_buf[i] = LittleEndian::read_f32(&raw_bytes[i * 4..(i + 1) * 4]);
            }
            let out_size = target_type.bytes_for_elements(elem_count)?;
            dst.resize(out_size, 0);
            quantize_f32_to_q4_0(&f32_buf, dst)?;
            Ok(())
        }
        ("F16", GgmlType::Q4_0) => {
            let elem_count = raw_bytes.len() / 2;
            let mut f32_buf = vec![0.0f32; elem_count];
            for i in 0..elem_count {
                let b = [raw_bytes[i * 2], raw_bytes[i * 2 + 1]];
                f32_buf[i] = f16::from_le_bytes(b).to_f32();
            }
            let out_size = target_type.bytes_for_elements(elem_count)?;
            dst.resize(out_size, 0);
            quantize_f32_to_q4_0(&f32_buf, dst)?;
            Ok(())
        }
        ("BF16", GgmlType::Q4_0) => {
            let elem_count = raw_bytes.len() / 2;
            let mut f32_buf = vec![0.0f32; elem_count];
            for i in 0..elem_count {
                let u = LittleEndian::read_u16(&raw_bytes[i * 2..(i + 1) * 2]);
                f32_buf[i] = f32::from_bits((u as u32) << 16);
            }
            let out_size = target_type.bytes_for_elements(elem_count)?;
            dst.resize(out_size, 0);
            quantize_f32_to_q4_0(&f32_buf, dst)?;
            Ok(())
        }
        _ => {
            let elem_count = match dtype_upper.as_str() {
                "F16" | "BF16" => raw_bytes.len() / 2,
                "F32" => raw_bytes.len() / 4,
                other => return Err(BridgeError::UnsupportedSourceDtype(other.to_string())),
            };
            let mut f32_buf = vec![0.0f32; elem_count];
            match dtype_upper.as_str() {
                "F16" => {
                    for i in 0..elem_count {
                        f32_buf[i] =
                            f16::from_le_bytes([raw_bytes[i * 2], raw_bytes[i * 2 + 1]]).to_f32();
                    }
                }
                "BF16" => {
                    for i in 0..elem_count {
                        let u = LittleEndian::read_u16(&raw_bytes[i * 2..(i + 1) * 2]);
                        f32_buf[i] = f32::from_bits((u as u32) << 16);
                    }
                }
                _ => {
                    for i in 0..elem_count {
                        f32_buf[i] = LittleEndian::read_f32(&raw_bytes[i * 4..(i + 1) * 4]);
                    }
                }
            }

            let out_size = target_type.bytes_for_elements(elem_count)?;
            dst.resize(out_size, 0);
            quantize_f32_to_q8_0(&f32_buf, dst)?;
            Ok(())
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_f32_to_f16() {
        let src = vec![1.0f32, -2.5f32, 0.0f32, 65504.0f32];
        let mut dst = vec![0u8; 8];
        let written = quantize_f32_to_f16(&src, &mut dst).unwrap();
        assert_eq!(written, 8);

        let h0 = f16::from_le_bytes([dst[0], dst[1]]);
        assert!((h0.to_f32() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_quantize_f32_to_q8_0() {
        let src: Vec<f32> = (0..32).map(|i| (i as f32) - 16.0).collect();
        let mut dst = vec![0u8; 34];
        let written = quantize_f32_to_q8_0(&src, &mut dst).unwrap();
        assert_eq!(written, 34);

        let d = f16::from_le_bytes([dst[0], dst[1]]).to_f32();
        assert!((d - (16.0 / 127.0)).abs() < 1e-3);
    }

    #[test]
    fn test_quantize_f32_to_q4_0() {
        let src: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let mut dst = vec![0u8; 18];
        let written = quantize_f32_to_q4_0(&src, &mut dst).unwrap();
        assert_eq!(written, 18);
    }
}
