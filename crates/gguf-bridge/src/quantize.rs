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

pub fn quantize_f32_to_q4_k(src: &[f32], dst: &mut [u8]) -> Result<usize, BridgeError> {
    if !src.len().is_multiple_of(256) {
        return Err(BridgeError::QuantizeError(format!(
            "element count {} is not a multiple of 256 for Q4_K",
            src.len()
        )));
    }

    let nb = src.len() / 256;
    let required = nb.saturating_mul(144);
    if dst.len() < required {
        return Err(BridgeError::BufferSizeMismatch {
            expected: required,
            found: dst.len(),
        });
    }

    let mut dst_ptr = 0usize;

    for chunk in src.as_chunks::<256>().0 {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &x in chunk {
            if x < min_val {
                min_val = x;
            }
            if x > max_val {
                max_val = x;
            }
        }

        let dmin = (-min_val).max(0.0f32);
        let range = max_val + dmin;
        let d = range / 15.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };

        let d_bytes = f16::from_f32(d).to_le_bytes();
        let dmin_bytes = f16::from_f32(dmin).to_le_bytes();

        dst[dst_ptr..dst_ptr + 2].copy_from_slice(&d_bytes);
        dst_ptr += 2;
        dst[dst_ptr..dst_ptr + 2].copy_from_slice(&dmin_bytes);
        dst_ptr += 2;

        dst[dst_ptr..dst_ptr + 12].fill(0);
        dst_ptr += 12;

        for i in 0..128 {
            let v0 = ((chunk[i] + dmin) * id).round().clamp(0.0, 15.0) as u8;
            let v1 = ((chunk[i + 128] + dmin) * id).round().clamp(0.0, 15.0) as u8;
            dst[dst_ptr + i] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
        }
        dst_ptr += 128;
    }

    Ok(required)
}

pub fn quantize_f32_to_q5_k(src: &[f32], dst: &mut [u8]) -> Result<usize, BridgeError> {
    if !src.len().is_multiple_of(256) {
        return Err(BridgeError::QuantizeError(format!(
            "element count {} is not a multiple of 256 for Q5_K",
            src.len()
        )));
    }

    let nb = src.len() / 256;
    let required = nb.saturating_mul(176);
    if dst.len() < required {
        return Err(BridgeError::BufferSizeMismatch {
            expected: required,
            found: dst.len(),
        });
    }

    let mut dst_ptr = 0usize;

    for chunk in src.as_chunks::<256>().0 {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &x in chunk {
            if x < min_val {
                min_val = x;
            }
            if x > max_val {
                max_val = x;
            }
        }

        let dmin = (-min_val).max(0.0f32);
        let range = max_val + dmin;
        let d = range / 31.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };

        let d_bytes = f16::from_f32(d).to_le_bytes();
        let dmin_bytes = f16::from_f32(dmin).to_le_bytes();

        dst[dst_ptr..dst_ptr + 2].copy_from_slice(&d_bytes);
        dst_ptr += 2;
        dst[dst_ptr..dst_ptr + 2].copy_from_slice(&dmin_bytes);
        dst_ptr += 2;

        dst[dst_ptr..dst_ptr + 12].fill(0);
        dst_ptr += 12;

        let qh_start = dst_ptr;
        dst[qh_start..qh_start + 32].fill(0);
        dst_ptr += 32;

        let qs_start = dst_ptr;
        for i in 0..128 {
            let v0 = ((chunk[i] + dmin) * id).round().clamp(0.0, 31.0) as u8;
            let v1 = ((chunk[i + 128] + dmin) * id).round().clamp(0.0, 31.0) as u8;

            dst[qs_start + i] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);

            let h0 = (v0 >> 4) & 1;
            let h1 = (v1 >> 4) & 1;

            dst[qh_start + (i / 8)] |= h0 << (i % 8);
            dst[qh_start + ((i + 128) / 8)] |= h1 << ((i + 128) % 8);
        }
        dst_ptr += 128;
    }

    Ok(required)
}

pub fn quantize_f32_to_q6_k(src: &[f32], dst: &mut [u8]) -> Result<usize, BridgeError> {
    if !src.len().is_multiple_of(256) {
        return Err(BridgeError::QuantizeError(format!(
            "element count {} is not a multiple of 256 for Q6_K",
            src.len()
        )));
    }

    let nb = src.len() / 256;
    let required = nb.saturating_mul(210);
    if dst.len() < required {
        return Err(BridgeError::BufferSizeMismatch {
            expected: required,
            found: dst.len(),
        });
    }

    let mut dst_ptr = 0usize;

    for chunk in src.as_chunks::<256>().0 {
        let mut amax = 0.0f32;
        for &x in chunk {
            let ax = x.abs();
            if ax > amax {
                amax = ax;
            }
        }

        let d = amax / 31.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };

        let ql_start = dst_ptr;
        let qh_start = ql_start + 128;
        let scales_start = qh_start + 64;
        let d_start = scales_start + 16;

        dst[qh_start..qh_start + 64].fill(0);
        dst[scales_start..scales_start + 16].fill(1);

        for i in 0..128 {
            let v0 = (chunk[i] * id).round().clamp(-32.0, 31.0) as i8;
            let u0 = (v0 + 32) as u8;
            let v1 = (chunk[i + 128] * id).round().clamp(-32.0, 31.0) as i8;
            let u1 = (v1 + 32) as u8;

            dst[ql_start + i] = (u0 & 0x0F) | ((u1 & 0x0F) << 4);

            let h0 = (u0 >> 4) & 0x03;
            let h1 = (u1 >> 4) & 0x03;

            dst[qh_start + (i / 4)] |= h0 << ((i % 4) * 2);
            dst[qh_start + ((i + 128) / 4)] |= h1 << (((i + 128) % 4) * 2);
        }

        let d_bytes = f16::from_f32(d).to_le_bytes();
        dst[d_start..d_start + 2].copy_from_slice(&d_bytes);

        dst_ptr += 210;
    }

    Ok(required)
}

pub fn quantize_f32_to_q2_k(src: &[f32], dst: &mut [u8]) -> Result<usize, BridgeError> {
    if !src.len().is_multiple_of(256) {
        return Err(BridgeError::QuantizeError(format!(
            "element count {} is not a multiple of 256 for Q2_K",
            src.len()
        )));
    }

    let nb = src.len() / 256;
    let required = nb.saturating_mul(84);
    if dst.len() < required {
        return Err(BridgeError::BufferSizeMismatch {
            expected: required,
            found: dst.len(),
        });
    }

    let mut dst_ptr = 0usize;

    for chunk in src.as_chunks::<256>().0 {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &x in chunk {
            if x < min_val {
                min_val = x;
            }
            if x > max_val {
                max_val = x;
            }
        }

        let dmin = (-min_val).max(0.0f32);
        let range = max_val + dmin;
        let d = range / 3.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };

        dst[dst_ptr..dst_ptr + 16].fill(0);
        dst_ptr += 16;

        let qs_start = dst_ptr;
        dst[qs_start..qs_start + 64].fill(0);
        for i in 0..64 {
            let q0 = ((chunk[i * 4] + dmin) * id).round().clamp(0.0, 3.0) as u8;
            let q1 = ((chunk[i * 4 + 1] + dmin) * id).round().clamp(0.0, 3.0) as u8;
            let q2 = ((chunk[i * 4 + 2] + dmin) * id).round().clamp(0.0, 3.0) as u8;
            let q3 = ((chunk[i * 4 + 3] + dmin) * id).round().clamp(0.0, 3.0) as u8;

            dst[qs_start + i] = (q0 & 3) | ((q1 & 3) << 2) | ((q2 & 3) << 4) | ((q3 & 3) << 6);
        }
        dst_ptr += 64;

        let d_bytes = f16::from_f32(d).to_le_bytes();
        let dmin_bytes = f16::from_f32(dmin).to_le_bytes();

        dst[dst_ptr..dst_ptr + 2].copy_from_slice(&d_bytes);
        dst_ptr += 2;
        dst[dst_ptr..dst_ptr + 2].copy_from_slice(&dmin_bytes);
        dst_ptr += 2;
    }

    Ok(required)
}

pub fn quantize_f32_to_q3_k(src: &[f32], dst: &mut [u8]) -> Result<usize, BridgeError> {
    if !src.len().is_multiple_of(256) {
        return Err(BridgeError::QuantizeError(format!(
            "element count {} is not a multiple of 256 for Q3_K",
            src.len()
        )));
    }

    let nb = src.len() / 256;
    let required = nb.saturating_mul(110);
    if dst.len() < required {
        return Err(BridgeError::BufferSizeMismatch {
            expected: required,
            found: dst.len(),
        });
    }

    let mut dst_ptr = 0usize;

    for chunk in src.as_chunks::<256>().0 {
        let mut amax = 0.0f32;
        for &x in chunk {
            let ax = x.abs();
            if ax > amax {
                amax = ax;
            }
        }

        let d = amax / 3.5f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };

        let hmask_start = dst_ptr;
        dst[hmask_start..hmask_start + 32].fill(0);
        dst_ptr += 32;

        let qs_start = dst_ptr;
        dst[qs_start..qs_start + 64].fill(0);

        for i in 0..64 {
            let q0 = ((chunk[i * 4] * id).round() + 4.0).clamp(0.0, 7.0) as u8;
            let q1 = ((chunk[i * 4 + 1] * id).round() + 4.0).clamp(0.0, 7.0) as u8;
            let q2 = ((chunk[i * 4 + 2] * id).round() + 4.0).clamp(0.0, 7.0) as u8;
            let q3 = ((chunk[i * 4 + 3] * id).round() + 4.0).clamp(0.0, 7.0) as u8;

            dst[qs_start + i] = (q0 & 3) | ((q1 & 3) << 2) | ((q2 & 3) << 4) | ((q3 & 3) << 6);

            let h0 = (q0 >> 2) & 1;
            let h1 = (q1 >> 2) & 1;
            let h2 = (q2 >> 2) & 1;
            let h3 = (q3 >> 2) & 1;

            dst[hmask_start + (i / 2)] |= (h0 << ((i % 2) * 4))
                | (h1 << ((i % 2) * 4 + 1))
                | (h2 << ((i % 2) * 4 + 2))
                | (h3 << ((i % 2) * 4 + 3));
        }
        dst_ptr += 64;

        dst[dst_ptr..dst_ptr + 12].fill(0);
        dst_ptr += 12;

        let d_bytes = f16::from_f32(d).to_le_bytes();
        dst[dst_ptr..dst_ptr + 2].copy_from_slice(&d_bytes);
        dst_ptr += 2;
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

    if dtype_upper == "F16" && target_type == GgmlType::F16 {
        dst.clear();
        dst.extend_from_slice(raw_bytes);
        return Ok(());
    }

    let elem_count = match dtype_upper.as_str() {
        "F16" | "BF16" => raw_bytes.len() / 2,
        "F32" => raw_bytes.len() / 4,
        other => return Err(BridgeError::UnsupportedSourceDtype(other.to_string())),
    };

    let mut f32_buf = vec![0.0f32; elem_count];
    match dtype_upper.as_str() {
        "F16" => {
            for i in 0..elem_count {
                f32_buf[i] = f16::from_le_bytes([raw_bytes[i * 2], raw_bytes[i * 2 + 1]]).to_f32();
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

    match target_type {
        GgmlType::F16 => quantize_f32_to_f16(&f32_buf, dst)?,
        GgmlType::Q8_0 => quantize_f32_to_q8_0(&f32_buf, dst)?,
        GgmlType::Q4_0 => quantize_f32_to_q4_0(&f32_buf, dst)?,
        GgmlType::Q4_K => quantize_f32_to_q4_k(&f32_buf, dst)?,
        GgmlType::Q5_K => quantize_f32_to_q5_k(&f32_buf, dst)?,
        GgmlType::Q6_K => quantize_f32_to_q6_k(&f32_buf, dst)?,
        GgmlType::Q2_K => quantize_f32_to_q2_k(&f32_buf, dst)?,
        GgmlType::Q3_K => quantize_f32_to_q3_k(&f32_buf, dst)?,
        other => {
            return Err(BridgeError::QuantizeError(format!(
                "quantization to {} not implemented",
                other.as_str()
            )));
        }
    };

    Ok(())
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

    #[test]
    fn test_quantize_f32_to_k_quants() {
        let src: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();

        let mut dst_q4k = vec![0u8; 144];
        assert_eq!(quantize_f32_to_q4_k(&src, &mut dst_q4k).unwrap(), 144);

        let mut dst_q5k = vec![0u8; 176];
        assert_eq!(quantize_f32_to_q5_k(&src, &mut dst_q5k).unwrap(), 176);

        let mut dst_q6k = vec![0u8; 210];
        assert_eq!(quantize_f32_to_q6_k(&src, &mut dst_q6k).unwrap(), 210);

        let mut dst_q2k = vec![0u8; 84];
        assert_eq!(quantize_f32_to_q2_k(&src, &mut dst_q2k).unwrap(), 84);

        let mut dst_q3k = vec![0u8; 110];
        assert_eq!(quantize_f32_to_q3_k(&src, &mut dst_q3k).unwrap(), 110);
    }
}
