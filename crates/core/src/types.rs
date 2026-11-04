use crate::error::CoreError;
use serde::{Deserialize, Serialize};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
}

impl GgmlType {
    pub const fn block_size(&self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
        }
    }

    pub const fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 290,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    pub fn bytes_for_elements(&self, n: usize) -> Result<usize, CoreError> {
        let bsize = self.block_size();
        if !n.is_multiple_of(bsize) {
            return Err(CoreError::BlockSizeMismatch {
                element_count: n,
                block_size: bsize,
            });
        }
        let blocks = n / bsize;
        Ok(blocks.saturating_mul(self.type_size()))
    }

    pub fn effective_bpw(&self) -> f64 {
        (self.type_size() * 8) as f64 / self.block_size() as f64
    }

    pub fn base_distortion(&self) -> f64 {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 0.0,
            Self::Q8_K | Self::Q8_1 | Self::Q8_0 => 0.0004,
            Self::Q6_K => 0.0012,
            Self::Q5_K => 0.0028,
            Self::Q5_1 | Self::Q5_0 => 0.0035,
            Self::Q4_K => 0.0065,
            Self::Q4_1 | Self::Q4_0 => 0.0085,
            Self::Q3_K => 0.0195,
            Self::Q2_K => 0.0620,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::F64 => 0.0,
        }
    }

    pub const fn from_u32(val: u32) -> Result<Self, CoreError> {
        match val {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            30 => Ok(Self::BF16),
            other => Err(CoreError::InvalidGgmlType(other)),
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::BF16 => "BF16",
        }
    }

    pub fn from_str_case_insensitive(s: &str) -> Option<Self> {
        let upper = s.to_ascii_uppercase();
        match upper.as_str() {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "Q4_0" => Some(Self::Q4_0),
            "Q4_1" => Some(Self::Q4_1),
            "Q5_0" => Some(Self::Q5_0),
            "Q5_1" => Some(Self::Q5_1),
            "Q8_0" => Some(Self::Q8_0),
            "Q8_1" => Some(Self::Q8_1),
            "Q2_K" => Some(Self::Q2_K),
            "Q3_K" => Some(Self::Q3_K),
            "Q4_K" | "Q4_K_M" | "Q4_K_S" => Some(Self::Q4_K),
            "Q5_K" | "Q5_K_M" | "Q5_K_S" => Some(Self::Q5_K),
            "Q6_K" => Some(Self::Q6_K),
            "Q8_K" => Some(Self::Q8_K),
            "I8" => Some(Self::I8),
            "I16" => Some(Self::I16),
            "I32" => Some(Self::I32),
            "I64" => Some(Self::I64),
            "F64" => Some(Self::F64),
            "BF16" => Some(Self::BF16),
            _ => None,
        }
    }
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    pub d: half::f16,
    pub qs: [i8; 32],
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub d: half::f16,
    pub qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_K {
    pub d: half::f16,
    pub dmin: half::f16,
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_block_sizes_and_type_sizes() {
        assert_eq!(GgmlType::Q4_0.block_size(), 32);
        assert_eq!(GgmlType::Q4_0.type_size(), 18);
        assert_eq!(GgmlType::Q8_0.block_size(), 32);
        assert_eq!(GgmlType::Q8_0.type_size(), 34);

        assert_eq!(GgmlType::Q2_K.block_size(), 256);
        assert_eq!(GgmlType::Q2_K.type_size(), 84);
        assert_eq!(GgmlType::Q3_K.block_size(), 256);
        assert_eq!(GgmlType::Q3_K.type_size(), 110);
        assert_eq!(GgmlType::Q4_K.block_size(), 256);
        assert_eq!(GgmlType::Q4_K.type_size(), 144);
        assert_eq!(GgmlType::Q5_K.block_size(), 256);
        assert_eq!(GgmlType::Q5_K.type_size(), 176);
        assert_eq!(GgmlType::Q6_K.block_size(), 256);
        assert_eq!(GgmlType::Q6_K.type_size(), 210);
        assert_eq!(GgmlType::Q8_K.block_size(), 256);
        assert_eq!(GgmlType::Q8_K.type_size(), 290);
    }

    #[test]
    fn test_bytes_for_elements() {
        let sz = GgmlType::Q4_0.bytes_for_elements(320).unwrap();
        assert_eq!(sz, 180);

        let err = GgmlType::Q4_0.bytes_for_elements(33);
        assert!(err.is_err());
    }

    #[test]
    fn test_effective_bpw() {
        assert!((GgmlType::F16.effective_bpw() - 16.0).abs() < 1e-6);
        assert!((GgmlType::Q4_0.effective_bpw() - 4.5).abs() < 1e-6);
        assert!((GgmlType::Q8_0.effective_bpw() - 8.5).abs() < 1e-6);
        assert!((GgmlType::Q4_K.effective_bpw() - 4.5).abs() < 1e-6);
        assert!((GgmlType::Q2_K.effective_bpw() - 2.625).abs() < 1e-6);
        assert!((GgmlType::Q6_K.effective_bpw() - 6.5625).abs() < 1e-6);
    }

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ4_K>(), 144);
    }
}
