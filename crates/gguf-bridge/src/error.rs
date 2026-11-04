use thiserror::Error;

#[derive(Error, Debug)]
pub enum BridgeError {
    #[error("core error: {0}")]
    Core(#[from] quant_core::CoreError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid gguf magic bytes: {0:?}, expected b\"GGUF\"")]
    InvalidMagic([u8; 4]),

    #[error("unsupported gguf version {0}, supported version is 3")]
    UnsupportedVersion(u32),

    #[error("invalid metadata value type tag: {0}")]
    InvalidMetadataType(u32),

    #[error("alignment violation: {0}")]
    AlignmentError(String),

    #[error("quantization failed: {0}")]
    QuantizeError(String),

    #[error("unsupported source dtype '{0}' for conversion")]
    UnsupportedSourceDtype(String),

    #[error("buffer size mismatch: expected at least {expected} bytes, found {found}")]
    BufferSizeMismatch { expected: usize, found: usize },
}
