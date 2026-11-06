use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("invalid ggml type identifier: {0}")]
    InvalidGgmlType(u32),

    #[error("alignment violation: expected {expected} bytes alignment, found offset {found}")]
    AlignmentError { expected: usize, found: usize },

    #[error("element count {element_count} is not a multiple of block size {block_size}")]
    BlockSizeMismatch {
        element_count: usize,
        block_size: usize,
    },

    #[error("safetensors buffer too small: expected at least {0} bytes")]
    SafetensorsHeaderTooSmall(usize),

    #[error("safetensors header length {0} exceeds buffer or memory limit")]
    SafetensorsInvalidLength(u64),

    #[error("safetensors json parse error: {0}")]
    SafetensorsJson(#[from] serde_json::Error),

    #[error("safetensors header corrupt: {detail}")]
    SafetensorsCorrupt { detail: String },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("http request error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("http range request rejected with status code: {0}")]
    HttpRangeRejected(u16),

    #[error("mckp budget infeasible: available budget {budget_bytes} bytes is below minimal required {minimum_bytes} bytes")]
    KnapsackInfeasible {
        budget_bytes: usize,
        minimum_bytes: usize,
    },

    #[error("empty tensor group or missing quantization candidates for: {0}")]
    MissingCandidates(String),

    #[error("safetensors index error: {0}")]
    SafetensorsIndexError(String),

    #[error("safetensors shard file not found: {0}")]
    SafetensorsShardNotFound(String),
}
