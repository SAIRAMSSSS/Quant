use thiserror::Error;

#[derive(Error, Debug)]
pub enum HwProbeError {
    #[error("failed to load nvml shared library: {0}")]
    NvmlLibraryLoad(String),

    #[error("nvml function '{function}' returned error code {code}")]
    NvmlCall { function: &'static str, code: i32 },

    #[error("io error while querying hardware interface: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to parse hardware metric: {0}")]
    Parse(String),

    #[error("unsupported platform for hardware probe backend")]
    UnsupportedPlatform,
}
