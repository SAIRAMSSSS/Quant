use crate::error::BridgeError;
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

#[derive(Debug, Clone, Default)]
pub struct LlamaQuantizeRunner {
    binary_path: Option<PathBuf>,
}

impl LlamaQuantizeRunner {
    pub fn new() -> Self {
        Self { binary_path: None }
    }

    pub fn with_binary_path<P: AsRef<Path>>(path: P) -> Self {
        Self {
            binary_path: Some(path.as_ref().to_path_buf()),
        }
    }

    pub fn resolve_binary(&self) -> Result<PathBuf, BridgeError> {
        if let Some(ref path) = self.binary_path {
            if path.exists() {
                return Ok(path.clone());
            }
        }

        if let Ok(env_path) = env::var("LLAMA_QUANTIZE_PATH") {
            let p = PathBuf::from(env_path);
            if p.exists() {
                return Ok(p);
            }
        }

        if let Ok(paths) = env::var("PATH") {
            for dir in env::split_paths(&paths) {
                let candidate = dir.join("llama-quantize");
                if candidate.exists() {
                    return Ok(candidate);
                }
                #[cfg(target_os = "windows")]
                {
                    let win_candidate = dir.join("llama-quantize.exe");
                    if win_candidate.exists() {
                        return Ok(win_candidate);
                    }
                }
            }
        }

        Err(BridgeError::LlamaQuantizeNotFound)
    }

    pub fn execute<P1: AsRef<Path>, P2: AsRef<Path>>(
        &self,
        input_gguf: P1,
        output_gguf: P2,
        quant_type: &str,
        extra_args: &[&str],
    ) -> Result<ExitStatus, BridgeError> {
        let binary = self.resolve_binary()?;
        let in_path = input_gguf.as_ref();
        let out_path = output_gguf.as_ref();

        if !in_path.exists() {
            return Err(BridgeError::QuantizeError(format!(
                "input gguf file does not exist: {}",
                in_path.display()
            )));
        }

        let mut cmd = Command::new(&binary);
        for &arg in extra_args {
            cmd.arg(arg);
        }
        cmd.arg(in_path);
        cmd.arg(out_path);
        cmd.arg(quant_type);

        let output = cmd.output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(BridgeError::LlamaQuantizeExecutionFailed {
                code: output.status.code(),
                stderr,
            });
        }

        Ok(output.status)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_binary_not_found() {
        let runner = LlamaQuantizeRunner::with_binary_path("/non/existent/path/llama-quantize");
        assert!(matches!(
            runner.resolve_binary(),
            Err(BridgeError::LlamaQuantizeNotFound)
        ));
    }
}
