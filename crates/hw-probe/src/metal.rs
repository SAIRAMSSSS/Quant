#[cfg(target_os = "macos")]
use crate::types::GpuBackend;
use crate::types::GpuDevice;

#[cfg(target_os = "macos")]
use std::process::Command;

pub fn probe_metal_gpus() -> Vec<GpuDevice> {
    #[cfg(target_os = "macos")]
    {
        probe_macos_metal()
    }

    #[cfg(not(target_os = "macos"))]
    {
        Vec::new()
    }
}

#[cfg(target_os = "macos")]
fn probe_macos_metal() -> Vec<GpuDevice> {
    let mut devices = Vec::new();

    // Query unified memory size via sysctl hw.memsize
    let output = match Command::new("sysctl").arg("-n").arg("hw.memsize").output() {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => return devices,
    };

    let total_bytes: u64 = match output.parse() {
        Ok(v) => v,
        Err(_) => return devices,
    };

    // Query chip model name via sysctl machdep.cpu.brand_string
    let brand = match Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.brand_string")
        .output()
    {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "Apple Silicon (Metal Unified Memory)".to_string(),
    };

    // On Apple Silicon, unified memory is shared between CPU and GPU;
    // default limit allocatable to Metal is typically up to 75% of total memory.
    let metal_vram = ((total_bytes as f64) * 0.75) as u64;

    devices.push(GpuDevice {
        name: brand,
        backend: GpuBackend::Metal,
        total_vram_bytes: metal_vram,
        free_vram_bytes: metal_vram,
    });

    devices
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_metal_does_not_panic() {
        let devs = probe_metal_gpus();
        #[cfg(not(target_os = "macos"))]
        assert!(devs.is_empty());
        #[cfg(target_os = "macos")]
        assert!(!devs.is_empty());
    }
}
