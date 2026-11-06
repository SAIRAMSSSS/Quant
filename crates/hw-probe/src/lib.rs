pub mod error;
pub mod metal;
pub mod nvml;
pub mod ram;
pub mod sysfs;
pub mod types;

pub use error::HwProbeError;
pub use types::{GpuBackend, GpuDevice, HardwareProfile, SystemRam};

pub fn probe_hardware() -> HardwareProfile {
    let mut gpus = nvml::probe_nvml_gpus();
    let amd_gpus = sysfs::probe_sysfs_amdgpu();
    gpus.extend(amd_gpus);
    let metal_gpus = metal::probe_metal_gpus();
    gpus.extend(metal_gpus);

    let system_ram = ram::probe_system_ram();

    HardwareProfile { gpus, system_ram }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_hardware_execution() {
        let profile = probe_hardware();
        assert!(profile.system_ram.total_bytes > 0);
        let rec = profile.recommend_vram_target(0.10);
        assert!(rec > 0);
    }
}
