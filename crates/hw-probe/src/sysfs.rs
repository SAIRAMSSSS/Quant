use crate::types::{GpuBackend, GpuDevice};
use std::fs;
use std::path::Path;

pub fn probe_sysfs_amdgpu() -> Vec<GpuDevice> {
    let drm_path = Path::new("/sys/class/drm");
    if !drm_path.exists() {
        return Vec::new();
    }

    let entries = match fs::read_dir(drm_path) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut devices = Vec::new();

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }

        let dev_dir = entry.path().join("device");
        let vram_total_file = dev_dir.join("mem_info_vram_total");
        let vram_used_file = dev_dir.join("mem_info_vram_used");

        if vram_total_file.exists() {
            let total_str = fs::read_to_string(&vram_total_file).unwrap_or_default();
            let used_str = fs::read_to_string(&vram_used_file).unwrap_or_default();

            let total_bytes: u64 = total_str.trim().parse().unwrap_or(0);
            let used_bytes: u64 = used_str.trim().parse().unwrap_or(0);

            if total_bytes > 0 {
                let free_bytes = total_bytes.saturating_sub(used_bytes);
                let dev_name = fs::read_to_string(dev_dir.join("product_name"))
                    .or_else(|_| fs::read_to_string(dev_dir.join("device_name")))
                    .unwrap_or_else(|_| format!("AMDGPU {name}"));

                devices.push(GpuDevice {
                    name: dev_name.trim().to_string(),
                    backend: GpuBackend::Rocm,
                    total_vram_bytes: total_bytes,
                    free_vram_bytes: free_bytes,
                });
            }
        }
    }

    devices
}
