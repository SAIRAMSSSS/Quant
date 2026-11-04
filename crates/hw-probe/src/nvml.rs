use crate::error::HwProbeError;
use crate::types::{GpuBackend, GpuDevice};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::process::Command;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

type NvmlInitV2Fn = unsafe extern "C" fn() -> i32;
type NvmlShutdownFn = unsafe extern "C" fn() -> i32;
type NvmlDeviceGetCountV2Fn = unsafe extern "C" fn(*mut u32) -> i32;
type NvmlDeviceGetHandleByIndexV2Fn = unsafe extern "C" fn(u32, *mut usize) -> i32;
type NvmlDeviceGetNameFn = unsafe extern "C" fn(usize, *mut c_char, u32) -> i32;
type NvmlDeviceGetMemoryInfoFn = unsafe extern "C" fn(usize, *mut NvmlMemory) -> i32;

pub fn probe_nvml_gpus() -> Vec<GpuDevice> {
    if let Ok(devices) = probe_nvml_dynamic() {
        if !devices.is_empty() {
            return devices;
        }
    }

    probe_nvidia_smi_cli()
}

fn probe_nvml_dynamic() -> Result<Vec<GpuDevice>, HwProbeError> {
    let lib_names = ["libnvidia-ml.so.1", "libnvidia-ml.so"];
    let mut lib_opt = None;

    for name in &lib_names {
        if let Ok(l) = unsafe { libloading::Library::new(name) } {
            lib_opt = Some(l);
            break;
        }
    }

    let lib = match lib_opt {
        Some(l) => l,
        None => {
            return Err(HwProbeError::NvmlLibraryLoad(
                "libnvidia-ml.so.1 not found".to_string(),
            ));
        }
    };

    unsafe {
        let nvml_init: libloading::Symbol<NvmlInitV2Fn> = lib
            .get(b"nvmlInit_v2\0")
            .map_err(|e| HwProbeError::NvmlLibraryLoad(e.to_string()))?;
        let nvml_shutdown: libloading::Symbol<NvmlShutdownFn> = lib
            .get(b"nvmlShutdown\0")
            .map_err(|e| HwProbeError::NvmlLibraryLoad(e.to_string()))?;
        let nvml_get_count: libloading::Symbol<NvmlDeviceGetCountV2Fn> = lib
            .get(b"nvmlDeviceGetCount_v2\0")
            .map_err(|e| HwProbeError::NvmlLibraryLoad(e.to_string()))?;
        let nvml_get_handle: libloading::Symbol<NvmlDeviceGetHandleByIndexV2Fn> = lib
            .get(b"nvmlDeviceGetHandleByIndex_v2\0")
            .map_err(|e| HwProbeError::NvmlLibraryLoad(e.to_string()))?;
        let nvml_get_name: libloading::Symbol<NvmlDeviceGetNameFn> = lib
            .get(b"nvmlDeviceGetName\0")
            .map_err(|e| HwProbeError::NvmlLibraryLoad(e.to_string()))?;
        let nvml_get_memory: libloading::Symbol<NvmlDeviceGetMemoryInfoFn> = lib
            .get(b"nvmlDeviceGetMemoryInfo\0")
            .map_err(|e| HwProbeError::NvmlLibraryLoad(e.to_string()))?;

        let init_ret = nvml_init();
        if init_ret != 0 {
            return Err(HwProbeError::NvmlCall {
                function: "nvmlInit_v2",
                code: init_ret,
            });
        }

        let mut device_count: u32 = 0;
        let count_ret = nvml_get_count(&mut device_count);
        if count_ret != 0 {
            let _ = nvml_shutdown();
            return Err(HwProbeError::NvmlCall {
                function: "nvmlDeviceGetCount_v2",
                code: count_ret,
            });
        }

        let mut devices = Vec::with_capacity(device_count as usize);

        for idx in 0..device_count {
            let mut handle: usize = 0;
            if nvml_get_handle(idx, &mut handle) != 0 {
                continue;
            }

            let mut name_buf = [0i8; 96];
            let name_str = if nvml_get_name(handle, name_buf.as_mut_ptr(), 96) == 0 {
                CStr::from_ptr(name_buf.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            } else {
                format!("NVIDIA GPU {idx}")
            };

            let mut mem = NvmlMemory::default();
            if nvml_get_memory(handle, &mut mem) == 0 {
                devices.push(GpuDevice {
                    name: name_str,
                    backend: GpuBackend::Cuda,
                    total_vram_bytes: mem.total,
                    free_vram_bytes: mem.free,
                });
            }
        }

        let _ = nvml_shutdown();
        Ok(devices)
    }
}

fn probe_nvidia_smi_cli() -> Vec<GpuDevice> {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(out) => out,
        Err(_) => return Vec::new(),
    };

    if !output.status.success() {
        return Vec::new();
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut devices = Vec::new();

    for line in text.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 3 {
            let name = parts[0].to_string();
            let total_mb: u64 = parts[1].parse().unwrap_or(0);
            let free_mb: u64 = parts[2].parse().unwrap_or(0);
            devices.push(GpuDevice {
                name,
                backend: GpuBackend::Cuda,
                total_vram_bytes: total_mb.saturating_mul(1024 * 1024),
                free_vram_bytes: free_mb.saturating_mul(1024 * 1024),
            });
        }
    }

    devices
}
