use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuBackend {
    Cuda,
    Rocm,
    Metal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuDevice {
    pub name: String,
    pub backend: GpuBackend,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemRam {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub free_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub gpus: Vec<GpuDevice>,
    pub system_ram: SystemRam,
}

impl HardwareProfile {
    pub fn primary_vram_bytes(&self) -> Option<u64> {
        self.gpus.iter().map(|g| g.total_vram_bytes).max()
    }

    pub fn primary_free_vram_bytes(&self) -> Option<u64> {
        self.gpus.iter().map(|g| g.free_vram_bytes).max()
    }

    pub fn recommend_vram_target(&self, headroom_ratio: f64) -> u64 {
        let valid_ratio = headroom_ratio.clamp(0.05, 0.50);
        if let Some(vram) = self.primary_vram_bytes() {
            let usable = (vram as f64) * (1.0 - valid_ratio);
            usable as u64
        } else {
            let usable = (self.system_ram.available_bytes as f64) * 0.70;
            usable as u64
        }
    }
}
