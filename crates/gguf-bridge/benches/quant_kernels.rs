#![allow(clippy::expect_used, clippy::unwrap_used)]

use quant_core::types::GgmlType;
use quant_gguf_bridge::quantize_f32_slice;
use std::time::Instant;

fn run_kernel_bench(name: &str, target_type: GgmlType, f32_data: &[f32], iterations: usize) {
    let mut out_buf = Vec::new();

    // Warmup
    quantize_f32_slice(f32_data, target_type, &mut out_buf).expect("warmup quantization failed");

    let start = Instant::now();
    for _ in 0..iterations {
        quantize_f32_slice(f32_data, target_type, &mut out_buf).expect("bench quantization failed");
    }
    let elapsed = start.elapsed();

    let total_bytes_processed = (f32_data.len() * 4 * iterations) as f64;
    let gib_processed = total_bytes_processed / (1024.0 * 1024.0 * 1024.0);
    let gib_per_sec = gib_processed / elapsed.as_secs_f64();
    let mib_per_sec = gib_per_sec * 1024.0;
    let avg_ms = (elapsed.as_secs_f64() * 1000.0) / (iterations as f64);

    println!(
        "| {:<16} | {:>8.2} BPW | {:>10.3} ms | {:>10.2} MiB/s | {:>8.3} GiB/s |",
        name,
        target_type.effective_bpw(),
        avg_ms,
        mib_per_sec,
        gib_per_sec
    );
}

fn main() {
    println!("================================================================================");
    println!("QUANTIZATION SIMD & KERNEL MICRO-BENCHMARK SUITE");
    println!("================================================================================");

    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = is_x86_feature_detected!("avx2");
        let fma = is_x86_feature_detected!("fma");
        println!(
            "Hardware Acceleration: x86_64 (AVX2: {}, FMA: {})",
            avx2, fma
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("Hardware Acceleration: Scalar Portable Fallback");
    }

    const ELEMS: usize = 1_048_576; // 1M elements = 4 MiB FP32 tensor
    const ITERS: usize = 50;

    println!(
        "Dataset: 1,048,576 FP32 elements (4.00 MiB) per iteration, {} iterations",
        ITERS
    );
    println!("+------------------+--------------+-------------+---------------+------------+");
    println!("| Kernel           | Bitwidth     | Avg Latency | Throughput    | Bandwidth  |");
    println!("+------------------+--------------+-------------+---------------+------------+");

    let f32_data: Vec<f32> = (0..ELEMS)
        .map(|i| ((i as f32) * 0.05).sin() * 1.5)
        .collect();

    run_kernel_bench("F32 -> F16", GgmlType::F16, &f32_data, ITERS);
    run_kernel_bench("F32 -> Q8_0", GgmlType::Q8_0, &f32_data, ITERS);
    run_kernel_bench("F32 -> Q4_0", GgmlType::Q4_0, &f32_data, ITERS);
    run_kernel_bench("F32 -> Q4_K", GgmlType::Q4_K, &f32_data, ITERS);
    run_kernel_bench("F32 -> Q5_K", GgmlType::Q5_K, &f32_data, ITERS);
    run_kernel_bench("F32 -> Q6_K", GgmlType::Q6_K, &f32_data, ITERS);

    println!("+------------------+--------------+-------------+---------------+------------+");
}
