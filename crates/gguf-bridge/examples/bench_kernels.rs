use quant_core::types::GgmlType;
use quant_gguf_bridge::error::BridgeError;
use quant_gguf_bridge::quantize::*;
use std::time::Instant;

type QuantFn = fn(&[f32], &mut [u8]) -> Result<usize, BridgeError>;

struct BenchmarkCase {
    name: &'static str,
    ggml_type: GgmlType,
    quant_fn: QuantFn,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("GGML QUANTIZATION KERNEL BENCHMARK (AVX2 / SIMD & SCALAR)");
    println!("================================================================================");

    #[cfg(target_arch = "x86_64")]
    {
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_fma = is_x86_feature_detected!("fma");
        println!(
            "CPU Features Detected: AVX2 = {}, FMA = {}",
            has_avx2, has_fma
        );
    }

    let n_elements = 16 * 1024 * 1024;
    let input_bytes = n_elements * 4;
    let input_gib = (input_bytes as f64) / (1024.0 * 1024.0 * 1024.0);

    println!(
        "Benchmark Payload:     {} elements ({:.2} MiB input buffer)\n",
        n_elements,
        (input_bytes as f64) / (1024.0 * 1024.0)
    );

    let mut src = vec![0.0f32; n_elements];
    for (i, x) in src.iter_mut().enumerate() {
        *x = (((i % 1000) as f32) - 500.0) * 0.0031415;
    }

    let iterations = 10;

    let benchmarks = vec![
        BenchmarkCase {
            name: "F16 (Half-Float)",
            ggml_type: GgmlType::F16,
            quant_fn: quantize_f32_to_f16,
        },
        BenchmarkCase {
            name: "Q8_0 (8-bit Linear)",
            ggml_type: GgmlType::Q8_0,
            quant_fn: quantize_f32_to_q8_0,
        },
        BenchmarkCase {
            name: "Q4_0 (4-bit Linear)",
            ggml_type: GgmlType::Q4_0,
            quant_fn: quantize_f32_to_q4_0,
        },
        BenchmarkCase {
            name: "Q4_K (4-bit Superblock)",
            ggml_type: GgmlType::Q4_K,
            quant_fn: quantize_f32_to_q4_k,
        },
        BenchmarkCase {
            name: "Q5_K (5-bit Superblock)",
            ggml_type: GgmlType::Q5_K,
            quant_fn: quantize_f32_to_q5_k,
        },
        BenchmarkCase {
            name: "Q6_K (6-bit Superblock)",
            ggml_type: GgmlType::Q6_K,
            quant_fn: quantize_f32_to_q6_k,
        },
        BenchmarkCase {
            name: "Q2_K (2-bit Superblock)",
            ggml_type: GgmlType::Q2_K,
            quant_fn: quantize_f32_to_q2_k,
        },
        BenchmarkCase {
            name: "Q3_K (3-bit Superblock)",
            ggml_type: GgmlType::Q3_K,
            quant_fn: quantize_f32_to_q3_k,
        },
    ];

    println!(
        "+-------------------------+----------+------------+------------+---------------+----------------+"
    );
    println!(
        "| {:<23} | {:<8} | {:>10} | {:>10} | {:>13} | {:>14} |",
        "Kernel / Target Type",
        "Ratio",
        "Time (ms)",
        "Throughput",
        "Rate (Melem/s)",
        "Output (MiB)"
    );
    println!(
        "+-------------------------+----------+------------+------------+---------------+----------------+"
    );

    for case in benchmarks {
        let expected_out_bytes = case.ggml_type.bytes_for_elements(n_elements)?;
        let mut dst = vec![0u8; expected_out_bytes];

        let _ = (case.quant_fn)(&src[..4096], &mut dst[..2048]);

        let start = Instant::now();
        for _ in 0..iterations {
            let written = (case.quant_fn)(&src, &mut dst)?;
            if written != expected_out_bytes {
                eprintln!("Size mismatch in {}", case.name);
            }
        }
        let elapsed = start.elapsed();
        let avg_time_sec = elapsed.as_secs_f64() / (iterations as f64);
        let avg_time_ms = avg_time_sec * 1000.0;

        let throughput_gib = input_gib / avg_time_sec;
        let melem_per_sec = ((n_elements as f64) / 1_000_000.0) / avg_time_sec;
        let out_mib = (expected_out_bytes as f64) / (1024.0 * 1024.0);
        let compression_ratio = (input_bytes as f64) / (expected_out_bytes as f64);

        println!(
            "| {:<23} | {:>7.2}x | {:>10.2} | {:>7.2} GiB/s | {:>10.2} M/s | {:>11.2} MiB |",
            case.name, compression_ratio, avg_time_ms, throughput_gib, melem_per_sec, out_mib
        );
    }

    println!(
        "+-------------------------+----------+------------+------------+---------------+----------------+"
    );

    Ok(())
}
