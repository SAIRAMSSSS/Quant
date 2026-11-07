#![allow(clippy::expect_used, clippy::unwrap_used)]

use quant_core::mckp::MckpSolver;
use std::collections::HashMap;
use std::time::Instant;

fn bench_solver(layer_count: usize, iterations: usize) {
    let mut tensors = HashMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![32000, 4096], 32000 * 4096),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        (vec![32000, 4096], 32000 * 4096),
    );
    tensors.insert("model.norm.weight".to_string(), (vec![4096], 4096));

    for i in 0..layer_count {
        let p = format!("model.layers.{i}");
        tensors.insert(format!("{p}.input_layernorm.weight"), (vec![4096], 4096));
        tensors.insert(
            format!("{p}.self_attn.q_proj.weight"),
            (vec![4096, 4096], 4096 * 4096),
        );
        tensors.insert(
            format!("{p}.self_attn.k_proj.weight"),
            (vec![1024, 4096], 1024 * 4096),
        );
        tensors.insert(
            format!("{p}.self_attn.v_proj.weight"),
            (vec![1024, 4096], 1024 * 4096),
        );
        tensors.insert(
            format!("{p}.self_attn.o_proj.weight"),
            (vec![4096, 4096], 4096 * 4096),
        );
        tensors.insert(
            format!("{p}.post_attention_layernorm.weight"),
            (vec![4096], 4096),
        );
        tensors.insert(
            format!("{p}.mlp.gate_proj.weight"),
            (vec![14336, 4096], 14336 * 4096),
        );
        tensors.insert(
            format!("{p}.mlp.up_proj.weight"),
            (vec![14336, 4096], 14336 * 4096),
        );
        tensors.insert(
            format!("{p}.mlp.down_proj.weight"),
            (vec![4096, 14336], 4096 * 14336),
        );
    }

    let groups = MckpSolver::build_tensor_groups(&tensors).expect("build groups failed");
    let target_budget = 16 * 1024 * 1024 * 1024; // 16 GiB target
    let kv_cache = 512 * 1024 * 1024; // 512 MiB
    let overhead = 256 * 1024 * 1024; // 256 MiB

    // Warmup
    let _ =
        MckpSolver::solve(&groups, target_budget, kv_cache, overhead).expect("warmup solve failed");

    let start = Instant::now();
    for _ in 0..iterations {
        let _ =
            MckpSolver::solve(&groups, target_budget, kv_cache, overhead).expect("solve failed");
    }
    let elapsed = start.elapsed();
    let avg_us = (elapsed.as_secs_f64() * 1_000_000.0) / (iterations as f64);
    let total_tensors = tensors.len();

    println!(
        "| {:>12} | {:>14} | {:>12.2} µs | {:>10.3} ms |",
        layer_count,
        total_tensors,
        avg_us,
        avg_us / 1000.0
    );
}

fn main() {
    println!("================================================================================");
    println!("MCKP KNAPSACK OPTIMIZATION SOLVER BENCHMARK");
    println!("================================================================================");
    println!("Target Budget: 4.00 GiB VRAM, Iterations: 100 per test");
    println!("+--------------+----------------+--------------+------------+");
    println!("| Layer Count  | Total Tensors  | Avg Latency  | Per Solve  |");
    println!("+--------------+----------------+--------------+------------+");

    bench_solver(16, 100);
    bench_solver(32, 100);
    bench_solver(64, 100);
    bench_solver(128, 100);

    println!("+--------------+----------------+--------------+------------+");
}
