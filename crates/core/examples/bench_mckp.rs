use quant_core::mckp::MckpSolver;
use std::collections::HashMap;
use std::time::Instant;

fn generate_synthetic_tensor_map(
    n_layers: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
) -> HashMap<String, (Vec<usize>, usize)> {
    let mut map = HashMap::new();

    let vocab_size = 32000;
    map.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![vocab_size, hidden_dim], vocab_size * hidden_dim),
    );
    map.insert(
        "lm_head.weight".to_string(),
        (vec![vocab_size, hidden_dim], vocab_size * hidden_dim),
    );
    map.insert(
        "model.norm.weight".to_string(),
        (vec![hidden_dim], hidden_dim),
    );

    for l in 0..n_layers {
        let p = format!("model.layers.{}", l);
        map.insert(
            format!("{}.input_layernorm.weight", p),
            (vec![hidden_dim], hidden_dim),
        );
        map.insert(
            format!("{}.post_attention_layernorm.weight", p),
            (vec![hidden_dim], hidden_dim),
        );

        map.insert(
            format!("{}.self_attn.q_proj.weight", p),
            (vec![hidden_dim, hidden_dim], hidden_dim * hidden_dim),
        );
        map.insert(
            format!("{}.self_attn.k_proj.weight", p),
            (vec![hidden_dim, hidden_dim], hidden_dim * hidden_dim),
        );
        map.insert(
            format!("{}.self_attn.v_proj.weight", p),
            (vec![hidden_dim, hidden_dim], hidden_dim * hidden_dim),
        );
        map.insert(
            format!("{}.self_attn.o_proj.weight", p),
            (vec![hidden_dim, hidden_dim], hidden_dim * hidden_dim),
        );

        map.insert(
            format!("{}.mlp.gate_proj.weight", p),
            (
                vec![intermediate_dim, hidden_dim],
                intermediate_dim * hidden_dim,
            ),
        );
        map.insert(
            format!("{}.mlp.up_proj.weight", p),
            (
                vec![intermediate_dim, hidden_dim],
                intermediate_dim * hidden_dim,
            ),
        );
        map.insert(
            format!("{}.mlp.down_proj.weight", p),
            (
                vec![hidden_dim, intermediate_dim],
                hidden_dim * intermediate_dim,
            ),
        );
    }

    map
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("MCKP DISCRETE DYNAMIC PROGRAMMING SOLVER BENCHMARK");
    println!("================================================================================");

    let scenarios = vec![
        (
            "135M Model (SmolLM Scale)",
            30,
            576,
            1536,
            4 * 1024 * 1024 * 1024,
        ),
        (
            "7B/8B Model (LLaMA-3 Scale)",
            32,
            4096,
            14336,
            8 * 1024 * 1024 * 1024,
        ),
        (
            "70B Model (LLaMA-3 Scale)",
            80,
            8192,
            28672,
            48 * 1024 * 1024 * 1024,
        ),
    ];

    println!(
        "+--------------------------------+----------+------------+------------+---------------+----------------+"
    );
    println!(
        "| {:<30} | {:>8} | {:>10} | {:>10} | {:>13} | {:>14} |",
        "Scenario / Architecture",
        "Tensors",
        "Params (B)",
        "Solve Time",
        "Memory Margin",
        "Distortion"
    );
    println!(
        "+--------------------------------+----------+------------+------------+---------------+----------------+"
    );

    for (name, layers, hidden, inter, target_vram) in scenarios {
        let tensor_map = generate_synthetic_tensor_map(layers, hidden, inter);
        let n_tensors = tensor_map.len();
        let total_elements: usize = tensor_map.values().map(|(_, e)| *e).sum();
        let total_billions = (total_elements as f64) / 1_000_000_000.0;

        let groups = MckpSolver::build_tensor_groups(&tensor_map)?;

        let kv_cache_bytes = 1024 * 1024 * 1024;
        let overhead_bytes = 512 * 1024 * 1024;

        let iters = 20;
        let start = Instant::now();
        let mut last_plan = None;
        for _ in 0..iters {
            let plan = MckpSolver::solve(&groups, target_vram, kv_cache_bytes, overhead_bytes)?;
            last_plan = Some(plan);
        }
        let elapsed = start.elapsed();
        let avg_ms = (elapsed.as_secs_f64() * 1000.0) / (iters as f64);

        if let Some(plan) = last_plan {
            let margin_gib = (plan.margin_bytes as f64) / (1024.0 * 1024.0 * 1024.0);

            println!(
                "| {:<30} | {:>8} | {:>10.3} | {:>8.3} ms | {:>10.3} GiB | {:>14.2} |",
                name, n_tensors, total_billions, avg_ms, margin_gib, plan.total_distortion
            );
        }
    }

    println!(
        "+--------------------------------+----------+------------+------------+---------------+----------------+"
    );

    Ok(())
}
