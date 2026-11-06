use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use quant_core::mckp::{MckpSolver, QuantizationPlan};
use quant_core::safetensors::{fetch_remote_header, parse_from_file, SafetensorsHeader};
use quant_core::types::GgmlType;
use quant_gguf_bridge::reader::inspect_gguf_file;
use quant_gguf_bridge::streamer::StreamQuantizer;
use quant_hw_probe::probe_hardware;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "quant",
    author = "Staff Systems & ML Infrastructure Engineer",
    version = "0.1.0",
    about = "Target-driven mixed-precision GGUF quantization engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Plan {
        #[arg(short, long, help = "Path to local model file or remote HF URL")]
        model: String,

        #[arg(
            short,
            long,
            default_value = "auto",
            help = "Target VRAM budget (e.g. '4GB', '3.5GiB', or 'auto')"
        )]
        target_vram: String,

        #[arg(short, long, default_value_t = 4096, help = "Context length in tokens")]
        ctx_size: usize,

        #[arg(
            short,
            long,
            default_value = "f16",
            help = "KV cache element type (f16, q8_0, q4_0)"
        )]
        kv_type: String,

        #[arg(long, default_value_t = 512, help = "Runtime overhead budget in MiB")]
        overhead_mb: usize,
    },

    Compress {
        #[arg(short, long, help = "Path to input .safetensors model")]
        model: PathBuf,

        #[arg(short, long, help = "Path to output .gguf model")]
        output: PathBuf,

        #[arg(
            short,
            long,
            default_value = "auto",
            help = "Target VRAM budget (e.g. '4GB', '3.5GiB', or 'auto')"
        )]
        target_vram: String,

        #[arg(short, long, default_value_t = 4096, help = "Context length in tokens")]
        ctx_size: usize,

        #[arg(
            short,
            long,
            default_value = "f16",
            help = "KV cache element type (f16, q8_0, q4_0)"
        )]
        kv_type: String,

        #[arg(long, default_value_t = 512, help = "Runtime overhead budget in MiB")]
        overhead_mb: usize,
    },

    Inspect {
        #[arg(
            short,
            long,
            help = "Path to local file (.safetensors, .gguf) or remote URL"
        )]
        model: String,
    },

    #[command(about = "Probe host hardware accelerators and available memory")]
    Probe,
}

fn parse_size_to_bytes(s: &str) -> Result<u64> {
    let lower = s.to_ascii_lowercase().trim().to_string();
    if lower == "auto" {
        let hw = probe_hardware();
        let target = hw.recommend_vram_target(0.10);
        return Ok(target);
    }

    if let Some(num) = lower.strip_suffix("gib") {
        let val: f64 = num.trim().parse().context("invalid GiB number")?;
        return Ok((val * 1024.0 * 1024.0 * 1024.0) as u64);
    }
    if let Some(num) = lower.strip_suffix("gb") {
        let val: f64 = num.trim().parse().context("invalid GB number")?;
        return Ok((val * 1_000_000_000.0) as u64);
    }
    if let Some(num) = lower.strip_suffix("mib") {
        let val: f64 = num.trim().parse().context("invalid MiB number")?;
        return Ok((val * 1024.0 * 1024.0) as u64);
    }
    if let Some(num) = lower.strip_suffix("mb") {
        let val: f64 = num.trim().parse().context("invalid MB number")?;
        return Ok((val * 1_000_000.0) as u64);
    }

    let raw: u64 = lower.parse().context("invalid byte count or unit")?;
    Ok(raw)
}

fn extract_model_tensors_map(header: &SafetensorsHeader) -> HashMap<String, (Vec<usize>, usize)> {
    let mut map = HashMap::with_capacity(header.tensors.len() + 1);
    for (k, v) in &header.tensors {
        map.insert(k.clone(), (v.shape.clone(), v.element_count()));
    }
    if !map.contains_key("lm_head.weight") {
        if let Some((shape, count)) = map.get("model.embed_tokens.weight").cloned() {
            map.insert("lm_head.weight".to_string(), (shape, count));
        }
    }
    map
}

fn estimate_kv_cache(
    header: &SafetensorsHeader,
    target: &str,
    ctx_len: usize,
    kv_type: GgmlType,
) -> usize {
    let mut n_layers = 0usize;
    let mut n_kv_heads = 8usize;
    let mut head_dim = 128usize;

    let p = Path::new(target);
    let config_path = if p.is_dir() {
        p.join("config.json")
    } else {
        p.parent().unwrap_or(Path::new(".")).join("config.json")
    };

    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(l) = val.get("num_hidden_layers").and_then(|v| v.as_u64()) {
                    n_layers = l as usize;
                }
                if let Some(kv) = val.get("num_key_value_heads").and_then(|v| v.as_u64()) {
                    n_kv_heads = kv as usize;
                }
                if let (Some(h), Some(heads)) = (
                    val.get("hidden_size").and_then(|v| v.as_u64()),
                    val.get("num_attention_heads").and_then(|v| v.as_u64()),
                ) {
                    if let Some(d) = h.checked_div(heads) {
                        head_dim = d as usize;
                    }
                }
            }
        }
    }

    if n_layers == 0 {
        for (name, meta) in &header.tensors {
            if name.contains("layers.") {
                let parts: Vec<&str> = name.split('.').collect();
                for (idx, &part) in parts.iter().enumerate() {
                    if part == "layers" && idx + 1 < parts.len() {
                        if let Ok(num) = parts[idx + 1].parse::<usize>() {
                            if num + 1 > n_layers {
                                n_layers = num + 1;
                            }
                        }
                    }
                }
            }
            if name.contains("k_proj") && meta.shape.len() >= 2 {
                head_dim = meta.shape[1].min(128);
            }
        }
    }

    if n_layers == 0 {
        n_layers = 32;
    }

    let cfg = quant_core::KvCacheConfig::new(n_layers, n_kv_heads, head_dim, ctx_len, kv_type);
    cfg.total_bytes().unwrap_or(512 * 1024 * 1024)
}

fn load_safetensors_header(target: &str) -> Result<SafetensorsHeader> {
    if target.starts_with("http://") || target.starts_with("https://") {
        let hdr =
            fetch_remote_header(target).context("failed to fetch remote safetensors header")?;
        Ok(hdr)
    } else {
        let path = Path::new(target);
        if !path.exists() {
            bail!("input file not found: {}", target);
        }
        if path.is_dir()
            || path.file_name().and_then(|s| s.to_str()) == Some("model.safetensors.index.json")
        {
            let multi = quant_core::load_multi_shard_headers(path)
                .context("failed to load multi-shard safetensors headers")?;
            Ok(SafetensorsHeader {
                tensors: multi.combined_tensors,
                metadata: multi.combined_metadata,
                header_size: 0,
            })
        } else {
            let hdr = parse_from_file(path).context("failed to parse local safetensors header")?;
            Ok(hdr)
        }
    }
}

fn render_plan_ascii_table(plan: &QuantizationPlan) {
    println!(
        "+--------------------------------------------------+--------------+--------------------+------------+--------+--------+------------+------------+"
    );
    println!(
        "| {:<48} | {:<12} | {:<18} | {:>10} | {:<6} | {:>6} | {:>10} | {:>10} |",
        "Tensor Name", "Role", "Shape", "Elements", "Quant", "BPW", "Size (MiB)", "Distortion"
    );
    println!(
        "+--------------------------------------------------+--------------+--------------------+------------+--------+--------+------------+------------+"
    );

    let display_limit = 24;
    let total_items = plan.tensor_plans.len();

    for (i, p) in plan.tensor_plans.iter().enumerate() {
        if total_items > display_limit && i == 12 {
            println!(
                "| {:<48} | {:<12} | {:<18} | {:>10} | {:<6} | {:>6} | {:>10} | {:>10} |",
                format!("... [{} tensors omitted for brevity] ...", total_items - 20),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-"
            );
            continue;
        }
        if total_items > display_limit && i > 12 && i < total_items - 8 {
            continue;
        }

        let name_trunc = if p.name.len() > 48 {
            format!("{}...", &p.name[..45])
        } else {
            p.name.clone()
        };

        let shape_str = format!("{:?}", p.shape);
        let shape_trunc = if shape_str.len() > 18 {
            format!("{}...", &shape_str[..15])
        } else {
            shape_str
        };

        let size_mb = (p.size_bytes as f64) / (1024.0 * 1024.0);
        let role_str = format!("{:?}", p.role);

        println!(
            "| {:<48} | {:<12} | {:<18} | {:>10} | {:<6} | {:>6.2} | {:>10.2} | {:>10.4} |",
            name_trunc,
            role_str,
            shape_trunc,
            p.num_elements,
            p.selected_type.as_str(),
            p.bpw,
            size_mb,
            p.distortion
        );
    }

    println!(
        "+--------------------------------------------------+--------------+--------------------+------------+--------+--------+------------+------------+"
    );

    let weight_gib = (plan.total_weight_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
    let kv_gib = (plan.kv_cache_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
    let overhead_gib = (plan.overhead_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
    let total_gib = (plan.total_vram_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
    let target_gib = (plan.target_vram_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
    let margin_gib = (plan.margin_bytes as f64) / (1024.0 * 1024.0 * 1024.0);

    println!("\nPLAN SUMMARY");
    println!("  Target VRAM Budget:    {:>8.3} GiB", target_gib);
    println!("  Total Weight Footprint:{:>8.3} GiB", weight_gib);
    println!("  KV Cache Footprint:    {:>8.3} GiB", kv_gib);
    println!("  Runtime Overhead:      {:>8.3} GiB", overhead_gib);
    println!("  ---------------------------------------");
    println!("  Total Projected Memory:{:>8.3} GiB", total_gib);
    println!("  Memory Margin:         {:>8.3} GiB", margin_gib);
    println!("  Cumulative Distortion: {:>8.4}", plan.total_distortion);

    let hist = plan.type_histogram();
    let mut hist_vec: Vec<(GgmlType, usize)> = hist.into_iter().collect();
    hist_vec.sort_by_key(|b| b.1);
    hist_vec.reverse();

    print!("  Quantization Mixture:  ");
    for (t, count) in hist_vec {
        print!("{}: {} tensors  ", t.as_str(), count);
    }
    println!();
}

fn render_probe_ascii_table(hw: &quant_hw_probe::HardwareProfile) {
    println!("SYSTEM HARDWARE MEMORY PROFILE");
    println!("System RAM:");
    let total_ram_gib = (hw.system_ram.total_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
    let avail_ram_gib = (hw.system_ram.available_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
    println!("  Total RAM:     {:>8.3} GiB", total_ram_gib);
    println!("  Available RAM: {:>8.3} GiB", avail_ram_gib);

    println!("\nDetected Accelerators:");
    if hw.gpus.is_empty() {
        println!("  No discrete GPU detected (falling back to system RAM budget).");
    } else {
        println!(
            "+--------------------------------------------------+----------+--------------+--------------+"
        );
        println!(
            "| {:<48} | {:<8} | {:>12} | {:>12} |",
            "Device Name", "Backend", "Total VRAM", "Free VRAM"
        );
        println!(
            "+--------------------------------------------------+----------+--------------+--------------+"
        );
        for gpu in &hw.gpus {
            let total_vram_gib = (gpu.total_vram_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
            let free_vram_gib = (gpu.free_vram_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
            let name_trunc = if gpu.name.len() > 48 {
                format!("{}...", &gpu.name[..45])
            } else {
                gpu.name.clone()
            };
            println!(
                "| {:<48} | {:<8?} | {:>10.3} GiB | {:>10.3} GiB |",
                name_trunc, gpu.backend, total_vram_gib, free_vram_gib
            );
        }
        println!(
            "+--------------------------------------------------+----------+--------------+--------------+"
        );
    }

    println!("\nRecommended VRAM Targets:");
    let rec10 = (hw.recommend_vram_target(0.10) as f64) / (1024.0 * 1024.0 * 1024.0);
    let rec15 = (hw.recommend_vram_target(0.15) as f64) / (1024.0 * 1024.0 * 1024.0);
    let rec20 = (hw.recommend_vram_target(0.20) as f64) / (1024.0 * 1024.0 * 1024.0);
    println!("  10% Headroom Margin: {:>8.3} GiB", rec10);
    println!("  15% Headroom Margin: {:>8.3} GiB", rec15);
    println!("  20% Headroom Margin: {:>8.3} GiB", rec20);
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Probe => {
            let hw = probe_hardware();
            render_probe_ascii_table(&hw);
        }
        Commands::Plan {
            model,
            target_vram,
            ctx_size,
            kv_type,
            overhead_mb,
        } => {
            let target_bytes = parse_size_to_bytes(&target_vram)?;
            let kv_enum = GgmlType::from_str_case_insensitive(&kv_type).unwrap_or(GgmlType::F16);
            let overhead_bytes = overhead_mb * 1024 * 1024;

            println!(
                "Inspecting model header from: {} (Zero-Weight-Load Planning)",
                model
            );
            let header = load_safetensors_header(&model)?;
            let tensor_map = extract_model_tensors_map(&header);
            let kv_bytes = estimate_kv_cache(&header, &model, ctx_size, kv_enum);

            let groups = MckpSolver::build_tensor_groups(&tensor_map)?;
            let plan = MckpSolver::solve(&groups, target_bytes as usize, kv_bytes, overhead_bytes)?;

            render_plan_ascii_table(&plan);
        }

        Commands::Compress {
            model,
            output,
            target_vram,
            ctx_size,
            kv_type,
            overhead_mb,
        } => {
            if !model.exists() {
                bail!("input model path does not exist: {}", model.display());
            }

            let target_bytes = parse_size_to_bytes(&target_vram)?;
            let kv_enum = GgmlType::from_str_case_insensitive(&kv_type).unwrap_or(GgmlType::F16);
            let overhead_bytes = overhead_mb * 1024 * 1024;

            println!("Generating quantization plan for: {}", model.display());
            let header = load_safetensors_header(&model.to_string_lossy())?;
            let tensor_map = extract_model_tensors_map(&header);
            let kv_bytes = estimate_kv_cache(&header, &model.to_string_lossy(), ctx_size, kv_enum);

            let groups = MckpSolver::build_tensor_groups(&tensor_map)?;
            let plan = MckpSolver::solve(&groups, target_bytes as usize, kv_bytes, overhead_bytes)?;

            render_plan_ascii_table(&plan);

            println!(
                "\nExecuting streaming quantization to: {}",
                output.display()
            );
            let written = StreamQuantizer::stream_safetensors_to_gguf(&model, &output, &plan)?;

            let written_gib = (written as f64) / (1024.0 * 1024.0 * 1024.0);
            println!(
                "Quantization complete. Payload size: {:.3} GiB. Output: {}",
                written_gib,
                output.display()
            );
        }

        Commands::Inspect { model } => {
            let path = Path::new(&model);
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                let gguf_hdr = inspect_gguf_file(path)?;
                println!("GGUF File Inspection: {}", model);
                println!("  Version:           {}", gguf_hdr.version);
                println!("  Alignment:         {}", gguf_hdr.alignment);
                println!("  Tensor Count:      {}", gguf_hdr.tensor_count);
                println!("  Metadata KV Count: {}", gguf_hdr.metadata_kv_count);
                println!("  Data Start Offset: 0x{:X}", gguf_hdr.data_offset);

                println!("\n  Metadata Entries:");
                for (k, v) in &gguf_hdr.metadata {
                    println!("    {}: {:?}", k, v);
                }

                println!("\n  Tensors (first 10):");
                for (i, t) in gguf_hdr.tensors.iter().take(10).enumerate() {
                    println!(
                        "    [{:03}] {:<40} {:<6} dims={:?} offset=0x{:X}",
                        i,
                        t.name,
                        t.ggml_type.as_str(),
                        t.dimensions,
                        t.offset
                    );
                }
            } else {
                let header = load_safetensors_header(&model)?;
                println!("Safetensors Inspection: {}", model);
                println!("  Header Size: {} bytes", header.header_size);
                println!("  Tensor Count: {}", header.tensors.len());

                let mut total_elements = 0usize;
                let mut total_bytes = 0usize;

                for meta in header.tensors.values() {
                    total_elements += meta.element_count();
                    total_bytes += meta.byte_size();
                }

                let total_mparams = (total_elements as f64) / 1_000_000.0;
                let total_gib = (total_bytes as f64) / (1024.0 * 1024.0 * 1024.0);

                println!("  Total Parameters: {:.2}M", total_mparams);
                println!("  Raw Weight Size:  {:.2} GiB", total_gib);

                if !header.metadata.is_empty() {
                    println!("\n  Metadata:");
                    for (k, v) in &header.metadata {
                        println!("    {}: {}", k, v);
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_size_to_bytes() {
        assert_eq!(parse_size_to_bytes("4GB").unwrap(), 4_000_000_000);
        assert_eq!(parse_size_to_bytes("2GiB").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size_to_bytes("512MiB").unwrap(), 512 * 1024 * 1024);
        assert!(parse_size_to_bytes("auto").unwrap() > 0);
    }

    #[test]
    fn test_cli_parse_plan() {
        let args = [
            "quant",
            "plan",
            "--model",
            "model.safetensors",
            "--target-vram",
            "4GB",
        ];
        let parsed = Cli::try_parse_from(args);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_cli_parse_inspect() {
        let args = ["quant", "inspect", "--model", "test.gguf"];
        let parsed = Cli::try_parse_from(args);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_cli_parse_probe() {
        let args = ["quant", "probe"];
        let parsed = Cli::try_parse_from(args).unwrap();
        assert!(matches!(parsed.command, Commands::Probe));
    }

    #[test]
    fn test_load_safetensors_header_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let s1_path = tmp.path().join("model-00001-of-00001.safetensors");
        let s1_json = r#"{"model.embed_tokens.weight":{"dtype":"F16","shape":[16,16],"data_offsets":[0,512]}}"#;
        let mut s1_buf = Vec::new();
        s1_buf.extend_from_slice(&(s1_json.len() as u64).to_le_bytes());
        s1_buf.extend_from_slice(s1_json.as_bytes());
        s1_buf.extend_from_slice(&vec![0u8; 512]);
        std::fs::write(&s1_path, s1_buf).unwrap();

        let hdr = load_safetensors_header(tmp.path().to_str().unwrap()).unwrap();
        assert_eq!(hdr.tensors.len(), 1);
        assert!(hdr.tensors.contains_key("model.embed_tokens.weight"));
    }
}
