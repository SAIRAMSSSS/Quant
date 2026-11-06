# quant-engine

Target-driven mixed-precision GGUF quantization engine written in Rust.

## Architecture

- `crates/core`: GGML block quantization mathematics, KV cache footprint accounting, multiple-choice knapsack problem (MCKP) solver, and zero-weight-load safetensors header parser.
- `crates/gguf-bridge`: Binary GGUF v3 reader/writer, memory-mapped streaming with page-aligned kernel cache eviction (`MADV_DONTNEED`), and AVX2-accelerated SIMD quantization kernels.
- `crates/hw-probe`: Hardware memory discovery covering NVML (CUDA), Linux sysfs (AMDGPU), and system RAM.
- `crates/cli`: Command-line interface (`quant`) with subcommands `plan`, `compress`, and `inspect`.
- `crates/py`: PyO3 Python bindings (`quant_engine`) for hardware probing, planning, inspection, and streaming quantization execution.

## Invariants

- **Planning & Inspection**: `< 50 MB RAM` resident memory footprint. Uses HTTP `Range` requests (`bytes=0-7` followed by `bytes=8..(8+N-1)`) for remote Hugging Face `.safetensors` headers, and `memmap2` for local files.
- **Quantization Streaming**: `< 500 MB RSS` resident set size limit. Evicts processed pages from the OS page cache using `madvise(MADV_DONTNEED)` on 4096-byte boundaries immediately after flushing each tensor block.
- **Memory Alignment**: Hardware data alignment strictly adheres to `GGUF_DEFAULT_ALIGNMENT = 32`.
- **Zero-Panic Production Discipline**: `.unwrap()`, `.expect()`, `todo!()`, and `unimplemented!()` are strictly denied across all non-test crates. Library crates use typed `thiserror` variants; CLI uses `anyhow`.

## CLI Usage

### Build
```bash
cargo build --release
```
The binary will be generated at `target/release/quant`.

### Planning
Generate an MCKP-optimized mixed-precision assignment table targeting hardware VRAM:
```bash
# Auto-detect primary GPU VRAM with 10% safety margin:
./target/release/quant plan --model https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/model.safetensors

# Explicit VRAM budget and custom KV cache config:
./target/release/quant plan \
  --model /path/to/model.safetensors \
  --target-vram 4GiB \
  --ctx-size 8192 \
  --kv-type q4_0 \
  --overhead-mb 512
```

### Quantization Streaming
Stream tensors from `.safetensors` (single file or sharded directory) into a binary GGUF v3 file:
```bash
./target/release/quant compress \
  --model /path/to/model-directory/ \
  --output /path/to/output.gguf \
  --target-vram auto
```

### Inspection
Inspect header metadata, tensor count, element count, and byte sizes without loading model weights into memory:
```bash
# Safetensors (local or remote):
./target/release/quant inspect --model https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/model.safetensors

# Binary GGUF:
./target/release/quant inspect --model /path/to/output.gguf
```

### Hardware Probing
Probe host compute accelerators (CUDA NVML, AMDGPU sysfs, Apple Silicon Metal unified memory) and system RAM:
```bash
./target/release/quant probe
```

## Python API

Install the Python bindings via `maturin`:
```bash
uv tool run maturin develop -m crates/py/Cargo.toml
```

Usage:
```python
import quant_engine

# Probe available system RAM and GPU VRAM
hw = quant_engine.probe_hardware()
target_vram = hw.recommend_vram(0.10)

# Zero-weight-load inspection
summary = quant_engine.inspect_model("https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/model.safetensors")
print(f"Tensor Count: {summary.tensor_count}, Total Elements: {summary.total_elements}")

# Formulate mixed-precision plan via MCKP solver
plan = quant_engine.plan_quantization(
    model_path="https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/model.safetensors",
    target_vram_bytes=0, # 0 = auto-probe hardware
    ctx_size=4096,
    kv_type="f16",
    overhead_mb=512
)
print("Quantization mixture:", plan.quantization_mixture())

# Execute streaming quantization
bytes_written = quant_engine.execute_quantization(
    "/path/to/local_model.safetensors",
    "/path/to/output.gguf",
    plan
)
```

## Verification

The workspace is protected by a strict pre-commit verification gate:
```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings -D clippy::unwrap_used -D clippy::expect_used -D clippy::todo
cargo test --workspace
```
