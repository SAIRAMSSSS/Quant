# quant

Target-driven mixed-precision GGUF quantization engine written in Rust.

Uniform quantization (for example, applying Q4_K_M across an entire network) forces a binary trade-off: either the model fits within hardware VRAM with uniform quality degradation, or it exceeds VRAM and drops to CPU offloading. `quant` frames mixed-precision quantization as a Multiple-Choice Knapsack Problem (MCKP). Given a target memory budget (such as GPU VRAM), context window length, and runtime overhead, it calculates the optimal layer-by-layer quantization mixture that minimizes cumulative model distortion while guaranteeing the final artifact and its KV cache fit within the hardware budget.

---

## Technical Overview

### 1. Mathematical Formulation (MCKP)

Let $T$ be the set of tensors in a model. For each tensor $i \in T$, a candidate set of GGML quantization types $Q_i$ is evaluated.

Each choice $q \in Q_i$ has:
- A memory weight $w(i, q)$ in bytes, computed from GGML block geometry and padding constraints.
- An estimated quantization distortion $\delta(i, q)$, scaled by tensor architectural sensitivity $s_i$:

$$\text{Distortion}(i, q) = s_i \cdot \delta(i, q)$$

The optimization objective minimizes cumulative distortion across all tensors:

$$\min \sum_{i \in T} s_i \cdot \delta(i, q_i)$$

Subject to:

$$\sum_{i \in T} w(i, q_i) + \text{KV}(L, \text{type}) + \text{Overhead} \le C_{\text{target}}$$

Where:
- $\text{KV}(L, \text{type})$ is the key-value cache footprint for context length $L$ across attention heads and layers.
- $\text{Overhead}$ accounts for CUDA runtime contexts, compute scratch buffers, and display server allocations.
- $C_{\text{target}}$ is the hardware memory ceiling (e.g. 3.598 GiB on a 4.0 GiB GPU with 10% headroom).

Sensitive layers (embeddings, LM head, normalization, attention projections) are preferentially allocated higher bit-widths (F32, F16, Q8_0, Q6_K), while bulk feed-forward layers are compressed to denser representations (Q5_K, Q4_K, Q4_0). All 1D tensors (biases and RMSNorm scales) are preserved in F32 to prevent runtime kernel exceptions.

---

## Architecture

```text
crates/
├── core/         # GGML block math, KV-cache equations, MCKP solver, and safetensors parsers
├── gguf-bridge/  # GGUF binary writer/reader, chunked streaming, and AVX2/scalar kernels
├── hw-probe/     # NVML (CUDA), Linux sysfs (AMDGPU), Metal, and system RAM discovery
├── cli/          # Command-line binary (`quant`) with plain-text ASCII reporting
└── py/           # PyO3 Python bindings (`quant_engine`)
```

### Memory and Streaming Invariants

- **Zero-Weight-Load Planning (`< 50 MB RAM`):** Remote Hugging Face repositories are inspected via HTTP `Range` requests (`bytes=0-7` for header length, followed by `bytes=8..(8+N-1)` for JSON metadata). Local files use zero-copy metadata parsing. Full model weights are never loaded during planning.
- **Constant-Memory Streaming (`< 500 MB RSS`):** During quantization, tensors are processed in 64K-element chunks (`CHUNK_ELEMS = 65,536`). Working buffers are capped at ~256 KB. After each tensor block is flushed to the output file, `madvise(MADV_DONTNEED)` is issued on 4096-byte page boundaries to evict processed input pages from the OS resident set. Memory usage remains bounded under 350 MiB regardless of whether the model has 0.5B or 120B parameters.
- **Hardware Alignment:** Binary tensor offsets strictly adhere to `GGUF_DEFAULT_ALIGNMENT = 32`.
- **Zero-Panic Discipline:** `.unwrap()`, `.expect()`, and `todo!()` are strictly forbidden outside of unit tests. Library crates use typed `thiserror` variants; the CLI binary uses `anyhow`.

---

## Empirical Benchmarks

Measured on host hardware: AMD Ryzen 5, 24 GiB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GiB VRAM), Linux 6.12.

| Metric | Measured Result | Constraint / Baseline |
| :--- | :--- | :--- |
| **Streaming Peak RSS** | **313.62 MiB** | `< 500 MB` RSS Cap |
| **Quantization Duration (0.5B, 291 layers)** | **6.627 seconds** | 3.3x speedup over non-chunked baseline |
| **Remote Header Inspection Latency** | **1.82 seconds** | Zero-weight load over HTTP |
| **Remote Planning RAM** | **11.4 MiB** | `< 50 MB` RAM Cap |
| **Inference Generation (Qwen2.5-Coder-0.5B Mixed)** | **59.61 tokens/sec** | Native `llama-server` on RTX 3050 |
| **Workspace Unit Tests** | **39/39 passing** | 100% test pass rate across 5 crates |

---

## Installation & Compilation

### Requirements
- Rust 1.80+ (stable toolchain)
- Cargo

### Build Release Binary
```bash
cargo build --release
```
The compiled executable is located at `target/release/quant`.

---

## CLI Reference

### 1. Hardware Probing
Probe installed accelerators and query recommended VRAM target budgets:
```bash
./target/release/quant probe
```

Output:
```text
SYSTEM HARDWARE MEMORY PROFILE
System RAM:
  Total RAM:       23.151 GiB
  Available RAM:   16.297 GiB

Detected Accelerators:
+--------------------------------------+----------+--------------+--------------+
| Device Name                          | Backend  |   Total VRAM |    Free VRAM |
+--------------------------------------+----------+--------------+--------------+
| NVIDIA GeForce RTX 3050 Laptop GPU   | Cuda     |    3.998 GiB |    2.129 GiB |
+--------------------------------------+----------+--------------+--------------+

Recommended VRAM Targets:
  10% Headroom Margin:    3.598 GiB
  15% Headroom Margin:    3.398 GiB
  20% Headroom Margin:    3.198 GiB
```

### 2. Planning (Zero-Weight-Load)
Generate an optimal mixed-precision tensor assignment table without downloading model weights:
```bash
# Remote Hugging Face model:
./target/release/quant plan \
  --model https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/model.safetensors \
  --target-vram 3.598GiB \
  --ctx-size 16384

# Local safetensors file or multi-shard directory:
./target/release/quant plan \
  --model /path/to/model/ \
  --target-vram 3.598GiB \
  --ctx-size 8192 \
  --kv-type q8_0 \
  --overhead-mb 500
```

### 3. Compression (Streaming Quantization)
Stream tensors into an optimized mixed-precision GGUF file:
```bash
./target/release/quant compress \
  --model /path/to/model.safetensors \
  --output /path/to/output-mixed.gguf \
  --target-vram 3.598GiB \
  --ctx-size 32768
```

Supports both single-file safetensors and multi-shard directory structures (`model.safetensors.index.json`).

### 4. Inspection
Inspect safetensors or GGUF metadata, tensor shapes, alignments, and dtypes:
```bash
# Safetensors inspection (local or remote URL):
./target/release/quant inspect --model /path/to/model.safetensors

# GGUF inspection:
./target/release/quant inspect --model /path/to/output.gguf
```

---

## Python API

Install Python bindings via Maturin:
```bash
pip install maturin
maturin develop -m crates/py/Cargo.toml --release
```

Usage:
```python
import quant_engine

# 1. Discover hardware
hw = quant_engine.probe_hardware()
target_vram = hw.recommend_vram(0.10)

# 2. Inspect remote header (Zero-Weight-Load)
summary = quant_engine.inspect_model("https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/model.safetensors")
print(f"Tensors: {summary.tensor_count}, Total Elements: {summary.total_elements}")

# 3. Solve MCKP allocation plan
plan = quant_engine.plan_quantization(
    model_path="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/model.safetensors",
    target_vram_bytes=target_vram,
    ctx_size=16384,
    kv_type="q8_0",
    overhead_mb=500
)
print("Quantization mixture:", plan.quantization_mixture())

# 4. Stream quantization
bytes_written = quant_engine.execute_quantization(
    "/path/to/local/model.safetensors",
    "/path/to/output.gguf",
    plan
)
print(f"Written: {bytes_written / (1024**3):.2f} GiB")
```

---

## Verification Gates

The repository enforces pre-commit verification gates. All checks must pass before changes are accepted:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings -D clippy::unwrap_used -D clippy::expect_used -D clippy::todo
cargo test --workspace
```
