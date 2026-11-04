# quant-engine

Target-driven mixed-precision GGUF quantization engine in Rust.

## Architecture
- `crates/core`: GGML block quantization mathematics, KV cache footprint accounting, multiple-choice knapsack problem (MCKP) solver, and zero-weight-load safetensors header parser.
- `crates/gguf-bridge`: Binary GGUF v3 reader/writer, memory-mapped streaming with page-aligned kernel cache eviction (`MADV_DONTNEED`), and quantization execution.
- `crates/hw-probe`: Hardware memory discovery covering NVML (CUDA), Linux sysfs (AMDGPU), and system RAM.
- `crates/cli`: Command-line interface with subcommands `plan`, `compress`, and `inspect`.
- `crates/py`: PyO3 Python bindings for planning, inspection, hardware probing, and quantization.

## Invariants
- Planning and inspection: `< 50 MB RAM` resident memory usage.
- Quantization streaming: `< 500 MB RSS` resident set size limit.
- Zero-panic production discipline: no `.unwrap()`, `.expect()`, `todo!()`, or `unimplemented!()` outside unit tests.
