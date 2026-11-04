# Target-Driven Mixed-Precision GGUF Quantization Engine - Agent Governance

## 1. Anti-Slop & Engineering Discipline
- Emojis are strictly banned from all source code, comments, CLI output, logging, and documentation.
- No marketing rhetoric or hyperbole ("blazingly fast", "seamless", "magical", "revolutionary"). All documentation and output must adhere to terse, technical systems prose.
- Narrative inline comments explaining self-evident operations are prohibited. Comments are restricted to hardware alignment constraints (`GGUF_DEFAULT_ALIGNMENT = 32`), `ggml` block math invariants, and safety preconditions for `unsafe` blocks.
- Panic triggers (`.unwrap()`, `.expect()`, `todo!()`, `unimplemented!()`) are strictly forbidden in production code paths. They are permitted exclusively inside `#[cfg(test)]`.
- Error handling in library crates (`core`, `gguf-bridge`, `hw-probe`, `py`) must use typed, actionable errors via `thiserror`. Application-level error handling in `cli` uses `anyhow` solely in `main.rs`.

## 2. Memory & Streaming Invariants
- Planning and inspection phase: Heap resident memory consumption must remain `< 50 MB RAM`. Full weight tensors must never be loaded into memory. Remote Hugging Face models are inspected via HTTP Range requests (`bytes=0-7` for little-endian `u64` header size $N$, then `bytes=8..(8+N-1)` for the JSON header). Local `.safetensors` headers are read using `memmap2`.
- Quantization execution phase: Resident Set Size (RSS) must not exceed `< 500 MB RSS Cap`. Input tensors must be mapped with `memmap2::Mmap` and streamed sequentially. Immediately following quantization and serialization of each tensor to the output GGUF file, the processed byte range must be evicted via `madvise(MADV_DONTNEED)` aligned to 4096-byte OS page boundaries.

## 3. Workspace Architecture
```text
.
├── AGENTS.md
├── Cargo.toml
├── .cargo/config.toml
├── .agents/skills/quant-math-ecc/SKILL.md
└── crates/
    ├── core/          # Pure math: ggml block sizes, KV-cache accounting, MCKP solver, safetensors header parser
    ├── gguf-bridge/   # GGUF binary header/tensor writer, mmap + MADV_DONTNEED streaming, llama-quantize bridge
    ├── hw-probe/      # NVML (CUDA), Linux sysfs/AMDGPU, Metal, and system RAM detection
    ├── cli/           # Clap CLI (plan, compress, inspect) + plain-text ASCII tables
    └── py/            # PyO3 + maturin bindings exposing planner and quantizer to Python
```

## 4. Mandatory ECC Verification Gates
Every modification must pass the following verification pipeline before commit:
```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings -D clippy::unwrap_used -D clippy::expect_used -D clippy::todo
cargo test --workspace
```
Failures in any gate require immediate localized correction before progressing.
