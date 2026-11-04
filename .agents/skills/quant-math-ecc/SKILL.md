---
name: quant-math-ecc
description: Mathematical foundations, ggml block structures, KV cache sizing, page eviction, and MCKP solver specification for target-driven mixed-precision quantization.
---

# Quantization Mathematics & Systems Specification

## 1. GGML Block Quantization Math & Structures

In `ggml`, tensor dimensions are grouped into quantized blocks along the contiguous dimension. The number of elements in a block is $B$, and the storage occupied in bytes is $S$.

### Standard Blocks ($B = 32$)
- **F32:** $B = 1$, $S = 4$ bytes (32.00 bpw).
- **F16:** $B = 1$, $S = 2$ bytes (16.00 bpw).
- **Q8_0:** $B = 32$, $S = 34$ bytes (8.50 bpw).
  - Structure: `d: f16` (scale factor, 2 bytes) + `qs: [i8; 32]` (quants, 32 bytes).
- **Q4_0:** $B = 32$, $S = 18$ bytes (4.50 bpw).
  - Structure: `d: f16` (scale factor, 2 bytes) + `qs: [u8; 16]` (nibble pairs, 16 bytes).
- **Q4_1:** $B = 32$, $S = 20$ bytes (5.00 bpw).
  - Structure: `d: f16` (2 bytes) + `m: f16` (min val, 2 bytes) + `qs: [u8; 16]` (16 bytes).
- **Q5_0:** $B = 32$, $S = 22$ bytes (5.50 bpw).
  - Structure: `d: f16` (2 bytes) + `qh: [u8; 4]` (high bits, 4 bytes) + `qs: [u8; 16]` (16 bytes).
- **Q5_1:** $B = 32$, $S = 24$ bytes (6.00 bpw).
  - Structure: `d: f16` (2 bytes) + `m: f16` (2 bytes) + `qh: [u8; 4]` (4 bytes) + `qs: [u8; 16]` (16 bytes).

### K-Quants Super-Blocks ($B = 256$)
K-quants organize 256 weights into a super-block structured into 8 or 16 sub-blocks:

- **Q2_K:** $B = 256$, $S = 84$ bytes (2.625 bpw).
  - Scales: 16-element 4-bit scales packed with 4-bit mins (`scales: [u8; 16]` -> 16 bytes).
  - Quants: 2 bits per weight (`qs: [u8; 64]` -> 64 bytes).
  - Scale master: `d: f16` (2 bytes) + `dmin: f16` (2 bytes).
  - Total: $2 + 2 + 16 + 64 = 84$ bytes.
- **Q3_K:** $B = 256$, $S = 110$ bytes (3.4375 bpw).
  - Quants: 64 bytes low 2-bits + 32 bytes high 1-bit (`hmask: [u8; 32]`).
  - Scales: 12 bytes 6-bit packed scales.
  - Scale master: `d: f16` (2 bytes).
  - Total: $2 + 12 + 32 + 64 = 110$ bytes.
- **Q4_K:** $B = 256$, $S = 144$ bytes (4.50 bpw).
  - Scale master: `d: f16` (2 bytes) + `dmin: f16` (2 bytes).
  - Sub-block scales/mins: 12 bytes packed 6-bit scales (`scales: [u8; 12]`).
  - Quants: 128 bytes 4-bit nibbles (`qs: [u8; 128]`).
  - Total: $2 + 2 + 12 + 128 = 144$ bytes.
- **Q5_K:** $B = 256$, $S = 176$ bytes (5.50 bpw).
  - Scale master: `d: f16` (2 bytes) + `dmin: f16` (2 bytes).
  - Sub-block scales: 12 bytes packed (`scales: [u8; 12]`).
  - High bits: 32 bytes (`qh: [u8; 32]`).
  - Quants: 128 bytes (`qs: [u8; 128]`).
  - Total: $2 + 2 + 12 + 32 + 128 = 176$ bytes.
- **Q6_K:** $B = 256$, $S = 210$ bytes (6.5625 bpw).
  - Quants low 4-bits: 128 bytes (`ql: [u8; 128]`).
  - Quants high 2-bits: 64 bytes (`qh: [u8; 64]`).
  - Scales: 16 bytes 8-bit scales (`scales: [i8; 16]`).
  - Scale master: `d: f16` (2 bytes).
  - Total: $128 + 64 + 16 + 2 = 210$ bytes.
- **Q8_K:** $B = 256$, $S = 290$ bytes (9.0625 bpw).
  - Scale master: `d: f32` (4 bytes).
  - Quants: 256 bytes (`qs: [i8; 256]`).
  - Sums: 16 x 2 bytes (`bsums: [i16; 16]` = 32 bytes).
  - Total: $4 + 256 + 30$ (padded to 290 or 256+34 bytes).

## 2. KV-Cache Memory Accounting

For an autoregressive transformer with:
- $L$: number of decoder layers.
- $C$: sequence context length in tokens.
- $H_{kv}$: number of key-value query heads.
- $D$: head dimension ($D = d_{model} / H_{attn}$).
- $T_{kv}$: KV storage element type (FP16 = 2 bytes, Q8_0 = 1.0625 bytes, Q4_0 = 0.5625 bytes).

The total memory required for storing both Keys and Values is:
$$M_{kv} = 2 \times L \times H_{kv} \times D \times \text{bytes\_per\_weight}(T_{kv}) \times C$$

Per-token KV cache footprint:
$$\Delta M_{kv} = 2 \times L \times H_{kv} \times D \times \text{bytes\_per\_weight}(T_{kv})$$

## 3. MADV_DONTNEED Page-Alignment Invariant

POSIX `madvise(void *addr, size_t length, int advice)` requires that `addr` be a multiple of the system page size ($P = 4096$).

Given an arbitrary byte range $[offset, offset + length)$:
- Eviction start:
  $$P_{start} = offset - (offset \pmod P) = offset \ \& \ \sim(P - 1)$$
- Eviction end:
  $$P_{end} = (offset + length + P - 1) - ((offset + length + P - 1) \pmod P) = (offset + length + P - 1) \ \& \ \sim(P - 1)$$
- Aligned length:
  $$L_{aligned} = P_{end} - P_{start}$$

Only ranges where $L_{aligned} > 0$ and within bounds are dispatched to `madvise(..., MADV_DONTNEED)`.

## 4. Multiple-Choice Knapsack Problem (MCKP) Formulation

### Optimization Goal
Let the model consist of $N$ tensor groups (or individual weight tensors) indexed by $i \in \{1, \dots, N\}$.
For each tensor group $i$, a set of quantization candidate types $J_i = \{1, \dots, K_i\}$ is available.

For each option $j \in J_i$:
- Weight size in bytes: $w_{ij}$
- Relative quantization distortion / error penalty: $e_{ij}$
- Binary decision variable: $x_{ij} \in \{0, 1\}$

Minimize total model distortion subject to VRAM/RAM budget $W_{budget}$:
$$\min \sum_{i=1}^N \sum_{j \in J_i} e_{ij} x_{ij}$$
$$\text{subject to } \sum_{i=1}^N \sum_{j \in J_i} w_{ij} x_{ij} \le W_{budget}$$
$$\sum_{j \in J_i} x_{ij} = 1 \quad \forall i \in \{1, \dots, N\}$$

### Available Weight Budget
Given:
- $V_{avail}$: Target device available memory (e.g. GPU VRAM).
- $M_{kv}$: KV-cache memory requirement.
- $M_{act}$: Activation scratch buffer footprint ($\approx \text{batch} \times C \times d_{model} \times \text{layers}$).
- $M_{overhead}$: CUDA runtime / driver / context overhead ($\approx 400 - 600\text{ MB}$).

$$W_{budget} = V_{avail} - M_{kv} - M_{act} - M_{overhead}$$

### Sensitivity & Importance Weighting
Tensors are assigned structural importance multipliers $\alpha_i$:
- Embeddings (`token_embd.weight`): High sensitivity ($\alpha \approx 1.5 - 2.0$).
- Output Head (`output.weight`): Critical sensitivity ($\alpha \approx 2.0 - 2.5$).
- Attention Q, K, V, O projections: High sensitivity ($\alpha \approx 1.2 - 1.4$).
- MLP Gate, Up projections: Medium sensitivity ($\alpha \approx 1.0$).
- MLP Down projections: Higher sensitivity than Gate/Up ($\alpha \approx 1.2$).
- Layer Norm / RMS Norm: Always kept at FP32 or FP16 ($\alpha = \infty$, fixed).

Distortion metric:
$$e_{ij} = \alpha_i \times \text{base\_error}(j) \times \text{numel}_i$$
where $\text{base\_error}(j)$ reflects empirical SNR / MSE degradation relative to FP16.
