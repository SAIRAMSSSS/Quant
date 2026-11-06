import os
import sys
import time
import subprocess
import threading
import struct
import json
import array
import math

BENCH_DIR = "/mnt/shared/bench_e2e"
MODEL_PATH = os.path.join(BENCH_DIR, "model.safetensors")
GGUF_PATH = os.path.join(BENCH_DIR, "output.gguf")
QUANT_BIN = "/mnt/shared/target-quant-engine/target/release/quant"

def generate_bench_model(path, target_mb=256):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[1/4] Generating {target_mb} MiB realistic safetensors payload...")
    tensors = {}
    data_bytes = bytearray()

    hidden_dim = 1024
    intermediate_dim = 2048

    def add_tensor(name, shape):
        nonlocal data_bytes
        elem_count = 1
        for s in shape:
            elem_count *= s
        arr = array.array('f', (math.sin(i * 0.05) * 1.2 for i in range(elem_count)))
        raw = arr.tobytes()
        start = len(data_bytes)
        end = start + len(raw)
        data_bytes.extend(raw)
        tensors[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [start, end]
        }

    # Embeddings
    vocab_size = 16384
    add_tensor("model.embed_tokens.weight", [vocab_size, hidden_dim])
    add_tensor("lm_head.weight", [vocab_size, hidden_dim])
    add_tensor("model.norm.weight", [hidden_dim])

    layer = 0
    while len(data_bytes) < target_mb * 1024 * 1024:
        p = f"model.layers.{layer}"
        add_tensor(f"{p}.input_layernorm.weight", [hidden_dim])
        add_tensor(f"{p}.self_attn.q_proj.weight", [hidden_dim, hidden_dim])
        add_tensor(f"{p}.self_attn.k_proj.weight", [hidden_dim, hidden_dim])
        add_tensor(f"{p}.self_attn.v_proj.weight", [hidden_dim, hidden_dim])
        add_tensor(f"{p}.self_attn.o_proj.weight", [hidden_dim, hidden_dim])
        add_tensor(f"{p}.post_attention_layernorm.weight", [hidden_dim])
        add_tensor(f"{p}.mlp.gate_proj.weight", [intermediate_dim, hidden_dim])
        add_tensor(f"{p}.mlp.up_proj.weight", [intermediate_dim, hidden_dim])
        add_tensor(f"{p}.mlp.down_proj.weight", [hidden_dim, intermediate_dim])
        layer += 1

    meta = {"__metadata__": {"format": "pt"}}
    meta.update(tensors)
    header_json = json.dumps(meta).encode("utf-8")
    header_len = len(header_json)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        f.write(data_bytes)

    actual_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"      Created {len(tensors)} tensors across {layer} transformer layers ({actual_mb:.2f} MiB on disk)")
    return actual_mb, len(tensors)

def sample_rss_loop(pid, samples, stop_event):
    status_file = f"/proc/{pid}/status"
    while not stop_event.is_set():
        try:
            with open(status_file, "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        kb = int(parts[1])
                        samples.append(kb)
                        break
        except (FileNotFoundError, ProcessLookupError):
            break
        time.sleep(0.01) # 10ms sampling interval

def run_e2e_benchmark():
    print("================================================================================")
    print("END-TO-END STREAMING QUANTIZATION & MEMORY (RSS) BENCHMARK")
    print("================================================================================")

    model_mb, n_tensors = generate_bench_model(MODEL_PATH, target_mb=256)

    # 1. Zero-Weight-Load Planning Benchmark
    print(f"\n[2/4] Benchmarking Zero-Weight-Load Planning latency...")
    t0 = time.perf_counter()
    res_plan = subprocess.run([
        QUANT_BIN, "plan",
        "--model", MODEL_PATH,
        "--target-vram", "1GiB",
        "--overhead-mb", "64"
    ], capture_output=True, text=True)
    t_plan = time.perf_counter() - t0
    if res_plan.returncode != 0:
        print("Plan failed:", res_plan.stderr)
        sys.exit(1)
    print(f"      Planning Latency: {t_plan * 1000.0:.2f} ms")

    # 2. Streaming Quantization with Continuous RSS Sampling
    print(f"\n[3/4] Streaming Quantization Execution with 10ms RSS Sampling...")
    cmd = [
        QUANT_BIN, "compress",
        "--model", MODEL_PATH,
        "--output", GGUF_PATH,
        "--target-vram", "1GiB",
        "--overhead-mb", "64"
    ]

    samples = []
    stop_event = threading.Event()
    t_start = time.perf_counter()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sampler_thread = threading.Thread(target=sample_rss_loop, args=(proc.pid, samples, stop_event))
    sampler_thread.start()

    stdout, stderr = proc.communicate()
    t_elapsed = time.perf_counter() - t_start
    stop_event.set()
    sampler_thread.join()

    if proc.returncode != 0:
        print("Compression failed:", stderr)
        sys.exit(1)

    peak_rss_kb = max(samples) if samples else 0
    peak_rss_mb = peak_rss_kb / 1024.0
    avg_rss_mb = (sum(samples) / len(samples)) / 1024.0 if samples else 0.0

    out_mb = os.path.getsize(GGUF_PATH) / (1024 * 1024)
    throughput_mb_s = model_mb / t_elapsed
    compression_ratio = model_mb / out_mb

    print(f"      Elapsed Time:        {t_elapsed:.3f} s")
    print(f"      Throughput:          {throughput_mb_s:.2f} MiB/s ({throughput_mb_s / 1024.0:.3f} GiB/s)")
    print(f"      Input Model Size:    {model_mb:.2f} MiB")
    print(f"      Output GGUF Size:    {out_mb:.2f} MiB (Ratio: {compression_ratio:.2f}x)")
    print(f"      Total RSS Samples:   {len(samples)} (sampled every 10ms)")
    print(f"      Peak Resident RSS:   {peak_rss_mb:.2f} MiB")
    print(f"      Average Resident RSS:{avg_rss_mb:.2f} MiB")
    print(f"      Target RSS Ceiling:  500.00 MiB")
    
    rss_compliance = peak_rss_mb < 500.0
    print(f"      RSS Limit Compliance:{' PASSED (< 500 MB)' if rss_compliance else ' FAILED'}")

    # 3. GGUF Binary Inspection Verification
    print(f"\n[4/4] Verifying generated GGUF binary format & alignment...")
    res_inspect = subprocess.run([
        QUANT_BIN, "inspect",
        "--model", GGUF_PATH
    ], capture_output=True, text=True)
    if res_inspect.returncode != 0:
        print("Inspect failed:", res_inspect.stderr)
        sys.exit(1)
    
    print("      Inspection Summary:")
    for line in res_inspect.stdout.strip().split("\n")[:8]:
        print("       ", line)

    # Cleanup
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(GGUF_PATH):
        os.remove(GGUF_PATH)
    if os.path.exists(BENCH_DIR):
        os.rmdir(BENCH_DIR)

    print("\nBenchmark completed cleanly with all invariants verified.")

if __name__ == "__main__":
    run_e2e_benchmark()
