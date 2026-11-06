import os
import struct
import tempfile
import json
import sys

so_path = "/mnt/shared/target-quant-engine/target/debug/libquant_engine.so"
target_link = "/mnt/shared/target-quant-engine/target/debug/quant_engine.so"
if not os.path.exists(target_link) and os.path.exists(so_path):
    os.symlink(so_path, target_link)

sys.path.insert(0, "/mnt/shared/target-quant-engine/target/debug")
import quant_engine

def test_py_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        st_path = os.path.join(tmpdir, "model.safetensors")
        out_gguf = os.path.join(tmpdir, "model.gguf")

        # Create dummy weights: 256 floats in f16 format (512 bytes)
        raw_weights = bytes(512)
        meta = {
            "__metadata__": {"format": "pt"},
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F16",
                "shape": [16, 16],
                "data_offsets": [0, len(raw_weights)]
            },
            "model.layers.0.mlp.down_proj.weight": {
                "dtype": "F16",
                "shape": [16, 16],
                "data_offsets": [len(raw_weights), len(raw_weights) * 2]
            }
        }
        json_bytes = json.dumps(meta).encode("utf-8")
        header_len = len(json_bytes)

        with open(st_path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(json_bytes)
            f.write(raw_weights)
            f.write(raw_weights)

        summary = quant_engine.inspect_model(st_path)
        assert summary.tensor_count == 2
        assert summary.total_elements == 512

        plan = quant_engine.plan_quantization(st_path, 10 * 1024 * 1024, ctx_size=2048, kv_type="q4_0", overhead_mb=1)
        assert plan.total_vram_bytes <= 10 * 1024 * 1024
        assert plan.target_vram_bytes == 10 * 1024 * 1024
        assert len(plan.assignments) == 2
        mix = plan.quantization_mixture()
        assert len(mix) > 0

        written = quant_engine.execute_quantization(st_path, out_gguf, plan)
        assert written > 0
        assert os.path.exists(out_gguf)
        assert os.path.getsize(out_gguf) > 0

        # Verify GGUF inspection in Python
        gguf_summary = quant_engine.inspect_model(out_gguf)
        assert gguf_summary.tensor_count == 2
        assert gguf_summary.total_elements == 512

        # Multi-shard test
        shard_dir = os.path.join(tmpdir, "sharded")
        os.makedirs(shard_dir, exist_ok=True)
        s1 = os.path.join(shard_dir, "model-00001-of-00002.safetensors")
        s2 = os.path.join(shard_dir, "model-00002-of-00002.safetensors")
        idx = os.path.join(shard_dir, "model.safetensors.index.json")

        s1_meta = {"model.embed_tokens.weight": {"dtype": "F16", "shape": [16, 16], "data_offsets": [0, 512]}}
        s1_json = json.dumps(s1_meta).encode("utf-8")
        with open(s1, "wb") as f:
            f.write(struct.pack("<Q", len(s1_json)))
            f.write(s1_json)
            f.write(bytes(512))

        s2_meta = {"lm_head.weight": {"dtype": "F16", "shape": [16, 16], "data_offsets": [0, 512]}}
        s2_json = json.dumps(s2_meta).encode("utf-8")
        with open(s2, "wb") as f:
            f.write(struct.pack("<Q", len(s2_json)))
            f.write(s2_json)
            f.write(bytes(512))

        index_data = {
            "metadata": {"total_size": 1024},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        }
        with open(idx, "w") as f:
            json.dump(index_data, f)

        multi_summary = quant_engine.inspect_model(shard_dir)
        assert multi_summary.tensor_count == 2

        multi_plan = quant_engine.plan_quantization(shard_dir, 100 * 1024 * 1024, ctx_size=2048, kv_type="q4_0", overhead_mb=1)
        multi_out = os.path.join(tmpdir, "multi.gguf")
        multi_written = quant_engine.execute_quantization(shard_dir, multi_out, multi_plan)
        assert multi_written > 0
        assert os.path.exists(multi_out)

    print("Python single-shard and multi-shard integration tests passed successfully!")

if __name__ == "__main__":
    test_py_end_to_end()
