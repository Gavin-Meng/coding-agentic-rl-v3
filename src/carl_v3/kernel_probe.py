"""SM80-only packed/serial BF16 hybrid forward and packed backward check."""

import os
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from trl.trainer import async_grpo_trainer as trl

from .common import write
from .preflight import runtime
from .tree_backend import select_sm80_backend


def main():
    device = runtime()
    torch.manual_seed(3127)
    select_sm80_backend()
    trl.register_tree_attention()
    # Match the pinned model's head sizes and grouped-query ratios at two layers.
    cfg = Qwen3_5TextConfig(vocab_size=64, hidden_size=512, intermediate_size=1024,
                          num_hidden_layers=2, num_attention_heads=24, num_key_value_heads=4,
                          head_dim=256, linear_conv_kernel_dim=4, linear_key_head_dim=128,
                          linear_value_head_dim=128, linear_num_key_heads=16, linear_num_value_heads=48,
                          layer_types=["linear_attention", "full_attention"], use_cache=False)
    cfg._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    if not trl.patch_tree_recurrent(model, "linear_attention"):
        raise RuntimeError("TRL's pinned gated DeltaNet patch did not recognize the Transformers model")

    sequences = ([3, 4, 5, 6, 14, 15, 16], [3, 4, 5, 6, 20, 21, 22])
    row = trl.TreeRow()
    for ids in sequences:
        row.add({"input_ids": np.asarray(ids, dtype=np.int64),
                 "completion_mask": np.asarray([0]*4 + [1]*3, dtype=np.int64),
                 "old_log_probs": np.zeros(len(ids)), "sampling_mask": [[] for _ in ids],
                 "sampler_ids": np.full((len(ids), 1), -1, dtype=np.int64),
                 "sampler_log_probs": np.full((len(ids), 1), -np.inf),
                 "advantage": 1.0, "group_id": 0})
    with torch.no_grad():
        serial = torch.cat([model(input_ids=torch.tensor([s], device="cuda"), use_cache=False)
                            .logits[0, 3:-1].float() for s in sequences])
    model.set_attn_implementation(trl.TREE_ATTENTION)
    batch = {key: tensor.unsqueeze(0).to("cuda") for key, tensor in row.pack().items()}
    packed = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"], use_cache=False,
                   **trl.unpad_tree(batch)).logits[0, batch["token_index"][0]].float()
    max_diff = (serial - packed).abs().max().item()
    # A deliberately generous threshold for the small BF16 synthetic fixture.
    torch.testing.assert_close(packed, serial, atol=0.05, rtol=0.05)
    packed.square().mean().backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not grads or not all(torch.isfinite(grad).all().item() for grad in grads):
        raise RuntimeError("Synthetic hybrid backward did not produce finite gradients")
    if not any(grad.abs().max().item() > 0 for grad in grads):
        raise RuntimeError("Synthetic backward gradients are all zero")
    write(Path(os.environ["CARL_V3_RUN"], f'sm80-kernel-{os.environ["CARL_SM80_PROBE_INDEX"]}.json'),
          {"passed": True, "fixture": "tiny randomly initialized Qwen3.5 hybrid",
           "packed_serial_max_absolute_error": max_diff, "device": device})


if __name__ == "__main__":
    main()
