"""Exercise the pinned upstream LoRA saver with two ranks and two versions."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor, init_device_mesh, Shard
from trl.trainer.async_grpo_trainer import save_lora_adapter

from carl_v3.common import sha


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.cuda:
        from carl_v3.preflight import runtime

        runtime(expected_devices=2)
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    else:
        assert not torch.cuda.is_available()
        device = torch.device("cpu")
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(backend, timeout=timedelta(minutes=3), device_id=device if args.cuda else None)
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    mesh = init_device_mesh(device.type, (2,))
    torch.manual_seed(3127)
    model = get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"])).to(device)
    live = {}
    for i, (name, tensor) in enumerate(model.named_parameters()):
        if ".lora_" not in name:
            continue
        values = torch.arange(tensor.numel(), dtype=tensor.dtype).reshape(tensor.shape) + i
        live[name] = values.detach().cpu()
        if args.cuda:
            with torch.no_grad():
                tensor.copy_(values)
        else:
            layer_name, parameter = name.rsplit(".", 1)
            setattr(model.get_submodule(layer_name), parameter,
                    nn.Parameter(distribute_tensor(values.clone(), mesh, [Shard(0)])))
    if args.cuda:
        fully_shard(model, mesh=mesh)
    assert all(isinstance(tensor, DTensor) for name, tensor in model.named_parameters() if name in live)
    adapter_only = get_peft_model_state_dict(model, state_dict=live)
    receipts = []
    for version in (1, 2):
        directory = output / f"trl-policy-v{version}"
        saver = SimpleNamespace(device=device, is_main_process=rank == 0)
        save_lora_adapter(model, saver, "default", str(directory))
        dist.barrier()
        exported = load_file(directory / "adapter_model.safetensors")
        assert set(exported) == set(adapter_only) and exported
        for name, value in exported.items():
            torch.testing.assert_close(value, adapter_only[name])
        receipts.append({"rank": rank, "backend": backend, "version": version, "tensors": len(exported),
                         "adapter_sha256": sha(directory / "adapter_model.safetensors"), "passed": True})
        if version == 1:
            with torch.no_grad():
                for name, tensor in model.named_parameters():
                    if name in live:
                        tensor.to_local().add_(1)
                        live[name].add_(1)
            adapter_only = get_peft_model_state_dict(model, state_dict=live)
    (output / f"rank{rank}.json").write_text(json.dumps(receipts) + "\n")
    dist.destroy_process_group()
    if rank == 0:
        print(f"PASS: upstream two-rank {backend} DTensor all-gather and both versioned adapter values")


if __name__ == "__main__":
    main()
