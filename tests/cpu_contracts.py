"""Deployment API checks with synthetic token data, without an engine or a sandbox."""

import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from carl_v3.common import CONFIG, read, write


def sampler():
    import torch
    import vllm._C_stable_libtorch
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from carl_v3.deploy import sampler_command

    root = FlexibleArgumentParser()
    ServeSubcommand().subparser_init(root.add_subparsers(dest="command"))
    args = root.parse_args(sampler_command(read(CONFIG), Path("/cpu-fixture/model"))[1:])
    assert args.enable_lora and args.language_model_only and args.tensor_parallel_size == 1
    assert args.logprobs_mode == "processed_logprobs" and args.max_model_len == 8192
    assert args.tool_call_parser == "qwen3_coder" and args.reasoning_parser == "qwen3"
    assert args.host == "127.0.0.1" and not torch.cuda.is_available()
    print("PASS: pinned vLLM CLI parses release launch flags; CUDA extension imports without CUDA visibility")


def learner():
    from accelerate.commands.config.config_args import load_config_from_file
    from carl_v3.common import ROOT
    from carl_v3.train import make_args
    from carl_v3.tree_backend import select_sm80_backend
    from openenv.core.harness.training import TrainingTrace, TrainingTurn
    from trl.experimental.async_grpo.openenv_harness import _turns_from_training_trace
    from trl.trainer.async_grpo_trainer import Rollout, _chain_to_sequences

    select_sm80_backend()
    fsdp = load_config_from_file(str(ROOT / "configs/accelerate-fsdp2.yaml"))
    assert fsdp.num_processes == 2 and fsdp.fsdp_config["fsdp_version"] == 2
    assert fsdp.fsdp_config["fsdp_activation_checkpointing"]
    # Stub the hardware capability gate solely to parse the exact GPU config on CPU.
    with patch("transformers.training_args.is_torch_bf16_gpu_available", return_value=True):
        args = make_args(Path("/tmp/carl-v3-argument-fixture"))
    assert args.num_generations == 4 and args.token_budget == 8192
    assert args.max_tool_calling_iterations == 8 and args.gradient_accumulation_steps == 1
    trace = TrainingTrace(turns=[
        TrainingTurn(node_id="a", prompt_token_ids=[1, 2], completion_token_ids=[3],
                     per_token_logps=[-0.4], loss_mask=[0, 0, 1]),
        TrainingTurn(node_id="b", prompt_token_ids=[1, 5], completion_token_ids=[8],
                     per_token_logps=[-0.3], loss_mask=[0, 0, 1]),
    ])
    rows, drifts = _chain_to_sequences(Rollout([], [], _turns_from_training_trace(trace)))
    assert len(rows) == 2 and drifts == [2]
    assert [row.completion_mask for row in rows] == [[0, 0, 1], [0, 0, 1]]
    print("PASS: FSDP2/AsyncGRPO argument contract and official non-prefix trace reconciliation")


def control():
    from fastapi.testclient import TestClient
    from openenv.core.harness.capture.validate_llm import LLMReport
    from openenv.core.harness.training import TrainingTrace, TrainingTurn
    from carl_v3 import actor
    from carl_v3.archive import collect
    from carl_v3.episodes import validate_trace
    from carl_v3.deploy import summarize

    with TemporaryDirectory(prefix="carl-v3-cpu-contract-") as temp:
        root = Path(temp)
        run, out = root / "run", root / "export"
        write(run / "run-plan.json", {"fixture": "synthetic"})
        report = LLMReport(ok=True, llm_url="http://127.0.0.1:9",
                           model=read(CONFIG)["model"]["repo"], capture_level="tokens",
                           tool_support="ok", logprobs_mode="processed")
        with patch.object(actor, "ASSETS", root), patch.object(actor, "validate_llm", return_value=report):
            write(root / "tasks.json", {})
            app, socket_app = actor.build(run, "smoke")
            assert TestClient(app).get("/health").status_code == 200
            assert TestClient(socket_app).get("/health").status_code == 403
            assert TestClient(socket_app).post(actor.RELAY_ROUTE, json={}).status_code == 403
        turn = TrainingTurn(node_id="engine", prompt_token_ids=[1, 2], completion_token_ids=[3],
                            per_token_logps=[-0.5], loss_mask=[0, 0, 1])
        trace = TrainingTrace(turns=[turn])
        assert validate_trace(trace, {"engine": {"prompt": [1, 2], "completion": [3]}})["supervised"] == 1
        write(run / "episodes/fixture/trace.json", {"api_key": "fixture_secret", "turns": []})
        collect(run, out, raw=True)
        assert read(out / "episodes/fixture/trace.json") == {"turns": []}
        subprocess.run(["sha256sum", "--strict", "--quiet", "-c", "SHA256SUMS"], cwd=out, check=True)
        assert summarize(run, 0, None, None)["optimizer_updates_this_launch"] == 0
    print("PASS: OpenEnv control health, capture-socket route boundary, typed receipt and hashed export fixture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("control", "sampler", "learner"))
    globals()[parser.parse_args().role]()
