"""Own one bounded local A800 experiment, its services and its evidence."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import httpx

from .common import ASSETS, CONFIG, ROOT, STATE, model_path, read, sha, write
from .preflight import installations, local_docker, require_gpu_topology


def python(role):
    return str(ROOT / ".venvs" / role / "bin/python")


def sampler_command(cfg, path):
    sampler = cfg["sampler"]
    adapters = str(cfg["train"]["max_staleness"] + 2)
    return [
        str(ROOT / ".venvs/sampler/bin/vllm"), "serve", str(path),
        "--served-model-name", cfg["model"]["repo"],
        "--host", "127.0.0.1", "--port", str(sampler["engine_port"]),
        "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--language-model-only",
        "--enforce-eager", "--max-model-len", str(sampler["max_model_len"]),
        "--max-num-seqs", str(sampler["max_num_seqs"]), "--gpu-memory-utilization",
        str(sampler["gpu_memory_utilization"]),
        "--generation-config", "vllm", "--default-chat-template-kwargs", '{"enable_thinking":false}',
        "--enable-auto-tool-choice", "--tool-call-parser", sampler["tool_call_parser"],
        "--reasoning-parser", sampler["reasoning_parser"], "--logprobs-mode", "processed_logprobs",
        "--return-tokens-as-token-ids", "--enable-lora", "--max-lora-rank", str(cfg["train"]["lora_rank"]),
        "--max-loras", adapters, "--max-cpu-loras", adapters,
    ]


def checkpoint_receipt(path):
    from safetensors import safe_open

    weights = path / "adapter_model.safetensors"
    from torch.distributed.checkpoint import FileSystemReader

    files = ["trainer_state.json", "scheduler.pt", "rng_state_0.pth", "rng_state_1.pth",
             "rollout_state.json", "adapter_config.json"]
    for folder in ("pytorch_model_fsdp_0", "optimizer_0"):
        directory = path / folder
        metadata = FileSystemReader(directory).read_metadata()
        if not metadata.state_dict_metadata or not list(directory.glob("*.distcp")):
            raise ValueError(f"Missing sharded checkpoint data: {folder}")
        files += [p.relative_to(path).as_posix() for p in directory.rglob("*") if p.is_file()]
    state = read(path / "trainer_state.json")
    with safe_open(str(weights), framework="pt", device="cpu") as adapter:
        tensor_count = len(adapter.keys())
        if any(".lora_" not in name for name in adapter.keys()):
            raise ValueError("Checkpoint contains a parameter outside LoRA")
    if tensor_count <= 0 or state["global_step"] <= 0:
        raise ValueError("Checkpoint has no valid adapter or optimizer step")
    for filename in files:
        if not (path / filename).is_file() or (path / filename).stat().st_size == 0:
            raise ValueError("Incomplete checkpoint: " + filename)
    return {"checkpoint": str(path), "adapter_sha256": sha(weights), "global_step": state["global_step"],
            "adapter_tensor_count": tensor_count, "world_size": 2,
            "restart_files_sha256": {file: sha(path / file) for file in sorted(files)}}


class LocalRun:
    def __init__(self, folder, cfg, gpus, deadline):
        self.folder, self.cfg, self.gpus, self.deadline = folder, cfg, gpus, deadline
        self.processes = []
        self.stop_requested = False
        self.capture_dir = Path(tempfile.mkdtemp(prefix="carl-v3-"))
        env = {k: v for k, v in os.environ.items()
               if not (k.endswith(("_API_KEY", "_TOKEN")) or k.upper().endswith("_PROXY"))}
        self.container_label = secrets.token_hex(16)
        self.env = {
            **env, "CARL_V3_RUN": str(folder), "CUDA_VISIBLE_DEVICES": "",
            "CARL_V3_CAPTURE_SOCKET": str(self.capture_dir / "capture.sock"),
            "CARL_V3_RUN_LABEL": self.container_label, "CODING_SANDBOX_RUN": self.container_label,
            "OPENENV_HARBOR_TRIALS_DIR": str(folder / "trials"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }

    def child(self, label, command, gpu="", add_env=None):
        self.folder.joinpath("logs").mkdir(exist_ok=True)
        path = self.folder / "logs" / (label + ".log")
        env = {**self.env, "CUDA_VISIBLE_DEVICES": gpu, **(add_env or {})}
        write(self.folder / "logs" / (label + ".command.json"), {"argv": command, "gpu": gpu or None})
        with path.open("wb") as stream:
            child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes.append((label, child))
        return child

    def stage(self, label, command, gpu="", add_env=None):
        child = self.child(label, command, gpu, add_env)
        while child.poll() is None:
            if time.time() >= self.deadline or (self.stop_requested and label != "train"):
                raise TimeoutError(f"Budget/operator stop during {label}")
            time.sleep(.5)
        if child.returncode:
            raise RuntimeError(f"{label} exit {child.returncode}. See {self.folder/'logs'/(label+'.log')}")

    def wait_http(self, port, path, proc):
        until = min(self.deadline, time.time() + 900)
        url = f"http://127.0.0.1:{port}{path}"
        while time.time() < until and not self.stop_requested:
            if not proc.poll() is None:
                raise RuntimeError(f"Local service exited during startup with {proc.returncode}")
            try:
                if httpx.get(url, timeout=3, trust_env=False).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise TimeoutError(f"Local service did not become ready at {url}")

    def layout(self, mode):
        s = self.cfg["sampler"]
        for i, gpu in enumerate(self.gpus):
            role = "sampler" if i == 0 else "learner"
            self.stage(f"gpu-{i}-cuda", [python(role), "-m", "carl_v3.preflight", "--runtime"], gpu)
        self.child("gpu-telemetry", [
            "nvidia-smi", "-i", ",".join(self.gpus),
            "--query-gpu=timestamp,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits", "-l", "1",
        ])
        command = sampler_command(self.cfg, model_path())
        engine = self.child("sampler", command, self.gpus[0],
                            {"VLLM_SERVER_DEV_MODE": "1", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"})
        self.wait_http(s["engine_port"], "/health", engine)
        actor = self.child("openenv", [python("control"), "-m", "carl_v3.actor", mode])
        self.wait_http(s["openenv_port"], "/health", actor)
        socket_path = Path(self.env["CARL_V3_CAPTURE_SOCKET"])
        for _ in range(40):
            if socket_path.is_socket():
                break
            if actor.poll() is not None:
                raise ValueError("Capture proxy exited before binding the sandbox socket")
            time.sleep(.25)
        if not socket_path.is_socket():
            raise ValueError("Capture socket was not created")
        probe = httpx.HTTPTransport(uds=str(socket_path))
        with httpx.Client(transport=probe, base_url="http://sandbox", timeout=4) as client:
            if client.get("/health").status_code != 403 or client.post("/capture/v1/chat/completions",
                                                                    json={}).status_code != 403:
                raise RuntimeError("Sandbox socket accepted a control-plane or unauthenticated request")

    def cleanup(self):
        for _, child in reversed(self.processes):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=22)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
                except ProcessLookupError:
                    child.wait()
        labels = [f"carl-v3.run={self.container_label}", f"coding-agentic-rl.run={self.container_label}"]
        removed = []
        for label in labels:
            ids = subprocess.check_output(["docker", "ps", "-aq", "--filter", f"label={label}"],
                                          timeout=20, text=True).split()
            for identifier in ids:
                subprocess.run(["docker", "rm", "-f", identifier], timeout=20, check=True, capture_output=True)
                removed.append(identifier)
        volumes = subprocess.check_output(["docker", "volume", "ls", "-q", "--filter", f"label={labels[0]}"],
                                          timeout=20, text=True).split()
        for item in volumes:
            subprocess.run(["docker", "volume", "rm", item], timeout=20, check=True, capture_output=True)
        shutil.rmtree(self.capture_dir)
        write(self.folder / "cleanup.json", {"passed": True, "containers_removed": len(removed),
                  "volumes_removed": len(volumes), "process_exit": {name: p.returncode for name, p in self.processes}})


def evidence_key():
    return {"config_sha256": sha(CONFIG),
            "fsdp2_config_sha256": sha(ROOT / "configs/accelerate-fsdp2.yaml"), "task_sha256": sha(ASSETS / "tasks.json"),
            "model_receipt_sha256": sha(ASSETS / "model.json"), "host": socket.gethostname()}


def completed_checkpoint(path):
    acceptance = read(path.parent.parent / "acceptance.json")
    if not acceptance["passed"] or acceptance["mode"] != "train" or acceptance["evidence"] != evidence_key():
        raise ValueError("Checkpoint comes from a different or unaccepted run")
    expected = acceptance["checkpoint"]
    if Path(expected["checkpoint"]).resolve() != path or sha(path / "adapter_model.safetensors") != expected["adapter_sha256"]:
        raise ValueError("Selected checkpoint does not match the verified run receipt")
    for name, digest in expected["restart_files_sha256"].items():
        if sha(path / name) != digest:
            raise ValueError(f"Checkpoint restart file changed: {name}")
    return expected


def summarize(folder, start_ns, checkpoint, target):
    timeline = folder / "timeline.jsonl"
    events = [json.loads(line) for line in timeline.read_text().splitlines()] if timeline.is_file() else []
    counts = Counter(entry["event"] for entry in events)
    rewards = Counter()
    for entry in events:
        if entry["event"] != "group_scored":
            continue
        scores = set(entry["rewards"])
        if None in scores:
            rewards["unscorable"] += 1
        elif scores == {0, 1}:
            rewards["mixed"] += 1
        elif scores == {1}:
            rewards["all_correct"] += 1
        else:
            rewards["all_failed"] += 1
    steps = {entry["step"] for entry in events if entry["event"] == "optimizer_updated"}
    masks = [e for e in events if e["event"] == "effective_mask" and e["step"] in steps]
    sync = [e for e in events if e["event"] == "weight_sync_finished"]
    dequeues = [e for e in events if e["event"] == "sample_dequeued"]
    starts = {(e["capture"], e["index"]): e["time_ns"] for e in events if e["event"] == "model_request_started"}
    requests = [(starts[(e["capture"], e["index"])], e["time_ns"]) for e in events
                if e["event"] == "model_request_finished" and "error" not in e
                and (e["capture"], e["index"]) in starts]
    forward_starts = {(e["learner_rank"], e["next_step"]): e["time_ns"] for e in events
                      if e["event"] == "forward_backward_started"}
    forwards = [(forward_starts[(e["learner_rank"], e["next_step"])], e["time_ns"]) for e in events
                if e["event"] == "forward_backward_finished" and (e["learner_rank"], e["next_step"]) in forward_starts]
    overlap = sum(any(start < finish and begin < end for begin, finish in forwards) for start, end in requests)
    group_versions = {e["group_id"]: e["sampling_policy_version"] for e in events if e["event"] == "group_started"}
    packed_groups = {(e["learner_rank"], e["batch_key"]): e["groups"] for e in events
                     if e["event"] == "batch_collated"}
    forward_ages = Counter(e["learner_policy_version"] - group_versions[group] for e in events
                           if e["event"] == "forward_backward_started"
                           for group in packed_groups.get((e["learner_rank"], e["batch_key"]), [])
                           if group in group_versions)
    telemetry = folder / "logs/gpu-telemetry.log"
    peaks = {}
    if telemetry.is_file():
        with telemetry.open() as stream:
            for row in csv.reader(stream, skipinitialspace=True):
                if len(row) == 4 and row[2].strip().isdigit():
                    peaks[row[1]] = max(peaks.get(row[1], 0), int(row[2]))
    return {
        "elapsed_seconds": (time.time_ns() - start_ns) / 1e9,
        "optimizer_updates_this_launch": len(steps),
        "optimizer_steps_this_launch": sorted(steps),
        "target_global_step": target,
        "verified_global_step": checkpoint["global_step"] if checkpoint else None,
        "training_groups_started": counts["group_started"],
        "training_groups_scored": counts["group_scored"],
        "training_group_rewards": dict(rewards),
        "training_rollouts_started": counts["rollout_started"],
        "training_rollouts_finished": counts["rollout_finished"],
        "training_rollouts_independently_scored": sum(e["event"] == "rollout_finished" and e["reward"] in (0, 1)
                                                     for e in events),
        "supervised_tokens_at_optimizer_updates": sum(e["supervised"] for e in masks),
        "effective_tokens_after_importance_mask": sum(e["trained"] for e in masks),
        "observed_device_memory_peak_mib": peaks,
        "weight_sync_seconds_by_version": {e["policy"]: e["seconds"] for e in sync},
        "dequeue_age_counts": dict(Counter(e["age"] for e in dequeues)),
        "forward_group_rank_age_counts": dict(forward_ages),
        "dequeue_queue_size_max": max((e["queue_size"] for e in dequeues), default=None),
        "request_windows_overlapping_learner": overlap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "evaluate"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("v3-%Y%m%dT%H%M%SZ"))
    parser.add_argument("--wall-minutes", type=int, default=60)
    parser.add_argument("--smoke-run", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-after", type=int, default=1)
    parser.add_argument("--max-groups", type=int)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.run_id):
        parser.error("Run ID must be a simple portable name")
    cfg = read(CONFIG)
    limit = cfg["train"]
    groups = limit["max_groups_per_launch"] if args.max_groups is None else args.max_groups
    if not 1 <= args.wall_minutes * 60 <= limit["max_wall_seconds"] or not 1 <= groups <= limit["max_groups_per_launch"]:
        parser.error("Requested wall or group budget exceeds the profile")
    if not (os.environ.get("TMUX") or os.environ.get("SLURM_JOB_ID")):
        parser.error("Launch long-running experiments inside tmux or an allocated Slurm job")
    if args.mode == "train":
        if args.wall_minutes * 60 < limit["checkpoint_and_probe_reserve_seconds"] + 180:
            parser.error("Training needs at least 11 minutes including checkpoint/probe allowance")
        if args.smoke_run is None:
            parser.error("Training requires --smoke-run from this exact prepared configuration")
        smoke = read(args.smoke_run / "acceptance.json")
        if not smoke["passed"] or smoke["mode"] != "smoke" or smoke["evidence"] != evidence_key():
            raise ValueError("Missing real 27B Pi/isolated verifier/SM80 smoke evidence")
        if not args.resume and args.stop_after != limit["first_stage_updates"]:
            parser.error("The first run must stop after one verified update")
        if args.resume is not None:
            args.resume = args.resume.resolve()
            before = completed_checkpoint(args.resume)
            if not before["global_step"] < args.stop_after <= limit["total_planned_updates"]:
                parser.error("--stop-after is an absolute optimizer step beyond the accepted checkpoint")
    if args.mode == "evaluate":
        if args.checkpoint is None:
            parser.error("Evaluation requires an accepted --checkpoint")
        args.checkpoint = args.checkpoint.resolve()
        completed_checkpoint(args.checkpoint)
    local_docker()
    installations()
    gpus = require_gpu_topology()
    model_path()
    STATE.mkdir(exist_ok=True)
    lock = (STATE / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    folder = STATE / "runs" / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    start_ns = time.time_ns()
    deadline = start_ns / 1e9 + args.wall_minutes * 60
    runner = LocalRun(folder, cfg, gpus, deadline)
    scope = os.environ.get("TMUX") or os.environ["SLURM_JOB_ID"]
    if os.environ.get("TMUX"):
        scope = subprocess.check_output(["tmux", "display-message", "-p", "#S"], text=True).strip()
    print(f"Host {socket.gethostname()}  session {scope}  logs {folder/'logs'}", flush=True)
    write(folder / "run-plan.json", {"mode": args.mode, "gpus": gpus, "scope": scope,
                                    "deadline": deadline, "config": cfg, "evidence": evidence_key(), "max_groups": groups,
                                    "release_source": read(ROOT / "SOURCE.json") if (ROOT / "SOURCE.json").is_file() else None})

    def stop(*_):
        runner.stop_requested = True
        (folder / "stop-requested").touch()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    checkpoint = None
    okay = False
    failures = []
    try:
        if args.mode == "smoke":
            runner.stage("verifier-probe", [python("control"), "-m", "carl_v3.episodes", "verifier-probe"])
            for index, gpu in enumerate(gpus[1:]):
                runner.stage(f"sm80-kernel-{index}", [python("learner"), "-m", "carl_v3.kernel_probe"],
                             gpu, {"CARL_SM80_PROBE_INDEX": str(index)})
            with tempfile.TemporaryDirectory(prefix="nccl-export-", dir=folder) as export:
                runner.stage("nccl-export-probe", [python("learner"), "-m", "torch.distributed.run",
                             "--standalone", "--nnodes=1", "--nproc-per-node=2",
                             str(ROOT / "tests/collective_export.py"), export, "--cuda"], ",".join(gpus[1:]))
                write(folder / "nccl-adapter-probe.json", {
                    "ranks": [read(Path(export) / f"rank{rank}.json") for rank in range(2)]})
        runner.layout(args.mode)
        if args.mode == "smoke":
            runner.stage("pi-smoke", [python("control"), "-m", "carl_v3.episodes", "smoke"])
        elif args.mode == "train":
            with socket.socket() as port:
                port.bind(("127.0.0.1", 0))
                rendezvous = port.getsockname()[1]
            command = [str(ROOT / ".venvs/learner/bin/accelerate"), "launch",
                       "--config_file", str(ROOT / "configs/accelerate-fsdp2.yaml"),
                       "--main_process_ip", "127.0.0.1", "--main_process_port", str(rendezvous),
                       "--module", "carl_v3.train", "--stop-after", str(args.stop_after)]
            if args.resume:
                command += ["--resume", str(args.resume)]
            runner.stage("train", command, ",".join(gpus[1:]), {"CARL_V3_MAX_GROUPS": str(groups),
                         "CARL_V3_DEADLINE": str(deadline - limit["checkpoint_and_probe_reserve_seconds"])})
            receipt = read(folder / "trainer-receipt.json")
            checkpoint_path = folder / "checkpoints" / f"checkpoint-{receipt['global_step']}"
            if (args.resume is None and (not read(folder / "cuda-logprob-parity.json")["passed"]
                                         or not (folder / "first-update.json").is_file()
                                         or not (folder / "policy-probe.json").is_file())):
                raise ValueError("First real optimizer or strategy update lacks evidence")
            runner.stage("checkpoint-reload", [python("learner"), "-m", "carl_v3.deploy", "verify",
                                                str(checkpoint_path)])
            checkpoint = read(folder / "checkpoint-reload.json")
            policy = read(folder / "latest-policy.json")["policy"]
            runner.stage("new-policy", [python("control"), "-m", "carl_v3.episodes", "followup",
                                        "--policy", policy, "--checkpoint", str(checkpoint_path)])
        else:
            runner.stage("paired-eval", [python("control"), "-m", "carl_v3.episodes", "evaluate",
                                         "--checkpoint", str(args.checkpoint)])
        okay = True
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            runner.cleanup()
        except Exception as exc:
            okay = False
            failures.append(f"Cleanup {type(exc).__name__}: {exc}")
            write(folder / "cleanup.json", {"passed": False, "error": failures[-1]})
        finally:
            write(folder / "acceptance.json", {"passed": okay, "mode": args.mode, "failures": failures,
                                                "checkpoint": checkpoint, "evidence": evidence_key(),
                                                "target_reached": checkpoint["global_step"] >= args.stop_after
                                                if checkpoint else None})
            write(folder / "summary.json", summarize(folder, start_ns, checkpoint,
                                                     args.stop_after if args.mode == "train" else None))
            print(f"Run evidence saved in {folder}", flush=True)
    if not okay:
        raise SystemExit(2)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "verify":
        write(Path(os.environ["CARL_V3_RUN"], "checkpoint-reload.json"),
              checkpoint_receipt(Path(sys.argv[2]).resolve()))
    else:
        main()
