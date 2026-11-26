"""Prepare a selected, checksummed incremental run export for review and rsync."""

import argparse
from pathlib import Path
import shutil

from .common import ROOT, STATE, read, sha, write


TOP_RECEIPTS = {
    "run-plan.json", "capture-preflight.json", "sm80-kernel-0.json", "sm80-kernel-1.json",
    "verifier-probe.json", "smoke.json", "nccl-adapter-probe.json", "cuda-logprob-parity.json", "cuda-logprob-parity-rank0.json",
    "cuda-logprob-parity-rank1.json", "first-update.json", "latest-policy.json", "policy-probe.json",
    "checkpoint-reload.json", "trainer-config.json", "trainer-receipt.json", "followup.json",
    "eval-plan.json", "evaluation-partial.json", "evaluation.json", "cleanup.json", "acceptance.json",
    "timeline.jsonl", "summary.json",
}

PRIVATE_KEYS = {"api_key", "apikey", "authorization", "password", "access_token", "env", "proxy_headers"}


def scrub(value):
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items() if key.lower() not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    return value


def collect(run, destination, checkpoint=None, raw=False):
    if not (run / "run-plan.json").is_file():
        raise ValueError("No V3 run plan in the selected directory")
    destination.mkdir(parents=True, exist_ok=True)
    selected = {}
    for file in run.iterdir():
        if file.name in TOP_RECEIPTS and file.is_file() and not file.is_symlink():
            selected[file.relative_to(run)] = file
    for file in sorted((run / "logs").glob("*")):
        if file.is_file() and not file.is_symlink():
            selected[file.relative_to(run)] = file
    for trial in sorted((run / "trials").glob("*")):
        for name in ("product-verifier.json", "agent/trajectory.json", "final-artifact.py"):
            target = trial / name
            if target.is_file() and not target.is_symlink() and (raw or name == "product-verifier.json"):
                selected[target.relative_to(run)] = target
    for folder in sorted((run / "episodes").glob("*")):
        for name in ("receipt.json", "trace.json"):
            target = folder / name
            if target.is_file() and not target.is_symlink() and (raw or name == "receipt.json"):
                selected[target.relative_to(run)] = target
    if raw:
        for target in sorted((run / "captures").glob("*.engine.json")):
            if not target.is_symlink():
                selected[target.relative_to(run)] = target

    source_dirs = ("scripts", "src", "images", "tests", "configs", "requirements")
    for directory in source_dirs:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                selected[Path("release", path.relative_to(ROOT))] = path
    for name in ("README.md", "COMPATIBILITY.md", "SOURCE.json", "SHA256SUMS"):
        path = ROOT / name
        if path.is_file() and not path.is_symlink():
            selected[Path("release", name)] = path

    if checkpoint is not None:
        expected = read(run / "checkpoint-reload.json")
        if checkpoint.resolve() != Path(expected["checkpoint"]).resolve() or checkpoint.resolve().parent != (run / "checkpoints").resolve():
            raise ValueError("Use the selected, verified checkpoint from this run")
        required = {**expected["restart_files_sha256"], "adapter_model.safetensors": expected["adapter_sha256"]}
        for name, value in required.items():
            path = checkpoint / name
            if path.is_symlink() or sha(path) != value:
                raise ValueError(f"Checkpoint contents changed: {name}")
            selected[Path("checkpoints", checkpoint.name, name)] = path

    for name, original in selected.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.suffix == ".json" and (name.parts[0] in {"trials", "episodes"}):
            sanitized = scrub(read(original))
            if not target.exists() or read(target) != sanitized:
                write(target, sanitized)
        elif not target.exists() or original.stat().st_size != target.stat().st_size or sha(original) != sha(target):
            shutil.copy2(original, target)
    files = sorted(destination / name for name in selected)
    (destination / "SHA256SUMS").write_text(
        "".join(f"{sha(item)}  {item.relative_to(destination).as_posix()}\n" for item in files))
    print(f"Prepared {len(files)} files for review in {destination}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run", type=Path)
    cli.add_argument("--checkpoint", type=Path)
    cli.add_argument("--with-raw-traces", action="store_true")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    run = args.run.resolve()
    collect(run, args.output or (STATE / "exports" / run.name), args.checkpoint, args.with_raw_traces)


if __name__ == "__main__":
    main()
