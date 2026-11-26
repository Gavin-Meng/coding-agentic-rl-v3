"""Shared configuration and receipts for the V3 deployment."""

from pathlib import Path
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/qwen27b_a800.yaml"
STATE = ROOT / ".state"
ASSETS = STATE / "assets"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    partial.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def event(kind, **attrs):
    line = json.dumps({"event": kind, "time_ns": time.time_ns(), **attrs}, allow_nan=False) + "\n"
    with Path(os.environ["CARL_V3_RUN"], "timeline.jsonl").open("ab", buffering=0) as file:
        file.write(line.encode())


def model_path():
    receipt = read(ASSETS / "model.json")
    cfg = read(CONFIG)["model"]
    if receipt["repo"] != cfg["repo"] or receipt["revision"] != cfg["revision"]:
        raise ValueError("Model snapshot differs from the locked revision")
    path = Path(receipt["path"])
    if path.name != cfg["revision"] or sha(path / "config.json") != receipt["config_sha256"]:
        raise ValueError("Model snapshot or config changed")
    if sha(path / "chat_template.jinja") != receipt["chat_template_sha256"]:
        raise ValueError("Model chat template changed")
    for name, size in receipt["weight_bytes"].items():
        if (path / name).stat().st_size != size:
            raise ValueError("Model shard changed: " + name)
    return path


def grade_program(task, source, root="/work/repo"):
    """Wrap the file in the V2 structured-return interface, run only in Docker."""
    return f"""
def __carl_grade__(*args):
    import importlib
    import sys
    import types
    sys.path.insert(0, {root!r} + '/src')
    imported = importlib.import_module({task['module']!r})
    edited = types.ModuleType(imported.__name__)
    edited.__package__ = imported.__package__
    edited.__file__ = imported.__file__
    sys.modules[edited.__name__] = edited
    exec(compile({source!r}, edited.__file__, 'exec'), edited.__dict__)
    if {task['repo']!r} == 'semver':
        instance = edited.Version.parse(args[0])
        result = getattr(instance, {task['call']!r})(*args[1:])
        return str(result) if isinstance(result, edited.Version) else result
    return getattr(edited, {task['call']!r})(*args)
"""
