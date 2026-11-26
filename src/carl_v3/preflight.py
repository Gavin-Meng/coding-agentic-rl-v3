"""Read-only checks for the authorized A800 machine."""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from urllib.request import urlopen

from .common import ASSETS, CONFIG, ROOT, read


def cmd(*args):
    return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True, timeout=20).strip()


def local_docker():
    if os.environ.get("DOCKER_CONTEXT"):
        endpoint = cmd("docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}")
    else:
        endpoint = os.environ.get("DOCKER_HOST") or cmd(
            "docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}")
    if not endpoint.startswith("unix:///"):
        raise ValueError("Docker must use a local Unix socket")
    cmd("docker", "info", "--format", "{{.ServerVersion}}")
    cmd("docker", "compose", "version")
    return endpoint


def gpu_rows():
    fields = "uuid,name,memory.total,memory.used,driver_version,mig.mode.current,compute_cap"
    data = cmd("nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits")
    return {parts[0]: parts[1:] for line in data.splitlines()
            if len(parts := [field.strip() for field in line.split(",")]) == 7}


def require_gpu_topology(check_idle=True):
    hw = read(CONFIG)["hardware"]
    keys = [os.getenv(name, "") for name in
            ("CARL_SAMPLER_GPU", "CARL_LEARNER_GPU0", "CARL_LEARNER_GPU1")]
    if os.getenv("CARL_GPU_AUTHORIZED") != "1" or len(set(keys)) != 3 or not all(
        re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", key) for key in keys
    ):
        raise ValueError("Set three distinct authorized GPU UUIDs and CARL_GPU_AUTHORIZED=1")
    rows = gpu_rows()
    busy = {line.strip() for line in
            cmd("nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader").splitlines()}
    for key in keys:
        if key not in rows:
            raise ValueError(f"Authorized GPU {key} is absent on this host")
        name, total, used, driver, mig, cc = rows[key]
        if hw["gpu_name_fragment"] not in name or int(total) < hw["minimum_gpu_mib"] or cc != "8.0":
            raise ValueError(f"{key}: needs a full A800 80 GB with compute capability 8.0")
        if mig.lower() != "disabled":
            raise ValueError(f"{key}: MIG must be disabled")
        if int(driver.split(".",1)[0]) < hw["minimum_driver_major"]:
            raise ValueError(f"{key}: CUDA 13.0 profile needs NVIDIA R580 or newer")
        if check_idle and (key in busy or int(used) > hw["maximum_idle_used_mib"]):
            raise ValueError(f"{key}: GPU is in use ({used} MiB)")
    return keys


def inspect():
    for args in (
        ("nvidia-smi",), ("nvidia-smi", "-L"), ("nvidia-smi", "topo", "-m"),
        ("free", "-h"), ("df", "-h", str(ROOT)), ("docker", "version"), ("docker", "compose", "version"),
    ):
        print("$", *args, flush=True)
        try:
            print(cmd(*args), flush=True)
        except (OSError, subprocess.SubprocessError) as error:
            print(getattr(error, "output", None) or str(error), flush=True)


def runtime(expected_devices=1):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != expected_devices:
        raise RuntimeError("Role process does not enumerate its authorized GPUs")
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if rank >= expected_devices:
        raise RuntimeError("LOCAL_RANK does not map to the allocated CUDA devices")
    torch.cuda.set_device(rank)
    device = torch.cuda.get_device_properties(rank)
    if (device.major, device.minor) != (8, 0) or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native SM80 BF16 required")
    if torch.version.cuda != read(CONFIG)["upstream"]["cuda"]:
        raise RuntimeError("PyTorch uses a different CUDA runtime than the lock")
    data = {"name": device.name, "sm": [device.major, device.minor], "bf16": True,
            "cuda": torch.version.cuda, "torch": torch.__version__, "nccl": torch.cuda.nccl.version(),
            "visible_devices": expected_devices, "local_rank": rank}
    print(json.dumps(data))
    return data


def installations():
    roles = {"control": ("openenv", "harbor"), "learner": ("torch", "trl", "transformers", "peft", "accelerate"),
             "sampler": ("torch", "vllm", "transformers")}
    versions = read(CONFIG)["upstream"]
    for role, packages in roles.items():
        python = ROOT / ".venvs" / role / "bin/python"
        if not python.is_file():
            raise ValueError(f"Run setup.sh to install the {role} lock")
        actual = json.loads(cmd(str(python), "-c",
            "import importlib.metadata as m, json, sys; print(json.dumps({p:m.version(p) for p in sys.argv[1:]}))",
            *packages))
        if any(actual[p] != versions[p] for p in packages):
            raise ValueError(f"Installed {role} packages differ from the lock: {actual}")
        print(role, json.dumps(actual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--skip-hub-probe", action="store_true")
    args = parser.parse_args()
    if args.runtime:
        runtime()
        return 0
    inspect()
    if args.inspect_only:
        return 0
    failures = []
    for check in (require_gpu_topology, local_docker, installations if args.installed else None):
        if check is None:
            continue
        try:
            print(check.__name__, check())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            failures.append(f"{check.__name__}: {getattr(exc, 'output', None) or str(exc)}")
    cfg = read(CONFIG)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        failures.append("The locked wheels target Linux x86_64")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 35):
        failures.append("The release requires glibc 2.35 or newer")
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < cfg["hardware"]["minimum_ram_gib"] * 2**30:
        failures.append("Available RAM below the pilot's 96 GiB staging allowance")
    minimum_disk = 20 if (ASSETS / "model.json").is_file() else cfg["hardware"]["minimum_assets_disk_gib"]
    if shutil.disk_usage(ROOT).free < minimum_disk * 2**30:
        failures.append(f"Free disk below {minimum_disk} GiB")
    if not args.skip_hub_probe:
        model = cfg["model"]
        try:
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/config.json"
            with urlopen(url, timeout=15) as response:
                config = json.load(response)
            if config["architectures"] != [model["architecture"]]:
                raise ValueError("Architecture differs from the locked snapshot")
            print("Public Hugging Face revision metadata is accessible")
        except (OSError, ValueError, KeyError) as error:
            failures.append(f"Hugging Face revision probe: {error}")
    for failure in failures:
        print("BLOCKED", failure, file=sys.stderr)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
