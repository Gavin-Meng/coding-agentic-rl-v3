"""Stage hashed public source revisions, mutate reviewed fixtures, and pin the images."""

import argparse
import ast
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

from .common import ASSETS, CONFIG, ROOT, digest, read, sha, write
from .support.value_codec import encode

CONTEXT = """services:
  main:
    image: IMAGE
    pull_policy: never
    labels:
      carl-v3.run: "${CARL_V3_RUN_LABEL:?set by trusted controller}"
    command: ["node", "/usr/local/lib/carl-relay.js"]
    network_mode: none
    user: "1000:1000"
    working_dir: /work/repo
    read_only: true
    cap_drop: [ALL]
    security_opt: [no-new-privileges:true]
    cpus: 1
    mem_limit: 1536m
    memswap_limit: 1536m
    pids_limit: 128
    logging:
      driver: none
    tmpfs:
      - /tmp:exec,uid=1000,gid=1000,size=64m
      - /home/agent:uid=1000,gid=1000,size=64m
      - /logs:uid=1000,gid=1000,size=64m
    volumes:
      - work:/work
      - type: bind
        source: "${CARL_V3_CAPTURE_SOCKET:?set by trusted OpenEnv process}"
        target: /run/carl/capture.sock
        read_only: true
        bind:
          create_host_path: false
volumes:
  work:
    labels:
      carl-v3.run: "${CARL_V3_RUN_LABEL:?set by trusted controller}"
"""


def mutate(source, target, statement):
    members = target.split(".")
    tree = ast.parse(source)
    for name in members:
        tree, = [node for node in tree.body if getattr(node, "name", None) == name]
    body = tree.body[1:] if ast.get_docstring(tree, clean=False) is not None else tree.body
    first, last = body[0], body[-1]
    lines = source.splitlines(keepends=True)
    lines[first.lineno - 1:last.end_lineno] = [" " * first.col_offset + statement + "\n"]
    changed = "".join(lines)
    ast.parse(changed)
    return changed


def source_tree(item, root):
    archive = root / "source.tgz"
    root.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        url = f"https://codeload.github.com/{item['repo']}/tar.gz/{item['commit']}"
        with urllib.request.urlopen(url, timeout=30) as response, archive.open("wb") as target:
            target.write(response.read(20_000_001))
    if sha(archive) != item["archive_sha256"]:
        raise ValueError(f"Public source digest mismatch for {item['repo']}")
    content = root / "checkout"
    if not content.is_dir():
        content.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(content, filter="data")
    trees = list(content.iterdir())
    source, = trees
    return source


def tasks(out):
    spec = read(ROOT / "configs/tasks.json")
    packed = {}
    for task in spec["tasks"]:
        source = source_tree(spec["sources"][task["repo"]], out / "sources" / task["repo"])
        task_dir = out / "tasks" / task["split"] / task["id"]
        env = task_dir / "environment"
        frozen = env / "repo"
        if frozen.exists():
            raise FileExistsError("Use an empty asset build directory")
        shutil.copytree(source / "src", frozen / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        metadata = spec["sources"][task["repo"]]
        for field in ("readme", "license"):
            shutil.copyfile(source / metadata[field], frozen / metadata[field])
        if task["repo"] == "humanize":
            (frozen / "src/humanize/_version.py").write_text('__version__ = "4.12.3"\n')
        module_file = "src/" + task["module"].replace(".", "/") + ".py"
        path = frozen / module_file
        path.write_text(mutate(path.read_text(), task["replace"], task["broken"]))
        image = f"carl-v3-task:{task['id']}-{digest(task)[:12]}"
        (env / "Dockerfile").write_text(
            "FROM carl-v3-runtime:pi-1.5.2\nCOPY --chown=1000:1000 repo /work/repo\n"
        )
        # HOME is a tmpfs at runtime; the agent's setup populates the NVM no-op.
        (env / "docker-compose.yaml").write_text(CONTEXT.replace("IMAGE", image))
        args, expected = task["cases"][0]
        instruction = (
            f"{task['instruction']}\n\n"
            f"Edit /work/repo/{module_file}. The verifier checks the final content of "
            "that file against the Python API. Other edits will not be graded. "
            f"Example input for {task['call']}: {args!r}. Expected: {expected!r}. "
            "Local source is in /work/repo/src; set PYTHONPATH=/work/repo/src for checks.\n"
        )
        (task_dir / "instruction.md").write_text(instruction)
        (task_dir / "task.toml").write_text(
            'schema_version = "1.3"\n'
            f'[task]\nname = "carl-v3/{task["id"]}"\ndescription = "Public repository regression"\n'
            '[agent]\ntimeout_sec = 180.0\n'
            '[environment]\nnetwork_mode = "no-network"\nworkdir = "/work/repo"\n'
            f'docker_image = "{image}"\ncpus = 1\nmemory_mb = 1536\nstorage_mb = 1024\n'
            '[verifier]\ntimeout_sec = 75.0\n'
        )
        # Harbor requires a verifier entry file during discovery. An accidental
        # invocation of the default verifier must fail closed.
        tests = task_dir / "tests"
        tests.mkdir()
        (tests / "test.sh").write_text(
            '#!/bin/sh\nprintf "%s\\n" "Use the independent V3 verifier" >&2\nexit 2\n'
        )
        if len(path.read_bytes()) >= 28000:
            raise ValueError("Initial product exceeds the independent verifier limit")
        packed[task["id"]] = {
            "split": task["split"], "repo": task["repo"], "repo_commit": metadata["commit"],
            "module": task["module"], "call": task["call"], "file": module_file, "image": image,
            "initial_sha256": sha(path),
            "protocol_files": {name: sha(task_dir / name) for name in
                               ("task.toml", "instruction.md", "environment/docker-compose.yaml", "tests/test.sh")},
            "function": "__carl_grade__",
            "score_cases": [{"case_id": str(i), "args": encode(tuple(args)), "kwargs": encode({}),
                             "expected": encode(value)} for i, (args, value) in enumerate(task["cases"])],
        }
    write(out / "tasks.json", packed)
    return packed


def images(out):
    docker = ["docker"]
    subprocess.run(docker + ["build", "--tag", "carl-v3-runtime:pi-1.5.2", str(ROOT / "images")], check=True)
    packed = read(out / "tasks.json")
    for task_id, task in sorted(packed.items()):
        environment = out / "tasks" / task["split"] / task_id / "environment"
        subprocess.run(docker + ["build", "--network", "none", "-t", task["image"], str(environment)], check=True)
        task["image_id"] = subprocess.check_output(docker + ["image", "inspect", "--format", "{{.Id}}", task["image"]],
                                                   text=True).strip()
    write(out / "tasks.json", packed)


def model(out):
    from huggingface_hub import snapshot_download

    cfg = read(CONFIG)["model"]
    if shutil.disk_usage(out).free < read(CONFIG)["hardware"]["minimum_assets_disk_gib"] * 2**30:
        if not (out / "model.json").is_file():
            raise RuntimeError("Insufficient free disk for the BF16 model and checkpoints")
    path = Path(snapshot_download(
        cfg["repo"], revision=cfg["revision"], cache_dir=out / "hf",
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "merges.txt", "vocab.*", "tokenizer.*"],
        token=False,
    ))
    config = read(path / "config.json")
    if config["architectures"] != [cfg["architecture"]]:
        raise ValueError("Wrong model architecture")
    weights = sorted(set(read(path / "model.safetensors.index.json")["weight_map"].values()))
    sizes = {name: (path / name).stat().st_size for name in weights}
    if path.name != cfg["revision"] or len(sizes) != 18 or sum(sizes.values()) < cfg["safetensors_bytes"]:
        raise ValueError("Incomplete or wrong model snapshot")
    write(out / "model.json", {"repo": cfg["repo"], "revision": cfg["revision"], "path": str(path),
                               "config_sha256": sha(path / "config.json"),
                               "chat_template_sha256": sha(path / "chat_template.jinja"),
                               "weight_bytes": sizes})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=ASSETS)
    cli.add_argument("--tasks-only", action="store_true", help="stage public sources and cases without model or Docker")
    ns = cli.parse_args()
    out = ns.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "tasks.json").is_file():
        tasks(out)
    if not ns.tasks_only:
        images(out)
        model(out)
    print(f"Public task assets: {out}")


if __name__ == "__main__":
    main()
