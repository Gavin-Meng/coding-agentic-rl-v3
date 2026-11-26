"""Check trusted public references and authored regression fixtures on CPU."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from harbor.models.task.task import Task
import yaml

from carl_v3.common import ROOT, grade_program, read, sha
from carl_v3.prepare import tasks


def run_reference(root, task, path):
    # The two choices here are verified public source and our fixed mutation.
    program = grade_program(task, path.read_text(), root=str(root))
    program += """
import json
import sys
results = []
for args in json.load(sys.stdin):
    try:
        results.append({"value": __carl_grade__(*args)})
    except Exception as error:
        results.append({"error": type(error).__name__})
print(json.dumps(results))
"""
    result = subprocess.run([sys.executable, "-I", "-c", program],
                            input=json.dumps([args for args, _ in task["cases"]]), text=True,
                            capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"{task['id']} public fixture failed to run: {result.stderr[-1000:]}")
    return json.loads(result.stdout)


def main():
    cfg = read(ROOT / "configs/tasks.json")
    with TemporaryDirectory(prefix="carl-v3-trusted-fixtures-") as tmp:
        staged = Path(tmp)
        out = tasks(staged)
        train = {t["repo"] for t in cfg["tasks"] if t["split"] == "train"}
        evaluation = {t["repo"] for t in cfg["tasks"] if t["split"] == "eval"}
        assert not train & evaluation
        for task in cfg["tasks"]:
            locked = out[task["id"]]
            public, = (staged / "sources" / task["repo"] / "checkout").iterdir()
            if task["repo"] == "humanize":
                (public / "src/humanize/_version.py").write_text('__version__ = "4.12.3"\n')
            task_dir = staged / "tasks" / task["split"] / task["id"]
            candidate = task_dir / "environment/repo"
            target = locked["file"]
            expected = [{"value": v} for _, v in task["cases"]]
            assert run_reference(public, task, public / target) == expected, task["id"]
            assert run_reference(candidate, task, candidate / target) != expected, task["id"]
            assert all(sha(task_dir / path) == value for path, value in locked["protocol_files"].items())
            assert not (candidate / ".git").exists()
            assert list((task_dir / "tests").iterdir()) == [task_dir / "tests/test.sh"]
            parsed = Task(task_dir=task_dir)
            assert parsed.short_name == task["id"] and parsed.config.environment.workdir == "/work/repo"
            compose = yaml.safe_load((task_dir / "environment/docker-compose.yaml").read_text())
            sandbox = compose["services"]["main"]
            assert sandbox["network_mode"] == "none" and sandbox["read_only"] and sandbox["user"] == "1000:1000"
            assert sandbox["volumes"][1]["read_only"] and len(sandbox["volumes"]) == 2
        print(f"PASS: {len(out)} real repository references, failing mutations, and Harbor task schemas")


if __name__ == "__main__":
    main()
