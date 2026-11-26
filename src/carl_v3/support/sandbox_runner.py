"""Docker/Podman-only execution, fail closed when runtime is unavailable.

The controller NEVER passes expected values, reference code, other tasks,
credentials, host paths, or scoring logic into a candidate container.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import time
import uuid

from .value_codec import decode


class InfrastructureError(RuntimeError):
    pass


@dataclass
class Execution:
    status: str
    value: object = None
    error: str = ""
    seconds: float = 0.0
    container: str | None = None
    executed: bool = False
    cleanup_verified: bool = False

    def to_dict(self):
        return asdict(self)


def _harness():
    # The candidate's stdout/stderr are redirected at fd level. Only the
    # dedicated result fd is used by the harness. No expected values are here.
    return Path(__file__).with_name("value_codec.py").read_text() + '''
import os, sys, traceback
out = os.fdopen(os.dup(1), "w", buffering=1)
null = os.open("/dev/null", os.O_WRONLY)
os.dup2(null, 1)
os.dup2(null, 2)
os.close(null)
request = json.loads(sys.stdin.read(65537))
write_result = out.write
json_dump = json.dumps
encode_value = encode
namespace = {"__name__": "candidate"}
try:
    exec(compile(request["code"], "candidate.py", "exec"), namespace)
    fn = namespace[request["function"]]
    result = fn(*decode(request["args"]), **decode(request["kwargs"]))
    record = {"status": "returned", "value": encode_value(result)}
except BaseException as error:
    record = {"status": "code_error", "error": type(error).__name__ + ": " + str(error)[:1200]}
write_result(json_dump(record, allow_nan=False) + "\\n")
out.flush()
'''


class SandboxRunner:
    def __init__(self, runtime=None, image=None, timeout=5.0, output_limit=65536):
        self.runtime = runtime or os.getenv("CODING_SANDBOX_RUNTIME") or shutil.which("docker") or shutil.which("podman")
        self.image = image or os.getenv("CODING_SANDBOX_IMAGE", "")
        self.timeout = timeout
        self.output_limit = output_limit
        self._ready = False
        self.last_cleanup = None
        self.run_label = os.getenv('CODING_SANDBOX_RUN', '')
        if self.run_label and not re.fullmatch(r'[0-9a-f]{32}', self.run_label):
            raise ValueError('invalid sandbox run identifier')

    def _argv(self, args):
        local = ['--remote=false'] if Path(self.runtime).name == 'podman' else []
        return [self.runtime, *local, *args]

    def _command(self, args, timeout=15):
        if not self.runtime or not shutil.which(self.runtime):
            raise InfrastructureError("Docker/Podman executable is unavailable")
        if Path(self.runtime).name not in ("docker", "podman"):
            raise InfrastructureError("only Docker/Podman runtimes are allowed")
        try:
            return subprocess.run(self._argv(args), capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise InfrastructureError(str(e)) from e

    def preflight(self):
        if self.runtime and Path(self.runtime).name == 'docker':
            endpoint = os.getenv('DOCKER_HOST') if not os.getenv('DOCKER_CONTEXT') else None
            if not endpoint:
                context = self._command(['context', 'inspect', '--format', '{{.Endpoints.docker.Host}}'])
                if context.returncode:
                    raise InfrastructureError('could not resolve the local Docker context')
                endpoint = context.stdout.strip()
            if not endpoint.startswith('unix:///'):
                raise InfrastructureError('Docker context must point to a local Unix socket')
        info = self._command(["info"])
        if info.returncode:
            raise InfrastructureError("container runtime denied/unavailable: " + info.stderr[-1500:])
        if not self.image or self.image.startswith('-') or any(ch.isspace() for ch in self.image):
            raise InfrastructureError('CODING_SANDBOX_IMAGE must name an approved local Python image')
        result = self._command(['image', 'inspect', self.image])
        if result.returncode:
            raise InfrastructureError("approved local image unavailable: " + result.stderr[-1500:])
        item = json.loads(result.stdout)[0]
        if item.get("Config", {}).get("Volumes"):
            raise InfrastructureError("image declares volumes; no volumes are permitted")
        self.image_id = item['Id']
        self._ready = True
        return {"runtime": self.runtime, "image": self.image, "image_id": item["Id"]}

    def _create_args(self, name):
        labels = ['--label', f'coding-agentic-rl.run={self.run_label}'] if self.run_label else []
        return ["create", "--name", name, "--label", "coding-agentic-rl=true", *labels,
                "--pull=never", "--network=none", "--ipc=none", "--read-only",
                "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
                "--user=65534:65534", "--cpus=1", "--memory=256m", "--memory-swap=256m",
                "--pids-limit=32", "--ulimit=nofile=64:64", "--ulimit=fsize=1048576:1048576",
                "--ulimit=core=0:0", "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m",
                "--log-driver=none", "--workdir=/tmp", "--interactive",
                "--entrypoint=python3", getattr(self, 'image_id', self.image), "-I", "-S", "-c", _harness()]

    def _cleanup(self, name):
        try:
            result = self._command(["rm", "--force", name])
            check = self._command(["ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"])
            ok = check.returncode == 0 and name not in check.stdout.splitlines()
            self.last_cleanup = {"container": name, "verified": ok, 'rm_exit_code': result.returncode}
            return ok
        except InfrastructureError as exc:
            self.last_cleanup = {"container": name, "verified": False, "error": str(exc)}
            return False

    def run(self, code, function, args, kwargs):
        start = time.monotonic()
        if not isinstance(code, str) or len(code.encode()) > 32768:
            return Execution("code_error", error="code exceeds 32768 bytes or is not a string")
        if not re.fullmatch(r"[A-Za-z_]\w*", function):
            raise ValueError("invalid frozen function name")
        # Validate controller-owned inputs; candidate strings are never exec'd here.
        decode(args)
        decode(kwargs)
        payload = json.dumps({"code": code, "function": function, "args": args, "kwargs": kwargs}, allow_nan=False).encode()
        if len(payload) > 65536:
            raise ValueError("input payload too large")
        name = "carl-" + uuid.uuid4().hex
        result = Execution("infra_error", container=name)
        creating = False
        proc = None
        try:
            if not self._ready:
                self.preflight()
            creating = True
            response = self._command(self._create_args(name))
            if response.returncode:
                raise InfrastructureError("create failed: " + response.stderr[-1500:])
            proc = subprocess.Popen(self._argv(['start', '--attach', '--interactive', name]),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    start_new_session=True)
            # Payload is at most 64 KiB; use a temporary writer thread so a
            # runtime that never consumes stdin cannot block the wall deadline.
            import threading
            def write_input():
                try:
                    proc.stdin.write(payload)
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            writer = threading.Thread(target=write_input, daemon=True)
            writer.start()
            chunks = {"stdout": bytearray(), "stderr": bytearray()}
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
            selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
            deadline = time.monotonic() + self.timeout
            stopped = None
            try:
                while selector.get_map():
                    if time.monotonic() >= deadline:
                        stopped = "timeout"
                        break
                    for key, _ in selector.select(min(0.05, max(0, deadline-time.monotonic()))):
                        data = os.read(key.fileobj.fileno(), 8192)
                        if not data:
                            selector.unregister(key.fileobj)
                        else:
                            chunks[key.data].extend(data)
                    if sum(map(len,chunks.values())) > self.output_limit:
                        stopped = "output_limit"
                        break
            finally:
                selector.close()
            state_response = self._command(["inspect", "--format", "{{json .State}}", name])
            if state_response.returncode:
                raise InfrastructureError("cannot inspect execution state")
            state = json.loads(state_response.stdout)
            result.executed = bool(state.get("StartedAt") and not state["StartedAt"].startswith("0001-"))
            if state.get("Error") or not result.executed:
                raise InfrastructureError("container did not start: " + str(state.get("Error", "")))
            if stopped:
                result.status, result.error = stopped, "candidate exceeded " + stopped
            else:
                proc.wait(timeout=2)
                if state.get("OOMKilled"):
                    result.status, result.error = "code_error", "candidate exceeded memory limit"
                else:
                    try:
                        data = json.loads(chunks["stdout"])
                        if data["status"] == "returned":
                            decode(data["value"])
                            result.status, result.value = "returned", data["value"]
                        elif data["status"] == "code_error":
                            result.status, result.error = "code_error", str(data.get("error", ""))[:1200]
                        else:
                            raise ValueError("unknown status")
                    except (ValueError, KeyError, TypeError, RecursionError):
                        result.status, result.error = "code_error", "no valid structured return value"
        except (InfrastructureError, OSError, subprocess.TimeoutExpired, ValueError, KeyError) as exc:
            result.status, result.error = "infra_error", str(exc)[:1500]
        finally:
            if creating:
                result.cleanup_verified = self._cleanup(name)
                if not result.cleanup_verified:
                    result.status, result.error = "infra_error", "container cleanup could not be verified"
            if proc is not None:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=3)
                writer.join(timeout=1)
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    if stream and not stream.closed:
                        stream.close()
            result.seconds = time.monotonic() - start
        return result
