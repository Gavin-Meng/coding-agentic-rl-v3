"""Run the upstream OpenEnv Capture Proxy and native Harbor Pi with local isolation."""

import asyncio
import contextvars
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import tarfile
import time

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier
from openenv.core.harness.capture.server import extract_api_key
from openenv.core.harness.capture.upstream import InferenceClient
from openenv.core.harness.capture.validate_llm import validate_llm
from openenv.harbor import rollout, serving
from openenv.harbor.install_fixes import InterceptPi, build_models_json
import uvicorn

from .common import ASSETS, CONFIG, event, grade_program, read, sha, write
from .support.sandbox_runner import SandboxRunner
from .support.scoring import IndependentScorer

CONFIGURED = read(CONFIG)
CONTAINER_UID = "1000:1000"
RELAY_ROUTE = "/capture/v1/chat/completions"
CURRENT_SESSION = contextvars.ContextVar("carl_v3_capture", default=None)


async def docker(*args, timeout=15):
    child = await asyncio.create_subprocess_exec("docker", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(child.communicate(), timeout)
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
    if child.returncode:
        raise RuntimeError(err.decode(errors="replace")[-500:])
    return out


class OfflinePi(InterceptPi):
    async def install(self, environment):
        version = CONFIGURED["upstream"]["pi"]
        command = ("mkdir -p ~/.nvm && touch ~/.nvm/nvm.sh && "
                   f'test "$(pi --version)" = {shlex.quote(version)}')
        result = await self.exec_as_agent(environment, command=command)
        if result.return_code != 0:
            raise RuntimeError("Preinstalled native Pi version failed its sandbox probe")

    async def setup(self, environment):
        await super().setup(environment)
        spec = self._intercept_config
        document = json.loads(build_models_json(spec["base_url"], spec["api_key"], spec["model"]))
        for provider in document["providers"].values():
            for model in provider["models"]:
                model["contextWindow"] = CONFIGURED["sampler"]["max_model_len"]
                model["maxTokens"] = CONFIGURED["rollout"]["max_completion_length"]
                model["reasoning"] = False
        command = f"printf %s {shlex.quote(json.dumps(document))} > ~/.pi/agent/models.json"
        result = await self.exec_as_agent(environment, command=command)
        if result.return_code != 0:
            raise RuntimeError("Could not set Pi's private capture endpoint")


class PiDocker(DockerEnvironment):
    """Retain Harbor's agent lifecycle with the mounted capture socket and no IP egress."""

    def __init__(self, *args, **kwargs):
        kwargs["mounts"] = []
        kwargs["auto_mounts"] = []
        kwargs["enable_environment_dir_upload"] = False
        super().__init__(*args, **kwargs)

    @staticmethod
    def _requires_egress_control(*, startup_network_policy, phase_network_policies):
        for policy in (startup_network_policy, *phase_network_policies):
            if policy.network_mode != NetworkMode.NO_NETWORK or policy.allowed_hosts:
                raise ValueError("All sandbox phases must be offline")
        return False

    async def exec(self, command, *, cwd=None, env=None, timeout_sec=None, user=None, **kwargs):
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec,
                                  user=CONTAINER_UID, **kwargs)

    @property
    def capabilities(self):
        return super().capabilities.model_copy(update={"mounted": False, "disable_internet": True})

    async def start(self, force_build):
        if force_build or not self.task_env_config.docker_image:
            raise ValueError("A task must use its prepared local image")
        await super().start(force_build=False)
        self.container_id = await self._platform._resolve_service_container("main")
        info, = json.loads(await docker("inspect", self.container_id))
        config, host = info["Config"], info["HostConfig"]
        src_socket = os.environ["CARL_V3_CAPTURE_SOCKET"]
        binds = [m for m in info["Mounts"] if m["Type"] == "bind"]
        if (len(binds) != 1 or binds[0]["Source"] != src_socket or binds[0]["RW"]
                or binds[0]["Destination"] != "/run/carl/capture.sock"):
            raise ValueError("Unexpected sandbox host mounts")
        for mount in info["Mounts"]:
            if mount["Type"] == "bind":
                continue
            if (mount["Type"], mount["Destination"]) not in {
                ("volume", "/work"), ("tmpfs", "/logs"), ("tmpfs", "/home/agent"), ("tmpfs", "/tmp"),
            }:
                raise ValueError("Unexpected sandbox storage")
        if (host["NetworkMode"] != "none" or host["Privileged"] or host["DeviceRequests"]
                or host["Devices"] or host["CapAdd"] or "ALL" not in (host["CapDrop"] or [])
                or not host["ReadonlyRootfs"] or config["User"] != CONTAINER_UID
                or not any("no-new-privileges" in option for option in host["SecurityOpt"] or [])
                or not 0 < host["Memory"] <= 1536 * 2**20 or not 0 < host["PidsLimit"] <= 128
                or not 0 < host["NanoCpus"] <= 10**9):
            raise ValueError("Sandbox isolation or resource limits failed inspection")
        images = {item["image"]: item["image_id"] for item in read(ASSETS / "tasks.json").values()}
        if info["Image"] != images[self.task_env_config.docker_image]:
            raise ValueError("Sandbox image differs from the staged task")
        await docker("exec", "--user", CONTAINER_UID, self.container_id, "node", "-e",
                     "const s=require('node:net').connect(18808,'127.0.0.1');"
                     "s.once('connect',()=>s.end());s.once('error',()=>process.exit(1));")
        event("sandbox_attested", container=info["Id"], image=info["Image"], network="none", user=config["User"],
              workspace="ephemeral-volume", capture_mount="single-readonly-uds")


async def read_artifact(container, path):
    """Freeze and read only the declared source file, without extracting any archive to host disk."""
    await docker("pause", container, timeout=8)
    try:
        child = await asyncio.create_subprocess_exec("docker", "cp", f"{container}:{path}", "-",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            async with asyncio.timeout(12):
                try:
                    data = await child.stdout.readexactly(65537)
                    if len(data) > 65536:
                        return None, "artifact_oversize"
                except asyncio.IncompleteReadError as exc:
                    data = exc.partial
                errors = (await child.stderr.read(4096)).decode(errors="replace")
                if await child.wait():
                    if "Could not find the file" in errors or "no such file or directory" in errors:
                        return None, "artifact_missing"
                    raise RuntimeError("Could not freeze task artifact")
        finally:
            if child.returncode is None:
                child.kill()
                await child.communicate()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
            items = tar.getmembers()
            if (len(items) != 1 or items[0].name != Path(path).name
                    or not items[0].isfile() or not 0 < items[0].size <= 28000):
                return None, "artifact_invalid"
            try:
                return tar.extractfile(items[0]).read().decode(), "artifact_ok"
            except UnicodeDecodeError:
                return None, "artifact_encoding"
    finally:
        await docker("unpause", container, timeout=8)


class RepoVerifier(BaseVerifier):
    async def verify(self):
        begin = time.monotonic()
        folder = Path(self.trial_paths.trial_dir)
        receipt = {"reward": None, "reason": "infrastructure", "artifact_sha256": None}
        try:
            task = read(ASSETS / "tasks.json")[self.task.short_name]
            artifact, reason = await read_artifact(self.environment.container_id, "/work/repo/" + task["file"])
            if artifact is None:
                receipt.update(reward=0.0, reason=reason)
            else:
                path = folder / "final-artifact.py"
                path.write_bytes(artifact.encode())
                receipt["artifact_sha256"] = sha(path)
                runner = SandboxRunner(runtime="docker", image=task["image_id"], timeout=4.0)
                score = await asyncio.to_thread(IndependentScorer(runner).score, task, grade_program(task, artifact))
                receipt.update(score.to_dict())
        except (AttributeError, OSError, RuntimeError, TimeoutError, ValueError, KeyError, tarfile.TarError) as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"[:500]
        receipt["seconds"] = time.monotonic() - begin
        write(folder / "product-verifier.json", receipt)
        event("verifier_finished", trial=folder.name, reward=receipt["reward"], seconds=receipt["seconds"])
        reward = receipt["reward"]
        return VerifierResult(rewards=None if reward is None else {"reward": reward})


def build(run, mode):
    cfg = CONFIGURED
    model = cfg["model"]["repo"]
    upstream = f"http://127.0.0.1:{cfg['sampler']['engine_port']}"
    probe = validate_llm(upstream, model, provider="vllm", check_tools=True, timeout=60)
    write(run / "capture-preflight.json", asdict(probe))
    if (not probe.ok or not probe.trainable or probe.logprobs_mode != "processed"
            or probe.tool_support != "ok" or probe.param_fixes):
        raise RuntimeError("OpenEnv vLLM tool and exact-token preflight failed")
    split = {"smoke": "smoke", "train": "train", "evaluate": "eval"}[mode]
    allowed = {name: task for name, task in read(ASSETS / "tasks.json").items() if task["split"] == split}
    datasets = [str(ASSETS / "tasks" / split)]
    configured = rollout.build_trial_config
    export = rollout.export_session
    post = InferenceClient._post
    assignments = {}

    def trial(**kwargs):
        path = Path(kwargs["task_dir"]).resolve()
        policy = kwargs["model"]
        if (kwargs["harness"] != "pi" or kwargs["sandbox"] != "docker"
                or path.parent != Path(datasets[0]) or path.name not in allowed
                or kwargs["force_build"] or kwargs["keep_sandbox"]
                or not (policy == model or policy == "carl-eval" or re.fullmatch(r"trl-policy-v[0-9]+", policy))):
            raise ValueError("Rejected a trial outside the frozen run protocol")
        task = allowed[path.name]
        if any(sha(path / name) != value for name, value in task["protocol_files"].items()):
            raise ValueError("Frozen task files changed")
        config = configured(**kwargs)
        config.agent.name = None
        config.agent.import_path = "carl_v3.actor:OfflinePi"
        config.agent.kwargs.update(version=cfg["upstream"]["pi"], reasoning_effort="off")
        config.environment.import_path = "carl_v3.actor:PiDocker"
        config.verifier.import_path = "carl_v3.actor:RepoVerifier"
        session = kwargs["session_id"]
        seed_key = path.name if mode == "evaluate" else f"{path.name}:{session}"
        seed_base = cfg["evaluation"]["seed"] if mode == "evaluate" else cfg["rollout"]["seed"]
        seed = int.from_bytes(hashlib.sha256(f"{seed_base}:{seed_key}".encode()).digest()[:4], "big")
        assignments[session] = {"policy": policy, "task": path.name, "trial": kwargs["trial_name"],
                                "requests": 0, "seed": seed}
        return config

    def capture(session, **kwargs):
        document = export(session, **kwargs)
        mirror = {node.node_id: {"prompt": node.prompt_ids, "completion": node.sampled_ids}
                  for node in session.graph.nodes()}
        write(run / "captures" / (session.session_id + ".engine.json"), mirror)
        return document

    async def measured(self, route, body, *args, **kwargs):
        ident = CURRENT_SESSION.get()
        if route != "/v1/chat/completions" or ident is None:
            return await post(self, route, body, *args, **kwargs)
        record = assignments[ident]
        index = record["requests"]
        record["requests"] += 1
        seed = (record["seed"] + index) % 2**32
        body = {**body, "seed": seed, "chat_template_kwargs": {"enable_thinking": False}}
        if body["model"] != record["policy"]:
            raise ValueError("OpenEnv request policy differs from the trial")
        marker = dict(trial=record["trial"], task=record["task"], capture=ident, index=index, seed=seed,
                      policy=record["policy"])
        marker["sampling"] = {key: body[key] for key in
                              ("temperature", "top_p", "top_k", "min_p", "max_tokens") if key in body}
        event("model_request_started", **marker)
        try:
            response = await post(self, route, body, *args, **kwargs)
            if response["model"] != record["policy"]:
                raise ValueError("vLLM served a different requested policy")
        except BaseException as error:
            event("model_request_finished", error=type(error).__name__, **marker)
            raise
        event("model_request_finished", engine_id=response.get("id"), served=response["model"],
              usage=response.get("usage"), **marker)
        return response

    InferenceClient._post = measured
    rollout.build_trial_config = trial
    rollout.export_session = capture
    service = serving.HarborService(llm_url=upstream, model=model, datasets=datasets, provider="vllm",
                                   expose="direct", capture_level="tokens",
                                   max_output_tokens=cfg["rollout"]["max_completion_length"])
    service.mounted = True
    service.public_url = f"http://127.0.0.1:{cfg['sampler']['sandbox_relay_port']}/capture"
    serving.HarborService.set_current(service)
    app = serving.build_app(datasets=datasets, llm_url=upstream, model=model,
                            llm={**asdict(probe), "rollout_type": probe.rollout_type})

    async def sandbox_only(scope, receive, send):
        if scope["type"] == "http":
            from starlette.datastructures import Headers
            from starlette.responses import Response

            session = extract_api_key(dict(Headers(scope=scope)))
            if (scope["method"] != "POST" or scope["path"] != RELAY_ROUTE
                    or session not in assignments):
                await Response(status_code=403)(scope, receive, send)
                return
            token = CURRENT_SESSION.set(session)
            try:
                await app(scope, receive, send)
            finally:
                CURRENT_SESSION.reset(token)
    return app, sandbox_only


async def serve(run, mode):
    app, sandbox = build(run, mode)
    cfg = CONFIGURED
    loopback = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=cfg["sampler"]["openenv_port"],
                                            timeout_graceful_shutdown=15))
    conduit = uvicorn.Server(uvicorn.Config(sandbox, uds=os.environ["CARL_V3_CAPTURE_SOCKET"],
                                            lifespan="off", ws="none", limit_concurrency=16,
                                            timeout_graceful_shutdown=15))
    with loopback.capture_signals():
        control = asyncio.create_task(loopback._serve())
        data = asyncio.create_task(conduit._serve())
        done, _ = await asyncio.wait({control, data}, return_when=asyncio.FIRST_COMPLETED)
        loopback.should_exit = conduit.should_exit = True
        await asyncio.gather(*done)
        await asyncio.gather(control, data)


if __name__ == "__main__":
    import sys
    asyncio.run(serve(Path(os.environ["CARL_V3_RUN"]), sys.argv[1]))
