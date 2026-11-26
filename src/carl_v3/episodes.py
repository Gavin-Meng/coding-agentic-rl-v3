"""Native Harbor sessions, exact-token checks, and one-shot paired assessment."""

import argparse
from collections import Counter
from copy import copy
import json
import math
import os
from pathlib import Path
import random
from statistics import fmean
from uuid import uuid4

from harbor_env.harness import HarborSessionFactory
import httpx

from .common import ASSETS, CONFIG, digest, grade_program, read, sha, write
from .support.sandbox_runner import SandboxRunner
from .support.scoring import IndependentScorer


def urls():
    sampler = read(CONFIG)["sampler"]
    return f"http://127.0.0.1:{sampler['openenv_port']}", f"http://127.0.0.1:{sampler['engine_port']}"


def session_args():
    cfg = read(CONFIG)
    env, engine = urls()
    return dict(server_url=env, llm_url=engine, model=cfg["model"]["repo"], provider="vllm",
                harness="pi", sandbox="docker", reward_key=cfg["rollout"]["reward_key"],
                agent_timeout_sec=cfg["rollout"]["agent_timeout_sec"],
                agent_step_limit=cfg["rollout"]["max_model_calls"], purpose="train",
                max_message_size_mb=128)


def validate_trace(trace, mirror):
    if not trace.turns:
        raise ValueError("No engine turns in the selected TrainingTrace")
    cfg = read(CONFIG)
    if len(trace.turns) > cfg["rollout"]["max_model_calls"]:
        raise ValueError("Trace exceeded the frozen model call limit")
    captured, used = 0, 0
    for turn in trace.turns:
        p, c = turn.prompt_token_ids, turn.completion_token_ids
        mask, logprobs = turn.loss_mask, turn.per_token_logps
        if mirror.get(turn.node_id) != {"prompt": p, "completion": c}:
            raise ValueError("Typed capture changed engine token IDs")
        if (len(c) != len(logprobs) or len(mask) != len(p) + len(c) or any(mask[:len(p)])
                or not all(math.isfinite(v) for v in logprobs)):
            raise ValueError("Capture mask and log-probability contract failed")
        if len(p) + len(c) > cfg["sampler"]["max_model_len"] or len(c) > cfg["rollout"]["max_completion_length"]:
            raise ValueError("Trace exceeded the frozen context or output budget")
        captured += len(c)
        used += sum(mask)
    if not used:
        raise ValueError("Trace has no model-generated supervision")
    return {"prompt_sha256": digest(trace.turns[0].prompt_token_ids),
            "generated": captured, "supervised": used, "model_calls": len(trace.turns)}


def capture_details(session, trace, policy):
    run = Path(os.environ["CARL_V3_RUN"])
    result = session.result
    if result.param_fixes:
        raise ValueError("The inference engine rejected and repaired sampling parameters")
    if result.exception_type not in (None, "AgentTimeoutError"):
        raise ValueError(f"Harbor trial failed: {result.exception_type}")
    mirror = read(run / "captures" / (result.session_id + ".engine.json"))
    details = validate_trace(trace, mirror)
    trial = run / "trials" / result.trial_name
    verifier = read(trial / "product-verifier.json")
    actual = session.verify([]).env_reward
    if actual != verifier["reward"]:
        raise ValueError("Independent verifier reward differs from the session reward")
    requests = [r for r in map(json.loads, (run / "timeline.jsonl").read_text().splitlines())
                if r["event"] == "model_request_finished" and r["capture"] == result.session_id]
    if not requests or not all(item.get("served") == policy for item in requests):
        raise ValueError("Capture is missing actual served policy evidence")
    atif = read(trial / "agent/trajectory.json")
    calls = [call for step in atif.get("steps", []) for call in (step.get("tool_calls") or [])]
    details.update(task=result.task_name, policy=policy, trial=result.trial_name, capture=result.session_id,
                   reward=actual, artifact_sha256=verifier["artifact_sha256"],
                   tool_calls=len(calls), tool_names=dict(Counter(call["function_name"] for call in calls)),
                   agent_timeout=result.exception_type == "AgentTimeoutError", seconds=result.wall_s,
                   budget_stop_count=result.budget_stop_count)
    return details


class AuditedFactory(HarborSessionFactory):
    """Keep the official session factory and bind actual traces to rollout IDs."""

    def __init__(self, *, sampling=None, **kwargs):
        super().__init__(sampling=sampling, **kwargs)
        self.group_policies = {}

    def create(self, task, seed=None, episode_id=None):
        policy = self.group_policies.get(seed, self.model)
        # The upstream worker's mutable factory.model is not a per-session version lock.
        owner = copy(self)
        owner.model = policy
        try:
            session = HarborSessionFactory.create(owner, task, seed=seed, episode_id=episode_id)
        finally:
            if owner._env is not None and owner._env is not self._env:
                owner._env.close()
        fetch = session.fetch_training_trace

        def audited_fetch():
            folder = Path(os.environ["CARL_V3_RUN"], "episodes", episode_id)
            try:
                trace = fetch()
                write(folder / "trace.json", trace.model_dump(mode="json"))
                details = capture_details(session, trace, policy)
            except Exception as exc:
                write(folder / "receipt.json", {"policy": policy, "reward": None,
                                               "error": f"{type(exc).__name__}: {exc}"[:700]})
                raise
            write(folder / "receipt.json", details)
            return trace

        session.fetch_training_trace = audited_fetch
        return session


def dataset_rows(factory):
    try:
        result = factory.prompt_rows()
        if not result:
            raise ValueError("The prepared split has no Harbor tasks")
        return result
    finally:
        if factory._env is not None:
            factory._env.close()
            factory._env = None


def episode(factory, row):
    episode_id = uuid4().hex
    run = Path(os.environ["CARL_V3_RUN"])
    session = None
    try:
        session = factory.create(row["prompt"], episode_id=episode_id)
        session.wait_for_completion()
        session.fetch_training_trace()
        return read(run / "episodes" / episode_id / "receipt.json")
    except Exception as exc:
        detail = {"task": row["task_name"], "reward": None, "policy": factory.model,
                  "error": f"{type(exc).__name__}: {exc}"[:700]}
        write(run / "episodes" / episode_id / "receipt.json", detail)
        return detail
    finally:
        if session is not None:
            session.close()


def sampling():
    cfg = read(CONFIG)["rollout"]
    return {key: cfg[key] for key in ("temperature", "top_p", "top_k", "min_p")}


def verifier_probe():
    task = read(ASSETS / "tasks.json")["semver-build"]
    source_dir, = (ASSETS / "sources/semver/checkout").iterdir()
    correct = (source_dir / task["file"]).read_text()
    broken = (ASSETS / "tasks/smoke/semver-build/environment/repo" / task["file"]).read_text()
    scorer = IndependentScorer(SandboxRunner(runtime="docker", image=task["image_id"], timeout=4.0))
    results = {key: scorer.score(task, grade_program(task, code)).to_dict()
               for key, code in {"public_reference": correct, "regression": broken, "print_pass": "print('PASS')"}.items()}
    write(Path(os.environ["CARL_V3_RUN"], "verifier-probe.json"), results)
    if [value["reward"] for value in results.values()] != [1.0, 0.0, 0.0]:
        raise ValueError("Independent verifier did not distinguish genuine output from wrong code")


def fixed_probe(policy, ids):
    _, engine = urls()
    wire = {"model": policy, "prompt": ids, "max_tokens": 1, "temperature": sampling()["temperature"],
            "top_p": 1.0, "top_k": -1, "seed": 3127, "prompt_logprobs": 1,
            "return_token_ids": True}
    raw = httpx.post(engine + "/v1/completions", json=wire, timeout=120, trust_env=False)
    raw.raise_for_status()
    value = raw.json()
    choice = value["choices"][0]
    prompt_ids = choice.get("prompt_token_ids") or value.get("prompt_token_ids")
    if prompt_ids != ids or value["model"] != policy:
        raise ValueError("Probe engine prefix or policy mismatch")
    probabilities = [scores[str(token)]["logprob"] for token, scores in
                     zip(ids[1:], choice["prompt_logprobs"][1:], strict=True)]
    if not all(math.isfinite(v) for v in probabilities):
        raise ValueError("Probe probabilities are not finite")
    return {"policy": policy, "ids_sha256": digest(ids), "logprobs": probabilities, "engine_id": value["id"]}


def load_adapter(path):
    _, engine = urls()
    raw = httpx.post(engine + "/v1/load_lora_adapter",
                    json={"lora_name": "carl-eval", "lora_path": str(path.resolve())},
                    timeout=120, trust_env=False)
    raw.raise_for_status()
    models = httpx.get(engine + "/v1/models", timeout=20, trust_env=False)
    models.raise_for_status()
    item, = [item for item in models.json()["data"] if item["id"] == "carl-eval"]
    if Path(item["root"]).resolve() != path.resolve():
        raise ValueError("vLLM advertises a different checkpoint source")
    return {"policy": item["id"], "adapter_sha256": sha(path / "adapter_model.safetensors")}


def paired_report(pairs):
    eligible = [p for p in pairs if all(p[arm]["reward"] in (0, 1) for arm in ("base", "adapter"))
                and p["base"]["prompt_sha256"] == p["adapter"]["prompt_sha256"]]
    changes = [p["adapter"]["reward"] - p["base"]["reward"] for p in eligible]
    cfg = read(CONFIG)["evaluation"]
    rng = random.Random(cfg["bootstrap_seed"])
    n = cfg["bootstrap_draws"]
    values = sorted(fmean(rng.choices(changes, k=len(changes))) for _ in range(n)) if changes else []
    return {"pairs_attempted": len(pairs), "paired_n": len(eligible),
            "infrastructure_or_prompt_mismatch": len(pairs) - len(eligible),
            "base_pass": sum(p["base"]["reward"] for p in eligible),
            "adapter_pass": sum(p["adapter"]["reward"] for p in eligible),
            "rescues": changes.count(1), "regressions": changes.count(-1),
            "delta": fmean(changes) if changes else None,
            "paired_ci95": [values[int(0.025*n)], values[int(0.975*n)-1]] if values else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "followup", "evaluate", "verifier-probe"))
    parser.add_argument("--policy")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    run = Path(os.environ["CARL_V3_RUN"])
    if args.mode == "verifier-probe":
        verifier_probe()
        return
    base = AuditedFactory(sampling=sampling(), **session_args())
    tasks = dataset_rows(base)
    if args.mode == "smoke":
        receipt = episode(base, tasks[0])
        write(run / "smoke.json", receipt)
        if receipt["reward"] is None or not receipt.get("tool_calls"):
            raise RuntimeError("Smoke did not yield a score, a typed trace and actual native tools")
    elif args.mode == "followup":
        if not args.policy or args.checkpoint is None:
            parser.error("followup requires a published --policy and --checkpoint")
        base.model = args.policy
        receipt = episode(base, tasks[0])
        if receipt["reward"] is None or receipt["policy"] != args.policy:
            raise RuntimeError("Novel rollout did not use the newly published policy")
        reloaded = load_adapter(args.checkpoint)
        base.model = reloaded["policy"]
        verify = episode(base, tasks[0])
        if verify["reward"] is None or verify["policy"] != reloaded["policy"]:
            raise RuntimeError("Reloaded checkpoint could not serve a scored Pi episode")
        write(run / "followup.json", {"published": receipt, "checkpoint_reload": verify, **reloaded})
    else:
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        reload_receipt = load_adapter(args.checkpoint)
        pairs = []
        plan = [{"task": task["task_name"], "order": ["base", "adapter"] if i % 2 == 0 else ["adapter", "base"]}
                for i, task in enumerate(tasks)]
        write(run / "eval-plan.json", {"tasks": plan, "adapter": reload_receipt, "evaluation": read(CONFIG)["evaluation"]})
        for task, pair in zip(tasks, plan, strict=True):
            row = {"task": pair["task"]}
            for arm in pair["order"]:
                base.model = session_args()["model"] if arm == "base" else reload_receipt["policy"]
                row[arm] = episode(base, task)
            pairs.append(row)
            write(run / "evaluation-partial.json", pairs)
        write(run / "evaluation.json", {"summary": paired_report(pairs), "pairs": pairs, "adapter": reload_receipt})


if __name__ == "__main__":
    main()
