"""Bounded native-Pi pilot around the official TRL HarnessAsyncGRPOTrainer."""

import argparse
from datetime import timedelta
from functools import partial
import hashlib
import math
import os
from pathlib import Path
from queue import Empty
import shutil
import time

from accelerate import PartialState
from datasets import Dataset
import httpx
from peft import LoraConfig
import torch
from torch.distributed import all_gather_object, broadcast_object_list, get_rank, get_world_size
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer, TrainerCallback
from trl import AsyncGRPOConfig
from trl.experimental.async_grpo.openenv_harness import HarnessAsyncGRPOTrainer, _HarnessRolloutLoop
from trl.trainer.async_grpo_trainer import RolloutSample

from .common import CONFIG, digest, event, model_path, read, sha, write
from .episodes import AuditedFactory, dataset_rows, fixed_probe, session_args, urls
from .preflight import runtime
from .tree_backend import select_sm80_backend


def correctness(outcome):
    reward = outcome.env_reward
    if reward is None or reward in (0, 1):
        return reward
    raise ValueError("The verifier produced a value outside the frozen binary contract")


class AuditedLoop(_HarnessRolloutLoop):
    def _fits_token_allowance(self):
        return (getattr(self, "started", 0) < int(os.environ["CARL_V3_MAX_GROUPS"])
                and time.time() < float(os.environ["CARL_V3_DEADLINE"])
                and not Path(os.environ["CARL_V3_RUN"], "stop-requested").exists()
                and super()._fits_token_allowance())

    def _repeat_iterator(self):
        # Pin the worker's task catalogue before the concurrent session threads.
        dataset_rows(self._factory)
        last = None
        for identifier, index, row in super()._repeat_iterator():
            if row is not None and identifier != last:
                self.started = getattr(self, "started", 0) + 1
                self._factory.group_policies[identifier] = self._request_model
                event("group_started", group_id=identifier, task=row["task_name"], policy=self._request_model,
                      sampling_policy_version=self.model_version)
            last = identifier
            yield identifier, index, row

    async def generate_one(self, prompt, tools, group_id=0):
        event("rollout_started", group_id=group_id)
        result = await super().generate_one(prompt, tools, group_id)
        event("rollout_finished", group_id=group_id, reward=result.reward,
              rollout_id=str(result.rollout_id), trainable=bool(result.turns))
        return result

    async def _score_group(self, group):
        samples = await super()._score_group(group)
        rewards = [float(v) if math.isfinite(v) else None for v in group.rewards]
        advantages = self.compute_advantages(group).tolist()
        run = Path(os.environ["CARL_V3_RUN"])
        episodes, prefixes, policies = {}, set(), set()
        for lines in group.completions_sequences:
            for seq in lines:
                if len(seq.input_ids) > self.config.token_budget:
                    raise ValueError("A reconciled sequence exceeds the complete per-rank token budget")
                folder = run / "episodes" / str(seq.rollout_id)
                details = read(folder / "receipt.json")
                episodes[str(seq.rollout_id)] = details
                prefixes.add(details["prompt_sha256"])
                policies.add(details["policy"])
        policy = self._factory.group_policies.pop(group.group_id)
        if policies - {policy} or len(prefixes) > 1:
            raise ValueError("A group mixed policy versions or exact first-prompt token IDs")
        complete = len(episodes) == self.config.num_generations and set(rewards) == {0.0, 1.0}
        accepted = samples if complete else []
        event("group_scored", group_id=group.group_id, rewards=rewards, advantages=advantages,
              rollouts=list(episodes), policy=policy, accepted_sequences=len(accepted),
              earned_tokens=sum(int(sample.completion_mask.sum()) for sample in accepted))
        self.scored = getattr(self, "scored", 0) + 1
        if self.scored == int(os.environ["CARL_V3_MAX_GROUPS"]):
            (run / "group-cap-scored").touch()
        return accepted


def local_tensor(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def rank_values(value):
    values = [None] * get_world_size()
    all_gather_object(values, value)
    return values


def tensor_hash(named_parameters):
    digest = hashlib.sha256()
    for name, weight in named_parameters:
        digest.update(name.encode())
        tensor = local_tensor(weight.detach()).flatten().contiguous().view(torch.uint8)
        for chunk in tensor.split(8 * 1024**2):
            digest.update(memoryview(chunk.cpu().numpy()))
    return digest.hexdigest()


def lora(model):
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not params or any(".lora_" not in name for name, _ in params):
        raise RuntimeError("The optimizer must see only the LoRA adapter")
    return params


def packed_key(input_ids, token_indices):
    return digest([input_ids.tolist(), token_indices.tolist()])


class ObservedQueue:
    def __init__(self, trainer, queue):
        self.trainer, self.queue = trainer, queue

    def get(self, *args, **kwargs):
        try:
            sample = self.queue.get(*args, **kwargs)
        except Empty as exc:
            run = self.trainer.run
            if ((run / "group-cap-scored").exists() or (run / "stop-requested").exists()
                    or time.time() >= float(os.environ["CARL_V3_DEADLINE"])):
                raise RuntimeError("Sampling budget reached while waiting for the next complete microbatch") from exc
            raise
        if isinstance(sample, RolloutSample):
            event("sample_dequeued", group_id=sample.group_id, sampled_version=sample.model_version,
                  learner_version=self.trainer.model_version, queue_size=self.qsize(),
                  queue_wait_s=time.time() - sample.enqueued_at if sample.enqueued_at is not None else None,
                  age=self.trainer.model_version - sample.model_version)
        return sample

    def qsize(self):
        return self.queue.qsize()


class UpdateAudit(TrainerCallback):
    def __init__(self, run, stop_after, first):
        self.run = run
        self.stop_after = stop_after
        self.first = first
        self.previous = None

    def on_train_begin(self, args, state, control, model, **kwargs):
        lora(model)
        if not any(isinstance(p, DTensor) for _, p in model.named_parameters()):
            raise RuntimeError("Learner model has no FSDP2 DTensor parameters")
        self.previous = rank_values(tensor_hash(lora(model)))
        if self.first:
            self.base_before = rank_values(tensor_hash((n, p) for n, p in model.named_parameters()
                                                      if not p.requires_grad))

    def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
        grads = [local_tensor(p.grad) for _, p in lora(model) if p.grad is not None]
        finite = bool(grads and all(torch.isfinite(g).all().item() for g in grads))
        norm2 = sum(g.detach().float().square().sum().item() for g in grads)
        stats = rank_values((finite, norm2))
        if not all(f and math.isfinite(n) for f, n in stats):
            raise RuntimeError("Missing or nonfinite gradients")
        self.shard_norms = [math.sqrt(n) for _, n in stats]
        self.norm = math.sqrt(sum(n for _, n in stats))
        if not math.isfinite(self.norm) or self.norm <= 0:
            raise RuntimeError("LoRA gradient norm is zero/nonfinite")

    def on_optimizer_step(self, args, state, control, model, optimizer, **kwargs):
        after = rank_values(tensor_hash(lora(model)))
        if self.previous == after:
            raise RuntimeError("An optimizer callback fired without changing any adapter bytes")
        step = state.global_step + 1
        state_step = max((int(local_tensor(v["step"])) if torch.is_tensor(v["step"]) else int(v["step"])
                          for v in optimizer.state.values() if "step" in v), default=0)
        observed = max(rank_values(state_step))
        if step != observed:
            raise ValueError("Actual AdamW state step differs from Trainer step")
        result = {"step": step, "gradient_norm_after_clip": self.norm,
                  "local_gradient_norms": self.shard_norms,
                  "adapter_rank_hashes_before": self.previous, "adapter_rank_hashes_after": after}
        if self.first and step == 1:
            base_after = rank_values(tensor_hash((n, p) for n, p in model.named_parameters()
                                                if not p.requires_grad))
            if self.base_before != base_after:
                raise ValueError("Frozen model weights changed")
            result.update(base_rank_hashes_before=self.base_before, base_rank_hashes_after=base_after)
            if get_rank() == 0:
                write(self.run / "first-update.json", result)
        if get_rank() == 0:
            event("optimizer_updated", **result)
        self.previous = after

    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step >= self.stop_after or time.time() >= float(os.environ["CARL_V3_DEADLINE"])
                or (self.run / "stop-requested").exists()):
            control.should_save = True
            control.should_training_stop = True
        return control


class AuditedTrainer(HarnessAsyncGRPOTrainer):
    loop_cls = AuditedLoop

    def __init__(self, *args, run, ids, first, **kwargs):
        self.run = run
        self.fixed_ids = ids
        self.first = first
        self.parity_done = False
        self.probe_before = None
        self.published_version = 0
        super().__init__(*args, **kwargs)
        plugin = self.accelerator.state.fsdp_plugin
        if (self.accelerator.num_processes != 2 or plugin is None or plugin.fsdp_version != 2
                or "SHARDED_STATE_DICT" not in str(plugin.state_dict_type) or not self._lora_sync):
            raise ValueError("Expected two FSDP2 Learner ranks and the official adapter-only synchronization")

    def get_train_dataloader(self):
        data = super().get_train_dataloader()
        if not self.accelerator.is_main_process:
            return data
        dataset = self._rollout_dataset
        dataset.queue = ObservedQueue(self, dataset.queue)
        collate_fn = data.base_dataloader.collate_fn

        def observe(features):
            batch = collate_fn(features)
            for rank, row in enumerate(features[0]):
                groups = sorted({sample["group_id"] for sample in row.samples})
                event("batch_collated", learner_rank=rank, groups=groups, learner_version=self.model_version,
                      batch_key=packed_key(batch["input_ids"][rank], batch["token_index"][rank]))
            return batch

        data.base_dataloader.collate_fn = observe
        return data

    def training_step(self, model, inputs, num_items_in_batch):
        torch.cuda.synchronize()
        stamp = {"next_step": self.state.global_step + 1, "learner_policy_version": self.published_version,
                 "learner_rank": get_rank(),
                 "batch_key": packed_key(inputs["input_ids"][0], inputs["token_index"][0])}
        event("forward_backward_started", **stamp)
        value = super().training_step(model, inputs, num_items_in_batch)
        torch.cuda.synchronize()
        event("forward_backward_finished", loss=float(value), supervised=int(inputs["global_n_tokens"][0]), **stamp)
        if not torch.isfinite(value).item():
            raise ValueError("Nonfinite learner loss")
        return value

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        output = {}

        def capture(_module, _args, result):
            output["logps"] = result.log_probs.detach().flatten()

        hook = model.register_forward_hook(capture)
        try:
            loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        finally:
            hook.remove()
        selected = inputs["token_index"] >= 0
        indices = inputs["token_index"][selected]
        old = inputs["old_log_probs"][selected].detach().float()
        now = output["logps"][indices].float()
        delta = (now - old).abs()
        ratio = (now - old).exp()
        masked = ((ratio < self.args.vllm_importance_sampling_clip_min)
                  | (ratio > self.args.vllm_importance_sampling_clip_max))
        finite = bool(torch.isfinite(old).all() and torch.isfinite(now).all() and torch.isfinite(ratio).all())
        n = ratio.numel()
        if not finite:
            raise ValueError("Nonfinite sampler-to-learner log-probability ratio")
        event("effective_mask", step=self.state.global_step + 1, learner_rank=get_rank(),
              supervised=n, trained=n - int(masked.sum()))
        if not self.first or self.parity_done:
            return loss
        fraction = masked.float().mean().item() if n else 1.0
        limit = read(CONFIG)["train"]["maximum_first_step_importance_mask_fraction"]
        passed = bool(n and finite and fraction <= limit)
        receipt = {"passed": passed, "n": n, "finite": finite, "masked_fraction": fraction,
                   "checks": "Learner gathered logps vs exact captured sampler logps, before the first update"}
        if n and finite:
            receipt.update(mean_abs=delta.mean().item(), p95_abs=delta.quantile(.95).item(),
                           max_abs=delta.max().item(),
                           ratio_min=ratio.min().item(), ratio_max=ratio.max().item(),
                           sampled_logps=old.cpu().tolist(), learner_logps=now.cpu().tolist())
        write(self.run / f"cuda-logprob-parity-rank{get_rank()}.json", receipt)
        ranks = rank_values({key: value for key, value in receipt.items() if not key.endswith("_logps")})
        accepted = all(item["passed"] for item in ranks)
        if self.accelerator.is_main_process:
            write(self.run / "cuda-logprob-parity.json", {"passed": accepted, "ranks": ranks})
        if not accepted:
            raise ValueError("First real learner batch failed the fixed probability contract")
        self.parity_done = True
        return loss

    def _sync_weight(self):
        started = time.monotonic()
        old = self.published_version
        event("weight_sync_started", previous_policy_version=old, learner_rank=get_rank())
        # TRL's PEFT export all-gathers DTensor shards, so every rank must enter.
        super()._sync_weight()
        result = [None]
        if self.accelerator.is_main_process:
            try:
                result[0] = self.check_publication(old, started)
            except Exception as exc:
                result[0] = {"error": f"{type(exc).__name__}: {exc}"}
        broadcast_object_list(result, src=0)
        if "error" in result[0]:
            raise RuntimeError(result[0]["error"])
        self.published_version = result[0]["policy_version"]
        event("weight_sync_collective_returned", learner_rank=get_rank(), **result[0])

    def check_publication(self, old, started):
        policy = f"{self._lora_name}-v{self.model_version}"
        path = Path(self._lora_dir, policy)
        _, engine = urls()
        status = httpx.get(engine + "/v1/models", timeout=30, trust_env=False)
        status.raise_for_status()
        loaded, = [m for m in status.json()["data"] if m["id"] == policy]
        if Path(loaded["root"]).resolve() != path.resolve():
            raise RuntimeError("Adapter publishing returned a different source")
        result = {"policy": policy, "previous_policy_version": old, "policy_version": self.model_version,
                  "global_step": self.state.global_step,
                  "adapter_sha256": sha(path / "adapter_model.safetensors"), "seconds": time.monotonic()-started}
        write(self.run / "latest-policy.json", result)
        event("weight_sync_finished", **result)
        if self.first and self.state.global_step <= 1:
            probe = fixed_probe(policy, self.fixed_ids)
            if self.probe_before is None:
                self.probe_before = probe
            else:
                changed = sum(a != b for a, b in zip(self.probe_before["logprobs"], probe["logprobs"], strict=True))
                if changed == 0:
                    raise RuntimeError("Post-update checkpoint did not change the fixed vLLM probabilities")
                write(self.run / "policy-probe.json",
                      {"before": self.probe_before, "after": probe, "changed_positions": changed})
        return result

    def _save_checkpoint(self, model, trial):
        # Native FSDP2 DCP adapter, optimizer shards, scheduler, per-rank RNG and rollout state.
        super()._save_checkpoint(model, trial)
        error = [None]
        if self.accelerator.is_main_process:
            try:
                receipt = read(self.run / "latest-policy.json")
                if receipt["global_step"] != self.state.global_step:
                    raise ValueError("Checkpoint and publication steps diverged")
                source = Path(self._lora_dir, receipt["policy"])
                checkpoint = Path(self.args.output_dir, f"checkpoint-{self.state.global_step}")
                for name in ("adapter_model.safetensors", "adapter_config.json"):
                    shutil.copyfile(source / name, checkpoint / name)
                if sha(checkpoint / "adapter_model.safetensors") != receipt["adapter_sha256"]:
                    raise ValueError("Checkpoint serving adapter differs from its synchronized snapshot")
                event("checkpoint_saved", step=self.state.global_step, policy=receipt["policy"])
            except Exception as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])


def make_args(run):
    cfg = read(CONFIG)
    rollout = cfg["rollout"]
    t = cfg["train"]
    return AsyncGRPOConfig(
        output_dir=str(run / "checkpoints"), max_steps=t["total_planned_updates"],
        save_strategy="steps", save_steps=1, save_total_limit=None, logging_steps=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_generations=t["num_generations"], token_budget=t["token_budget"],
        learning_rate=t["learning_rate"], lr_scheduler_type="constant",
        temperature=rollout["temperature"], top_p=rollout["top_p"], top_k=rollout["top_k"],
        min_p=rollout["min_p"], max_completion_length=rollout["max_completion_length"],
        score_centering=False, max_staleness=t["max_staleness"], weight_sync_steps=t["weight_sync_steps"],
        max_concurrent_rollouts=t["max_concurrent_rollouts"], vllm_server_base_url=urls()[1],
        max_tool_calling_iterations=rollout["max_model_calls"],
        dtype="bfloat16", bf16=True, optim="adamw_torch",
        gradient_checkpointing=False,  # Accelerate's single FSDP2 activation checkpoint wrapper.
        vllm_server_timeout=180.0,
        model_init_kwargs={"use_cache": False, "local_files_only": True},
        chat_template_kwargs={"enable_thinking": False}, seed=rollout["seed"], report_to="none",
    )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--stop-after", type=int, required=True)
    cli.add_argument("--resume", type=Path)
    args = cli.parse_args()
    gpu = runtime(expected_devices=2)
    state = PartialState(timeout=timedelta(minutes=5))
    if state.num_processes != 2:
        raise RuntimeError("Accelerate did not start both FSDP2 ranks")
    select_sm80_backend()
    run = Path(os.environ["CARL_V3_RUN"])
    cfg = read(CONFIG)
    training = cfg["train"]
    if args.resume is None and args.stop_after != training["first_stage_updates"]:
        cli.error("The first run is limited to one verified update")
    args_config = make_args(run)
    factory = partial(AuditedFactory, **session_args())
    data = Dataset.from_list(dataset_rows(factory()))
    local = model_path()
    text = AutoTokenizer.from_pretrained(str(local), local_files_only=True, trust_remote_code=False)
    ids = text("Check the version parsing requirements and implement the documented behavior in Python.",
               add_special_tokens=False)["input_ids"]
    adapter = LoraConfig(r=training["lora_rank"], lora_alpha=training["lora_alpha"],
                         target_modules=training["lora_target_modules"], task_type="CAUSAL_LM")
    audit = UpdateAudit(run, args.stop_after, args.resume is None)
    model = AuditedTrainer(model=str(local), processing_class=text, args=args_config, peft_config=adapter,
                           harness_session_factory=factory, rollout_reward_fn=correctness, train_dataset=data,
                           callbacks=[audit], run=run, ids=ids, first=args.resume is None)
    if state.is_main_process:
        write(run / "trainer-config.json", args_config.to_dict())
    event("learner_ready", device=gpu, learner_rank=get_rank())
    model.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    completed = int(model.state.global_step)
    observations = rank_values({"rank": get_rank(),
                                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()})
    if state.is_main_process:
        write(run / "trainer-receipt.json", {"global_step": completed, "ranks": observations})
    if not (run / "checkpoints" / f"checkpoint-{completed}").is_dir():
        raise RuntimeError("No completed checkpoint at the stopping boundary")


if __name__ == "__main__":
    main()
