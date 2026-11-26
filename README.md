# Coding Agentic RL V3

**Harness-native Agentic Reinforcement Learning with Distributed AsyncGRPO**

Coding Agentic RL V3 targets Qwen3.8-27B with the native Pi coding harness.
OpenEnv captures model requests from Pi inside Harbor, and an independent
verifier grades each final artifact. TRL AsyncGRPO trains LoRA adapters on
two FSDP2 ranks and publishes versioned adapters to a separate vLLM sampler.

[中文部署手册](DEPLOYMENT.zh-CN.md) · [Compatibility and pinned sources](COMPATIBILITY.md)

## Validation status

- **V3 CPU checks passed.** The pinned environments and CLI entry points
  were checked. Nine public task fixtures and two-rank DTensor LoRA export
  passed on CPU. The source release has a per-file SHA256 manifest.
- **Earlier 4B experiments ran on CUDA.** The project ran synchronous GRPO
  and a small AsyncGRPO pilot with versioned LoRA synchronization. Those
  experiments used Qwen3-4B and the earlier harness configuration.
- **V3 GPU acceptance is pending.** The three-A800 smoke run, the first
  verified 27B optimizer step, policy reload and the pilot are the GPU
  acceptance sequence.

## Architecture

```text
Harbor sandbox + Pi
    | model calls                       final artifact
    v                                        |
OpenEnv Capture Proxy <--> A800 0 / vLLM     v
    | TrainingTrace                 Independent Verifier
    | token IDs, logprobs, masks             | binary reward
    +-------------------+--------------------+
                        v
        A800 1 + A800 2 / TRL AsyncGRPO Learner
                  FSDP2 + BF16 LoRA
                        |
      versioned adapter + checkpoint --> vLLM sampler
```

The [A800 configuration](configs/qwen27b_a800.yaml) fixes the single-node
GPU roles, model revision, rollout limits and training budget.
[SOURCE.json](SOURCE.json) records the provenance of the original deployment
source. Git tags and the checkout hash identify the exact release.

## Requirements

- One Linux x86_64 host with three authorized NVIDIA A800 80 GB GPUs.
- glibc 2.35+, NVIDIA driver R580+, local Docker with Compose v2, Bash,
  `uv` and `tmux`. An allocated Slurm job is also supported.
- At least 96 GiB of available RAM and 90 GiB of free disk space.
- Network access for package installation and the pinned public assets.
  The model weights are downloaded on the deployment host.

## Install and smoke test

```bash
git clone --single-branch --branch v3-a800-4a24fb0 \
  https://github.com/Gavin-Meng/coding-agentic-rl-v3.git &&
  cd coding-agentic-rl-v3 &&
  git switch --detach "$(git rev-parse 'v3-a800-4a24fb0^{commit}')" &&
  sha256sum --strict -c SHA256SUMS &&
  tmux new -s carl-v3-a800
```

Inside tmux, enter the complete UUIDs of the three allocated GPUs.

```bash
nvidia-smi -L
read -r -p 'Sampler GPU UUID: ' CARL_SAMPLER_GPU
read -r -p 'Learner rank 0 GPU UUID: ' CARL_LEARNER_GPU0
read -r -p 'Learner rank 1 GPU UUID: ' CARL_LEARNER_GPU1
export CARL_SAMPLER_GPU CARL_LEARNER_GPU0 CARL_LEARNER_GPU1
export CARL_GPU_AUTHORIZED=1

bash scripts/preflight.sh &&
  bash scripts/setup.sh &&
  bash scripts/preflight.sh --installed &&
  bash scripts/download_assets.sh &&
  bash scripts/smoke_test.sh --run-id a800-smoke --wall-minutes 30
```

Setup installs the three locked environments in `.venvs/`. Asset downloads
write to `.state/assets/`. The smoke test exercises the independent Docker
verifier, hybrid attention on both learner GPUs, NCCL/FSDP2 adapter export,
27B inference, Pi tool calls and token capture. It records its verdict in
`.state/runs/a800-smoke/acceptance.json`.

## Train and resume

After smoke passes, run one verified optimizer update with the same assets
and configuration.

```bash
bash scripts/train.sh --run-id a800-first \
  --smoke-run .state/runs/a800-smoke --stop-after 1 --wall-minutes 60
cat .state/runs/a800-first/acceptance.json
```

The first-step checks cover finite gradients, a real adapter update, versioned
vLLM reload and checkpoint metadata. After `passed: true`, use a new run ID
to resume from the accepted checkpoint. `--stop-after` is the absolute
optimizer step, so this command can add up to five updates.

```bash
bash scripts/train.sh --run-id a800-pilot \
  --smoke-run .state/runs/a800-smoke \
  --resume .state/runs/a800-first/checkpoints/checkpoint-1 \
  --stop-after 6 --wall-minutes 60
```

Each saved checkpoint includes the adapter, distributed model and optimizer
state, scheduler, RNG and trainer state. Read `summary.json` for the actual
step count. To request a stop at an optimizer boundary from another shell,
create `.state/runs/a800-pilot/stop-requested`.

## Evaluate and return artifacts

Use an accepted checkpoint path. The example uses step 6.

```bash
checkpoint=.state/runs/a800-pilot/checkpoints/checkpoint-6
bash scripts/evaluate.sh --run-id a800-eval --wall-minutes 30 \
  --checkpoint "$checkpoint"

bash scripts/archive.sh .state/runs/a800-pilot --with-raw-traces \
  --checkpoint "$checkpoint"
bash scripts/archive.sh .state/runs/a800-eval --with-raw-traces
(cd .state/exports/a800-pilot && sha256sum --strict -c SHA256SUMS)
(cd .state/exports/a800-eval && sha256sum --strict -c SHA256SUMS)
```

Review the export before sending it to an approved lab destination. Set
`CARL_APPROVED_DEST` to that destination, then transfer the results.

```bash
rsync -a --partial -- .state/exports/a800-pilot/ "$CARL_APPROVED_DEST/a800-pilot/"
rsync -a --partial -- .state/exports/a800-eval/ "$CARL_APPROVED_DEST/a800-eval/"
```

## License

Project-authored code and documentation are available under the
[MIT License](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md)
for the separately sourced software, model and task sources.
