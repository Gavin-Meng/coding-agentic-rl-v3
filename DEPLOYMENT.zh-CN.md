# Coding Agentic RL V3 · A800 27B 交付

本包执行固定的单机三卡方案。一张 A800 80GB 运行 CUDA vLLM，
两张 A800 80GB 共同运行 FSDP2 Learner。Pi 在 Harbor 隔离会话中执行工具。
OpenEnv 采集每次 vLLM 请求的训练数据，独立 Verifier 评分最终文件。
TRL 负责 AsyncGRPO 和版本化 LoRA 同步。

本地验收覆盖依赖锁、入口、九道公开任务、双进程 adapter 导出和源包校验。
A800 上的首次运行从下面的 smoke 开始。模型与组件版本见 [兼容性记录](COMPATIBILITY.md)。

## 1. 解包与检查

在获准的实验室服务器解包。需要 Linux x86_64、glibc 2.35+、NVIDIA R580
或更新驱动、三张完整 A800 80GB、本机 Docker daemon 和 Compose v2。
安装工具使用 uv，网络只在安装和下载公开资产阶段使用。
建议运行目录有至少 90 GiB 空闲，节点有至少 96 GiB 可用内存。

```bash
tar -xzf 'v3-a800-release-<short-commit>.tar.gz'
cd v3-a800-release
sha256sum --strict -c SHA256SUMS
nvidia-smi -L
tmux new -s carl-v3-a800
```

在 tmux 中按实际分配填写完整 UUID。启用前确认三张卡的授权和时间窗。
同一台机器上也可在已获准的 Slurm 作业内运行。

```bash
export CARL_GPU_AUTHORIZED=1
export CARL_SAMPLER_GPU='GPU-<sampler-uuid>'
export CARL_LEARNER_GPU0='GPU-<learner-uuid-0>'
export CARL_LEARNER_GPU1='GPU-<learner-uuid-1>'

bash scripts/preflight.sh
bash scripts/setup.sh
bash scripts/preflight.sh --installed
bash scripts/download_assets.sh
```

前置检查输出 GPU 型号、空闲显存、驱动、MIG、PCIe/NVLink 拓扑、内存、
磁盘、Docker/Compose 和公开模型 revision 可达性。根据实验室的批准窗口设定
每次 `--wall-minutes`。运行中的 vLLM 和 OpenEnv 仅监听本机 18300、18301。
三个独立环境写入 `.venvs/`，公开资产写入 `.state/assets/`。
`bash scripts/download_assets.sh --tasks-only` 可单独准备公开任务源。

## 2. 首步与 pilot

```bash
bash scripts/smoke_test.sh --run-id a800-smoke --wall-minutes 30
bash scripts/train.sh --run-id a800-first \
  --smoke-run .state/runs/a800-smoke --stop-after 1 --wall-minutes 60
```

Smoke 依次核验独立 Docker verifier、两张 Learner 卡的混合注意力前反向、
NCCL/FSDP2 小模型的两版 adapter 导出、27B BF16 vLLM、Pi 工具调用、
逐请求 Token Capture 和最终产物评分。
训练需要读取同节点、同配置和同资产生成的 smoke 回执。

首步训练只接收同题、同初始 token、同策略版本、四条完整可评分且奖励
有差异的组。首个微批检查两 rank 的真实概率与重要性屏蔽，再核对有限梯度、
实际 AdamW 更新、冻结基模与变动的 LoRA。同步验证读取 vLLM 返回的 adapter
路径、固定输入概率及下一次真实 Pi 请求命中的版本。一个新进程读取
adapter 和 DCP 分片元数据，后续推理使用该 checkpoint 目录再次评测。

首步 `acceptance.json` 中 `passed: true` 后续训。下面从第 1 步恢复到
第 6 步，因此 pilot 最多增加 5 次更新。可将目标设在 6–10 之间。

```bash
bash scripts/train.sh --run-id a800-pilot \
  --smoke-run .state/runs/a800-smoke \
  --resume .state/runs/a800-first/checkpoints/checkpoint-1 \
  --stop-after 6 --wall-minutes 60
```

每次启动默认最多 20 个新训练组，每组 4 条，`--max-groups` 可调低。
最多 120 分钟，留 450 秒做保存和采样验收。首步和 pilot 的墙钟窗口分别计时。
`summary.json` 以实际 optimizer 更新计数，`acceptance.json` 的
`target_reached` 表示有没有达到本次目标。组预算、零方差或算子故障会在日志中
留下已完成量。实际预取会影响 staleness，`max_staleness=1` 用于官方队列出队过滤。

## 3. 查看、停止与续训

```bash
tmux attach -t carl-v3-a800
tail -f .state/runs/a800-first/logs/train.log
cat .state/runs/a800-first/summary.json
cat .state/runs/a800-first/acceptance.json
```

从另一窗口请求在优化器边界停止，运行 ID 按实际替换。

```bash
touch .state/runs/a800-pilot/stop-requested
```

训练脚本每个可用更新边界保存 adapter、两 rank 的模型和 optimizer DCP、
scheduler、RNG、Trainer 和 rollout 状态。以回执中核验的 checkpoint 为准。
恢复时使用新的 `--run-id` 和绝对目标步数，传入选定的 `--resume`。
强制中止时保留此前已落盘的 checkpoint 和日志。
使用 `cleanup.json` 核对属于本次运行的服务、容器和工作卷已退出。

## 4. 评测与回传

选定已验收 checkpoint 后执行独立评测。下面的步数按实际核验结果替换。

```bash
bash scripts/evaluate.sh --run-id a800-eval --wall-minutes 30 \
  --checkpoint .state/runs/a800-pilot/checkpoints/checkpoint-6

bash scripts/archive.sh .state/runs/a800-pilot --with-raw-traces \
  --checkpoint .state/runs/a800-pilot/checkpoints/checkpoint-6
bash scripts/archive.sh .state/runs/a800-eval --with-raw-traces
```

评测按固定顺序和种子比较 Base/Adapter，每道任务每路一次。
仅同题初始 token 一致且双方可评分的任务进入配对差值与 bootstrap 区间。
四道评测题来自与训练不同的公开仓库，适用于部署验收。
结果见 `evaluation.json`，失败原因及部分结果见 `evaluation-partial.json`。

归档时可增量纳入源码、配置、锁、原始 capture、Pi 工具轨迹、最终产物、
验收记录、日志和所选可恢复 checkpoint。运行目录和原始文件可留在节点。
核对导出目录内容和 SHA 后，使用实验室批准的路径或传输方式回传。
下面的命令需先由操作者设置 `CARL_APPROVED_DEST`。

```bash
(cd .state/exports/a800-pilot && sha256sum --strict -c SHA256SUMS)
rsync -a --partial -- .state/exports/a800-pilot/ "$CARL_APPROVED_DEST/a800-pilot/"
```

在训练中可先选归档日志和轨迹，最终再追加已核验的 checkpoint。
源包的 Git commit 由 `SOURCE.json` 记录，实际进程参数保存在各运行的
`logs/*.command.json`。
