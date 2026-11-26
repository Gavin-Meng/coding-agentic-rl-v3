# V3 兼容性记录

## 1. 锁定版本与上游关系

- 基模和 tokenizer 为 Qwen/Qwen3.8-27B，revision
  `37561816e66b5b9153b1b64c10968c18c1e6f4a0`。
  发布配置声明 `Qwen3_5ForConditionalGeneration`。文本分支有
  16 层全注意力与 48 层 Gated DeltaNet，BF16 索引中的权重约 55.6 GB。
  [模型锁定快照](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0)
- TRL 1.19.0，Git commit `39c749ed9ce139c594b4e8540fecd0b3cbcf19c9`。
  锁定版 AsyncGRPO 文档列出 FSDP2、PEFT、原生 Harness 与混合模型支持。
  Learner 的 Gated DeltaNet 树打包要求 Transformers >=5.15.0。
  [锁定文档](https://github.com/huggingface/trl/blob/39c749ed9ce139c594b4e8540fecd0b3cbcf19c9/docs/source/async_grpo_trainer.md)
  [锁定实现](https://github.com/huggingface/trl/blob/39c749ed9ce139c594b4e8540fecd0b3cbcf19c9/trl/trainer/async_grpo_trainer.py)
- OpenEnv 0.15.0、Harbor 0.27.0、Pi 1.5.2、vLLM 0.34.0、PyTorch
  2.14.0+cu130、Transformers 5.22.0、PEFT 0.22.0、Accelerate 1.16.0。
  控制端、采样端、学习端分别使用哈希锁，解决 Harbor 和 vLLM 的 FastAPI
  版本约束冲突。vLLM 采样端启用纯语言路径、BF16、处理后 logprobs、
  精确 token IDs、Qwen 工具解析和运行时 LoRA。
- HF 发布的 Pi 配方为双 B200、rank 32、组大小 8、
  65,536 token 级别的预算及工具效率奖励。V3 固定三张 A800、组大小 4、
  rank 16 和纯产物奖励，旨在验收原生 Harness 训练路径。
  [HF 方法](https://huggingface.co/blog/harness-adapters-at-scale)
  [HF Pi Adapter 配方](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted)

## 2. SM80 与 FSDP2 验收

锁定版 TRL 中 `AsyncGRPOTrainer` 通过 `patch_tree_recurrent` 按分支
重建 Gated DeltaNet 和卷积状态。V3 沿用 V2.1 的 PyTorch CUDA
`varlen_attn` 接口为全注意力提供不等长序列和 GQA 支持。A800 使用
CUDA varlen 接口。后端选择写在 `src/carl_v3/tree_backend.py`。

已安装 PyTorch 的编译架构包含 SM80，vLLM CUDA 共享扩展可以 CPU 导入。
这两项验证的是 wheel 和接口。真实 SM80 由
`src/carl_v3/kernel_probe.py` 验收，包括一个随机初始化的两层混合模型
的分支打包与串行 BF16 对比和非零有限梯度。探针保留锁定 27B 的 head 维度
和 GQA 比例。两张 Learner 卡都要通过。
27B 实际全尺寸的模型、混合缓存、FSDP2 分片和 CUDA 算子只有首步能验证。

`configs/accelerate-fsdp2.yaml` 固定两进程 BF16、按 Qwen decoder layer
分片、前向后重分片、激活重算和 `SHARDED_STATE_DICT`。
初始每 rank 的 token budget 为 8,192，等于服务端完整上下文长度。
每条最多 8 次模型请求，每次最多 768 生成 token。采样端最多 2 个同时
生成序列，最多 4 个会话。第一步重要性采样的范围采用 TRL 锁定版
[0.5, 5.0]。配方额外预定首步保留至少 90% 的可训练位置作为启动验收。
日志同时给出两 rank 各自的差异与范围。

## 3. 全量同步修复与版本化 LoRA

Issue #7103 对应 FSDP2 全量 NCCL 权重传输中发送 tensor 与 manifest
dtype 不一致问题。PR #7131 的合并提交
`c6eac19f63be27498f2a7c8108d1e8a4dffde23f` 已验证为 TRL 锁定
commit 的祖先。锁定版的 `weight_transfer.py` 在流式发送时检查并转换 dtype。
[Issue #7103](https://github.com/huggingface/trl/issues/7103)
[PR #7131](https://github.com/huggingface/trl/pull/7131)
[实际代码](https://github.com/huggingface/trl/blob/39c749ed9ce139c594b4e8540fecd0b3cbcf19c9/trl/generation/weight_transfer.py)

本配方使用 TRL 的 `save_lora_adapter` 和 `_sync_weight_lora`。
每个 rank 调用 `DTensor.full_tensor()`，rank 0 写入临时目录后再发布。
随后 vLLM 读取该目录并注册新名称 `trl-policy-vN`，拥有独立的前缀缓存空间。
CPU 双进程 Gloo 已验证真实 LoRA 聚合和两个版本的导出值。真正的
A800 验收由两 rank 的 CUDA 更新、实际加载源路径、adapter SHA256、固定
token 概率变化及新 Pi 会话命中新策略构成。FSDP2 Learner 进程间的 NCCL
通信和发布 adapter 给 vLLM 是两条独立链路。

## 4. 当前验收与扩大范围的条件

- Python 3.12.14 上实际安装三套锁，三个依赖检查通过。Pi CLI 可报告
  1.5.2。两进程 DTensor 导出与 9 道任务的已知正确和预设回归契约通过。
- Smoke 评估独立 Docker 中的真实正确、真实错误及只打印 PASS 三种情况。
  奖励为 0、1 或 None。Pi 无 IP 出网，代理通过每次运行自己的 Unix socket
  提供带会话键的生成路由。Verifier 只读取约定文件并在独立 Docker 评分。
- 首步取到混合奖励、有限梯度、AdamW 产生更新、LoRA 哈希变化后，核验策略
  发布和新请求命中。随后 pilot 通过官方入口恢复两 rank 的 optimizer、scheduler、
  RNG 与 rollout 位置。`timeline.jsonl` 记录实际异步区间及版本，重叠指标
  是请求区间与前反向区间的交集。
- 遇到 OOM，可在首步之前降低完整上下文和 token budget，并重新做 smoke。
  两者随同配置锁定，训练序列长度由采样上限和打包检验共同保证。
  FSDP2 阻塞复现后才考虑分别预检一张 Sampler 和一张 Learner 的方案。
  单卡 Learner 需要重新核对 27B BF16 显存、导出方法及 checkpoint 恢复格式。
  第三张卡可空置。默认入口保持静态三卡拓扑。

## 5. 两台服务器的批准和验证

交付入口默认固定单机。如果实验室安排 Sampler 和 Learner 位于不同主机，
先由管理员明确批准控制端访问 vLLM 的接口和端口，并为 adapter 选择一处
角色用户可访问的共享目录。Learner 写入目录的绝对路径必须与
Sampler 进程看到的路径一致。两个 Learner rank 优先保持在同一主机。
同时从 Learner 到获准的 vLLM 端点核对 `/health`、`/v1/models`、
OpenEnv 的捕获调用和 TRL 的 pause/resume 及加载权限。

以下命令仅在双方选定同一个已获准路径后运行。

```bash
# Learner 端写入批准的共享目录
printf '%s\n' 'carl-v3-shared-path-check' > "$CARL_APPROVED_SHARED/carl-v3.probe"
sha256sum -- "$CARL_APPROVED_SHARED/carl-v3.probe"
# Sampler 端用运行 vLLM 的身份核对同一绝对路径及哈希
test -r "$CARL_APPROVED_SHARED/carl-v3.probe"
sha256sum -- "$CARL_APPROVED_SHARED/carl-v3.probe"
# Learner 端访问管理员已经批准并配置好的端点
curl --fail --max-time 5 "$CARL_APPROVED_SAMPLER_URL/health"
```

路径与网络探测之后还需在实际 vLLM 加载 adapter，复核加载回执和新请求。
本地环境未取得跨节点访问授权，因此交付包含上述检查步骤。
