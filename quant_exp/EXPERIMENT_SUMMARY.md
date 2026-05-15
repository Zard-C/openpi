# QuantVLA-OpenPI 量化测速实验总结

## 背景

本实验尝试把 QuantVLA 中的选择性量化、OHB 输出能量修正和 ATM 注意力温度修正思路迁移到 OpenPI pi0.5 PyTorch 推理链路中。当前目标不是直接得到最终部署加速，而是先建立可复现的误差和性能评估路径。

实验对象为 `pi05_droid`，推理使用固定 observation、固定 noise 和固定 denoise step 数，主要关注 action expert 的 MLP。除特别说明外，attention、norm、action head 和 time MLP 保持 BF16。

## 模型权重规模

以下统计来自本地 `pi05_droid_pytorch_bf16` checkpoint，口径为 PyTorch 参数张量大小。`actual_MiB` 按 checkpoint 中真实 dtype 计算；理论 BF16/INT8/INT4 大小只按参数个数乘以位宽估算，不包含量化 scale、zero point、packing metadata 和 runtime cache。

### 大脑与小脑总量

| 模块 | 说明 | 参数量 | actual MiB | BF16 MiB | INT8 MiB | INT4 MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 大脑: PaliGemma VLM | `paligemma_with_expert.paligemma` | 2.923B | 5577.82 | 5575.82 | 2787.91 | 1393.95 |
| 语言模型部分 | `paligemma.language_model` | 2.509B | 4784.79 | 4784.64 | 2392.32 | 1196.16 |
| 视觉塔部分 | `paligemma.vision_tower` | 0.412B | 788.53 | 786.67 | 393.34 | 196.67 |
| 多模态 projector | `paligemma.multi_modal_projector` | 0.002B | 4.50 | 4.50 | 2.25 | 1.13 |
| 小脑: Gemma Expert model | `gemma_expert.model` | 0.428B | 1038.43 | 816.22 | 408.11 | 204.05 |
| 小脑: Gemma Expert full | `gemma_expert`，含 unused LM head | 0.691B | 1540.68 | 1318.47 | 659.23 | 329.62 |

当前推理 forward 实际使用的是 `gemma_expert.model`，不是完整 `GemmaForCausalLM` 的 `lm_head`。因此“小脑”用于量化收益估算时应优先看 `gemma_expert.model` 的 0.428B 参数，而不是含 unused LM head 的 0.691B。

### 小脑内部拆分

| Expert 子模块 | 参数量 | BF16 MiB | INT8 MiB | INT4 MiB |
| --- | ---: | ---: | ---: | ---: |
| MLP | 226.49M | 432.00 | 216.00 | 108.00 |
| Attention | 84.93M | 162.00 | 81.00 | 40.50 |
| Norm / AdaRMS | 113.36M | 216.21 | 108.11 | 54.05 |
| Other | 3.15M | 6.01 | 3.00 | 1.50 |

本次实验只量化 `expert_mlp`，所以理论上最多直接影响约 432 MiB 的 BF16 权重。W8 weight-only 将这部分权重降到约 216 MiB，与实测 inference peak allocated 约 215 MiB 的下降量基本吻合。

### Action adapter 参数量

| 模块 | 参数量 | actual MiB |
| --- | ---: | ---: |
| `action_in_proj` | 33.79K | 0.13 |
| `time_mlp_in` | 1.05M | 4.00 |
| `time_mlp_out` | 1.05M | 4.00 |
| `action_out_proj` | 32.80K | 0.13 |

Action adapter 总量约 2.17M 参数，相比 VLM 和 Gemma Expert 很小，不是当前显存优化的主要目标。

## 已完成的工程改动

- 新增 `quant_exp/run_acceleration.py`，用于本地 checkpoint 的 BF16 baseline 与量化版本测速。
- 支持 `--opt-mode dynamic-w8a8`、`--opt-mode weight-only`、`--opt-mode weight-only-4bit`。
- 支持 `--fuse-gate-up`，将 expert MLP 中 `gate_proj` 和 `up_proj` 合并为一次 larger linear，减少 small-M 场景下的 Linear 调用次数。
- 支持 `--enable-alpha-fold`，默认关闭 attention alpha fold，符合 MLP-only MVP。
- 支持局部 `torch.compile`，编译目标从外层 `sample_actions` 收缩到 `gemma_expert`，避免编译整个 VLM/前处理链路。
- 增加 CUDA 显存统计，分别输出模型 ready 状态和推理阶段 peak memory。
- 修改 `PI0Pytorch.sample_actions()` 中的 denoise 循环，从 Tensor 条件 `while` 改为基于 `num_steps` 的 Python `for`，并将 attention implementation 的 eager 配置前移到初始化阶段，减少 compile graph break 风险。

## 关键发现

### 1. `torch.compile` 会带来 BF16 数值漂移

在不量化、不折叠 beta/alpha 的情况下：

- eager BF16 baseline vs compiled BF16 accelerated 曾出现 `max_abs ~= 0.017`。
- compiled BF16 baseline vs compiled BF16 accelerated 的 action delta 为 0。

因此，eager 与 compiled 之间的误差主要来自 Inductor/BF16 kernel、融合和归约顺序差异，而不是随机噪声或权重加载差异。后续评估应分开比较：

- eager BF16 vs eager quant：量化误差。
- compiled BF16 vs compiled quant：同后端部署误差。
- eager BF16 vs compiled BF16：compile 后端数值漂移。

### 2. 全量编译 `sample_actions` 不适合本链路

直接编译 `sample_actions` 会把图像前处理、VLM、expert transformer 和 denoise loop 一起交给 Dynamo/Inductor，编译时间长且容易 graph break。局部编译 `gemma_expert` 更合理，但在当前 small-M 和 TorchAO tensor subclass 路径下没有明显加速收益。

### 3. dynamic W8A8 不适合当前 small-M action expert

`Int8DynamicActivationInt8WeightConfig` 会对 activation 做动态 per-token quant/dequant。当前 DROID action horizon 下常见 GEMM 形状为 `M = batch * action_horizon = 15`，每个 denoise step 中 expert MLP 有大量 small-M Linear 调用。动态 activation quant 的 scale 计算、dispatch 和 dequant 开销超过 INT8 GEMM 收益。

同时，TorchAO dynamic W8A8 在 `torch.compile` 下会触发 `_int_mm` 的 small-M 限制：

```text
RuntimeError: self.size(0) needs to be greater than 16, but got 15
```

目前通过对量化 expert MLP Linear 使用 `torch.compiler.disable` 可以绕过该 runtime error，但这也意味着量化 Linear 无法被 Inductor 融合优化。

## 实验结果

所有数值均来自同一台机器上的本地测试，命令使用 `uv run python quant_exp/run_acceleration.py`，默认 `num_steps=10`，固定 seed/noise。时间会受 CUDA 缓存和热启动影响，重点看相对趋势。

### BF16 baseline

| 指标 | 数值 |
| --- | ---: |
| baseline latency | 约 190-205 ms |
| inference peak allocated | 7315.47 MiB |

### W8A8 dynamic activation + int8 weight

| 指标 | 数值 |
| --- | ---: |
| accelerated latency | 约 1010 ms |
| speedup | 约 0.19x |
| action L2 | 约 0.00156 |

结论：功能可运行，但 single-batch small-M 下明显慢于 BF16，不适合作为当前默认加速路线。

### W8 weight-only + fused gate/up

命令：

```bash
uv run python quant_exp/run_acceleration.py \
  --fuse-gate-up \
  --opt-mode weight-only \
  --warmup-runs 1 \
  --benchmark-runs 3
```

| 指标 | 数值 |
| --- | ---: |
| baseline latency | 204.83 ms |
| accelerated latency | 210.22 ms |
| speedup | 0.9743x |
| action max_abs | 0.002721 |
| action mean_abs | 0.000514 |
| action L2 | 0.000705 |
| inference peak allocated | 7100.39 MiB |
| peak allocated saving | 约 215 MiB |

结论：这是当前综合最好的版本。误差小，显存有轻微收益，但速度仍略慢于 BF16 baseline。

### W4 weight-only + fused gate/up

TorchAO 默认 int4 `PLAIN` packing 依赖 `mslk>=1.0.0`，当前环境不可用。已改用 CUDA 可运行的 `Int4PackingFormat.TILE_PACKED_TO_4D` tinygemm 路径。

命令：

```bash
uv run python quant_exp/run_acceleration.py \
  --fuse-gate-up \
  --opt-mode weight-only-4bit \
  --warmup-runs 1 \
  --benchmark-runs 3
```

| 指标 | 数值 |
| --- | ---: |
| baseline latency | 188.62 ms |
| accelerated latency | 210.33 ms |
| speedup | 0.8968x |
| action max_abs | 0.016637 |
| action mean_abs | 0.004186 |
| action L2 | 0.005552 |
| inference peak allocated | 6998.66 MiB |
| peak allocated saving | 约 317 MiB |

结论：W4 比 W8 weight-only 再省约 100 MiB 显存，但误差显著变大，速度没有改善。因此不建议作为默认版本。

### alpha fold / ATM

在当前 MVP 中 attention 不量化，因此 alpha fold 默认关闭。实验中开启 `--enable-alpha-fold` 后：

- 10-step eager 下 action L2 与 no-alpha 基本相同。
- max error 略变差。
- 1-step compiled 路径下误差更明显变大。

结论：当前 `expert_attention_alpha_step10.pt` 没有稳定收益，不应纳入默认路线。

## 当前结论

1. 当前最稳妥路线是 `expert_mlp` 的 W8 weight-only + OHB beta + gate/up fusion。
2. dynamic W8A8 在 single-batch small-M 下太慢，不适合作为当前部署候选。
3. W4 weight-only 的显存收益有限，误差代价较大，暂不推荐默认使用。
4. alpha fold 不应默认启用，因为 MVP 中 attention 保持 BF16，且当前实验未显示稳定收益。
5. `torch.compile` 应局部作用于 expert，而不是外层 `sample_actions`。但在当前 TorchAO weight-only 路径下，compile 没有带来稳定收益。

## 后续建议

- 优先保持 `--fuse-gate-up --opt-mode weight-only` 作为当前最佳实验配置。
- 若目标是速度，应继续探索 small-M 专用 kernel、fused MLP kernel、grouped GEMM 或 denoise/env batching，而不是继续依赖通用 TorchAO dynamic quant。
- 若目标是显存，应考虑扩大到 VLM/LLM 主干的 weight-only 或 W4/W8 分层策略，因为只量化 action expert MLP 的显存收益有限。
- 若目标是真实部署，应在 fake/weight-only 误差稳定后，再考虑 Triton、Marlin/AWQ/GPTQ、TensorRT-LLM 或自定义 kernel。