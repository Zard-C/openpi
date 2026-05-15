import argparse
import collections
import dataclasses
import os
import pathlib
import shutil
import sys

import numpy as np
import torch
from torch import nn

from openpi.policies import aloha_policy
from openpi.policies import droid_policy
from openpi.policies import libero_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.convert_jax_model_to_pytorch import convert_pi0_checkpoint


DEFAULT_CACHE_ROOT = pathlib.Path("/root/autodl-tmp")
DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_droid"
OFFICIAL_INFERENCE_NUM_STEPS = 10
LINEAR_KINDS = (
    "expert_attention",
    "expert_mlp",
    "vlm_attention",
    "vlm_mlp",
    "action_head",
    "time_mlp",
    "vision_tower",
    "other",
)
ATTENTION_KINDS = ("expert_attention", "vlm_attention", "vision_tower", "other")


@dataclasses.dataclass(frozen=True)
class FakeQuantConfig:
    weight_bits: int = 8
    activation_bits: int = 8
    quantize_weight: bool = True
    quantize_input: bool = True
    quantize_output: bool = True


@dataclasses.dataclass(frozen=True)
class AttentionFakeQuantConfig:
    query_bits: int = 8
    key_bits: int = 8
    quantize_query: bool = True
    quantize_key: bool = True


@dataclasses.dataclass(frozen=True)
class LinearModuleInfo:
    name: str
    kind: str
    in_features: int
    out_features: int
    weight_dtype: str
    weight_shape: tuple[int, ...]
    weight_absmax: float
    weight_rms: float
    weight_per_out_channel_absmax: torch.Tensor
    weight_per_out_channel_rms: torch.Tensor


@dataclasses.dataclass(frozen=True)
class AttentionModuleInfo:
    name: str
    kind: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    scaling: float


@dataclasses.dataclass
class FakeQuantRuntimeStats:
    calls: int = 0
    input_scale_last: float | None = None
    output_scale_last: float | None = None
    weight_scale_mean_last: float | None = None
    weight_scale_max_last: float | None = None
    weight_scale_per_out_channel_last: torch.Tensor | None = None

    def update_input_scale(self, scale: float) -> None:
        self.calls += 1
        self.input_scale_last = scale

    def update_output_scale(self, scale: float) -> None:
        self.output_scale_last = scale

    def update_weight_scale(self, scales: torch.Tensor) -> None:
        scales = scales.detach().cpu().to(torch.float32)
        self.weight_scale_per_out_channel_last = scales
        self.weight_scale_mean_last = float(scales.mean().item())
        self.weight_scale_max_last = float(scales.max().item())

    def export(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "input_scale_last": self.input_scale_last,
            "output_scale_last": self.output_scale_last,
            "weight_scale_mean_last": self.weight_scale_mean_last,
            "weight_scale_max_last": self.weight_scale_max_last,
            "weight_scale_per_out_channel_last": self.weight_scale_per_out_channel_last,
        }


@dataclasses.dataclass
class AttentionLogitsHookStats:
    kind: str
    calls: int = 0
    logits_std_sum: float = 0.0
    logits_mean_sum: float = 0.0
    logits_absmax: float = 0.0
    logits_shape: tuple[int, ...] | None = None
    logits_dtype: str | None = None
    per_head_std_sum: torch.Tensor | None = None
    query_scale_last: float | None = None
    key_scale_last: float | None = None

    def update(self, logits: torch.Tensor, *, query_scale: float | None, key_scale: float | None) -> None:
        values = logits.detach().to(torch.float32)
        self.calls += 1
        self.logits_std_sum += tensor_std(values)
        self.logits_mean_sum += float(values.mean().item())
        self.logits_absmax = max(self.logits_absmax, float(values.abs().amax().item()))
        self.logits_shape = tuple(int(dim) for dim in values.shape)
        self.logits_dtype = str(logits.dtype)
        per_head_std = values.permute(1, 0, 2, 3).reshape(values.shape[1], -1).std(dim=1, unbiased=False).cpu()
        if self.per_head_std_sum is None:
            self.per_head_std_sum = per_head_std
        else:
            self.per_head_std_sum += per_head_std
        self.query_scale_last = query_scale
        self.key_scale_last = key_scale

    def export(self) -> dict[str, object]:
        denom = max(self.calls, 1)
        per_head_std_mean = None if self.per_head_std_sum is None else self.per_head_std_sum / denom
        return {
            "kind": self.kind,
            "calls": self.calls,
            "logits_std_mean": self.logits_std_sum / denom,
            "logits_mean_mean": self.logits_mean_sum / denom,
            "logits_absmax": self.logits_absmax,
            "logits_shape": list(self.logits_shape) if self.logits_shape is not None else None,
            "logits_dtype": self.logits_dtype,
            "per_head_std_mean": per_head_std_mean,
            "query_scale_last": self.query_scale_last,
            "key_scale_last": self.key_scale_last,
        }


@dataclasses.dataclass
class TensorHookStats:
    calls: int = 0
    num_vectors: int = 0
    rms_sum: float = 0.0
    std_sum: float = 0.0
    absmax: float = 0.0
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    sum_per_channel: torch.Tensor | None = None
    sum_sq_per_channel: torch.Tensor | None = None
    absmax_per_channel: torch.Tensor | None = None

    def update(self, tensor: torch.Tensor) -> None:
        values = flatten_feature_tensor(tensor)
        self.calls += 1
        self.num_vectors += values.shape[0]
        self.rms_sum += tensor_rms(values)
        self.std_sum += tensor_std(values)
        self.absmax = max(self.absmax, float(values.abs().amax().item()))
        self.dtype = str(tensor.dtype)
        self.shape = tuple(int(dim) for dim in tensor.shape)

        sum_per_channel = values.sum(dim=0).cpu()
        sum_sq_per_channel = values.square().sum(dim=0).cpu()
        absmax_per_channel = values.abs().amax(dim=0).cpu()

        if self.sum_per_channel is None:
            self.sum_per_channel = sum_per_channel
            self.sum_sq_per_channel = sum_sq_per_channel
            self.absmax_per_channel = absmax_per_channel
            return

        self.sum_per_channel += sum_per_channel
        self.sum_sq_per_channel += sum_sq_per_channel
        self.absmax_per_channel = torch.maximum(self.absmax_per_channel, absmax_per_channel)

    def export(self) -> dict[str, object]:
        denom = max(self.calls, 1)
        vector_count = max(self.num_vectors, 1)
        if self.sum_per_channel is None or self.sum_sq_per_channel is None or self.absmax_per_channel is None:
            channel_mean = None
            channel_rms = None
            channel_std = None
            channel_absmax = None
        else:
            channel_mean = self.sum_per_channel / vector_count
            channel_rms = torch.sqrt((self.sum_sq_per_channel / vector_count).clamp_min(1e-8))
            channel_var = self.sum_sq_per_channel / vector_count - channel_mean.square()
            channel_std = torch.sqrt(channel_var.clamp_min(1e-8))
            channel_absmax = self.absmax_per_channel

        return {
            "calls": self.calls,
            "num_vectors": self.num_vectors,
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape is not None else None,
            "rms_mean": self.rms_sum / denom,
            "std_mean": self.std_sum / denom,
            "absmax": self.absmax,
            "channel_mean": channel_mean,
            "channel_rms": channel_rms,
            "channel_std": channel_std,
            "channel_absmax": channel_absmax,
        }


@dataclasses.dataclass
class LinearHookStats:
    kind: str
    input: TensorHookStats = dataclasses.field(default_factory=TensorHookStats)
    output: TensorHookStats = dataclasses.field(default_factory=TensorHookStats)
    fake_quant: FakeQuantRuntimeStats = dataclasses.field(default_factory=FakeQuantRuntimeStats)

    def export(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "calls": max(self.input.calls, self.output.calls),
            "input": self.input.export(),
            "output": self.output.export(),
            "fake_quant": self.fake_quant.export(),
        }


class LinearHookScaffold:
    def __init__(
        self,
        model: nn.Module,
        *,
        target_kinds: tuple[str, ...],
        fake_quant_config: FakeQuantConfig | None = None,
    ):
        self._model = model
        self._target_kinds = set(target_kinds)
        self._fake_quant_config = fake_quant_config
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._stats: dict[str, LinearHookStats] = {}
        self._module_infos = collect_linear_modules(model)
        self._original_weights: dict[str, torch.Tensor] = {}

    @property
    def stats(self) -> dict[str, LinearHookStats]:
        return self._stats

    @property
    def module_infos(self) -> list[LinearModuleInfo]:
        return self._module_infos

    def register(self) -> None:
        for name, module in self._model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            kind = classify_linear_module(name)
            if kind not in self._target_kinds:
                continue
            self._stats[name] = LinearHookStats(kind=kind)
            self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
            self._handles.append(module.register_forward_hook(self._make_post_hook(name)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def export(self) -> dict[str, dict[str, float | int | str]]:
        return {name: stats.export() for name, stats in self._stats.items()}

    def _make_pre_hook(self, module_name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...] | None:
            if not inputs:
                return None
            tensor = inputs[0]
            if not isinstance(tensor, torch.Tensor):
                return None

            updated_inputs = list(inputs)

            if self._fake_quant_config is not None:
                if self._fake_quant_config.quantize_weight and isinstance(module, nn.Linear):
                    self._original_weights[module_name] = module.weight.detach().clone()
                    fake_weight, weight_scales = fake_quantize_linear_weight(
                        module.weight,
                        num_bits=self._fake_quant_config.weight_bits,
                    )
                    module.weight.data.copy_(fake_weight)
                    self._stats[module_name].fake_quant.update_weight_scale(weight_scales)

                if self._fake_quant_config.quantize_input:
                    tensor, input_scale = fake_quantize_activation(
                        tensor,
                        num_bits=self._fake_quant_config.activation_bits,
                    )
                    updated_inputs[0] = tensor
                    self._stats[module_name].fake_quant.update_input_scale(input_scale)

            with torch.no_grad():
                self._stats[module_name].input.update(tensor)

            return tuple(updated_inputs)

        return hook

    def _make_post_hook(self, module_name: str):
        def hook(module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
            if not isinstance(output, torch.Tensor):
                return output

            if self._fake_quant_config is not None and self._fake_quant_config.quantize_weight:
                original_weight = self._original_weights.pop(module_name, None)
                if original_weight is not None and isinstance(module, nn.Linear):
                    module.weight.data.copy_(original_weight)

            if self._fake_quant_config is not None and self._fake_quant_config.quantize_output:
                output, output_scale = fake_quantize_activation(
                    output,
                    num_bits=self._fake_quant_config.activation_bits,
                )
                self._stats[module_name].fake_quant.update_output_scale(output_scale)

            with torch.no_grad():
                self._stats[module_name].output.update(output)

            return output

        return hook


def configure_cache_env() -> None:
    os.environ.setdefault("OPENPI_DATA_HOME", str(DEFAULT_CACHE_ROOT / "openpi_data"))
    os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_CACHE_ROOT / ".cache" / "huggingface" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(DEFAULT_CACHE_ROOT / ".cache" / "torch"))


def tensor_rms(tensor: torch.Tensor, eps: float = 1e-8) -> float:
    with torch.no_grad():
        value = tensor.detach().to(torch.float32)
        return float(torch.sqrt(torch.mean(value.square()) + eps).item())


def tensor_std(tensor: torch.Tensor, eps: float = 1e-8) -> float:
    with torch.no_grad():
        value = tensor.detach().to(torch.float32)
        return float(torch.sqrt(torch.var(value, unbiased=False) + eps).item())


def flatten_feature_tensor(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().to(torch.float32)
    if value.ndim == 0:
        return value.reshape(1, 1)
    if value.ndim == 1:
        return value.reshape(1, -1)
    return value.reshape(-1, value.shape[-1])


def signed_qrange(num_bits: int) -> tuple[int, int]:
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -qmax
    return qmin, qmax


def fake_quantize_activation(tensor: torch.Tensor, *, num_bits: int) -> tuple[torch.Tensor, float]:
    value = tensor.detach().to(torch.float32)
    qmin, qmax = signed_qrange(num_bits)
    scale = value.abs().amax().clamp_min(1e-8) / float(qmax)
    quantized = torch.clamp(torch.round(value / scale), qmin, qmax)
    dequantized = quantized * scale
    return dequantized.to(dtype=tensor.dtype), float(scale.item())


def fake_quantize_linear_weight(weight: torch.Tensor, *, num_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    value = weight.detach().to(torch.float32)
    qmin, qmax = signed_qrange(num_bits)
    scales = value.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / float(qmax)
    quantized = torch.clamp(torch.round(value / scales), qmin, qmax)
    dequantized = quantized * scales
    return dequantized.to(dtype=weight.dtype), scales.squeeze(1)


class AttentionLogitsScaffold:
    def __init__(
        self,
        model: nn.Module,
        *,
        target_kinds: tuple[str, ...],
        fake_quant_config: AttentionFakeQuantConfig | None = None,
    ):
        self._model = model
        self._target_kinds = set(target_kinds)
        self._fake_quant_config = fake_quant_config
        self._module_infos = collect_attention_modules(model)
        self._module_name_by_id = {id(module): name for name, module in model.named_modules()}
        self._stats = {
            info.name: AttentionLogitsHookStats(kind=info.kind)
            for info in self._module_infos
            if info.kind in self._target_kinds
        }
        self._registered = False
        self._original_attention_forward = None

    @property
    def module_infos(self) -> list[AttentionModuleInfo]:
        return self._module_infos

    def register(self) -> None:
        if self._registered:
            return
        from transformers.models.gemma import modeling_gemma as gemma_modeling

        self._original_attention_forward = gemma_modeling.eager_attention_forward

        def wrapped_eager_attention_forward(
            module: nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None,
            scaling: float,
            dropout: float = 0.0,
            **kwargs,
        ):
            module_name = self._module_name_by_id.get(id(module))
            module_kind = classify_attention_module(module_name) if module_name is not None else "other"
            use_target = module_name in self._stats and module_kind in self._target_kinds

            working_query = query
            working_key = key
            query_scale = None
            key_scale = None
            if use_target and self._fake_quant_config is not None:
                if self._fake_quant_config.quantize_query:
                    working_query, query_scale = fake_quantize_activation(
                        working_query,
                        num_bits=self._fake_quant_config.query_bits,
                    )
                if self._fake_quant_config.quantize_key:
                    working_key, key_scale = fake_quantize_activation(
                        working_key,
                        num_bits=self._fake_quant_config.key_bits,
                    )

            key_states = gemma_modeling.repeat_kv(working_key, module.num_key_value_groups)
            value_states = gemma_modeling.repeat_kv(value, module.num_key_value_groups)
            raw_attn_logits = torch.matmul(working_query, key_states.transpose(2, 3)) * scaling
            attn_logits = raw_attn_logits
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_logits = attn_logits + causal_mask

            if use_target:
                self._stats[module_name].update(raw_attn_logits, query_scale=query_scale, key_scale=key_scale)

            attn_weights = nn.functional.softmax(attn_logits, dim=-1, dtype=torch.float32).to(working_query.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights

        gemma_modeling.eager_attention_forward = wrapped_eager_attention_forward
        self._registered = True

    def remove(self) -> None:
        if not self._registered:
            return
        from transformers.models.gemma import modeling_gemma as gemma_modeling

        gemma_modeling.eager_attention_forward = self._original_attention_forward
        self._registered = False

    def export(self) -> dict[str, dict[str, object]]:
        return {name: stats.export() for name, stats in self._stats.items()}


def classify_attention_module(module_name: str | None) -> str:
    if module_name is None:
        return "other"
    if module_name.startswith("paligemma_with_expert.gemma_expert.model.layers."):
        return "expert_attention"
    if module_name.startswith("paligemma_with_expert.paligemma.model.language_model.layers."):
        return "vlm_attention"
    if "vision_tower" in module_name:
        return "vision_tower"
    return "other"


def collect_attention_modules(model: nn.Module) -> list[AttentionModuleInfo]:
    modules: list[AttentionModuleInfo] = []
    for name, module in model.named_modules():
        if not hasattr(module, "q_proj") or not hasattr(module, "k_proj") or not hasattr(module, "scaling"):
            continue
        if not hasattr(module, "num_key_value_groups") or not hasattr(module, "head_dim"):
            continue
        modules.append(
            AttentionModuleInfo(
                name=name,
                kind=classify_attention_module(name),
                num_attention_heads=int(getattr(module, "num_attention_heads", getattr(module, "num_heads", 0))),
                num_key_value_heads=int(getattr(module, "num_key_value_heads", 0) or getattr(module, "num_kv_heads", 0) or 0),
                head_dim=int(module.head_dim),
                scaling=float(module.scaling),
            )
        )
    return modules


def print_attention_summary(attention_scaffold: AttentionLogitsScaffold, *, max_rows: int = 16) -> None:
    print("attention logits summary:")
    sorted_stats = sorted(
        attention_scaffold.export().items(),
        key=lambda item: float(item[1]["logits_std_mean"]),
        reverse=True,
    )
    for module_name, stats in sorted_stats[:max_rows]:
        print(
            "  "
            f"{stats['kind']:16s} {module_name} "
            f"calls={stats['calls']} "
            f"logits_std={float(stats['logits_std_mean']):.5f} "
            f"logits_absmax={float(stats['logits_absmax']):.5f} "
            f"q_scale={float(stats['query_scale_last'] or 0.0):.6f} "
            f"k_scale={float(stats['key_scale_last'] or 0.0):.6f}"
        )


def classify_linear_module(module_name: str) -> str:
    if module_name in {"action_in_proj", "action_out_proj", "state_proj"}:
        return "action_head"
    if module_name in {"time_mlp_in", "time_mlp_out", "action_time_mlp_in", "action_time_mlp_out"}:
        return "time_mlp"
    if module_name.startswith("paligemma_with_expert.gemma_expert.model.layers."):
        if ".self_attn." in module_name:
            return "expert_attention"
        if ".mlp." in module_name:
            return "expert_mlp"
    if module_name.startswith("paligemma_with_expert.paligemma.model.language_model.layers."):
        if ".self_attn." in module_name:
            return "vlm_attention"
        if ".mlp." in module_name:
            return "vlm_mlp"
    if "vision_tower" in module_name or "multi_modal_projector" in module_name:
        return "vision_tower"
    return "other"


def summarize_linear_weight(module: nn.Linear) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    weight = module.weight.detach().to(torch.float32)
    per_out_channel_absmax = weight.abs().amax(dim=1).cpu()
    per_out_channel_rms = torch.sqrt(weight.square().mean(dim=1).clamp_min(1e-8)).cpu()
    return (
        float(weight.abs().amax().item()),
        float(torch.sqrt(weight.square().mean().clamp_min(1e-8)).item()),
        per_out_channel_absmax,
        per_out_channel_rms,
    )


def collect_linear_modules(model: nn.Module) -> list[LinearModuleInfo]:
    modules: list[LinearModuleInfo] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight_absmax, weight_rms, weight_per_out_channel_absmax, weight_per_out_channel_rms = summarize_linear_weight(
            module
        )
        modules.append(
            LinearModuleInfo(
                name=name,
                kind=classify_linear_module(name),
                in_features=module.in_features,
                out_features=module.out_features,
                weight_dtype=str(module.weight.dtype),
                weight_shape=tuple(int(dim) for dim in module.weight.shape),
                weight_absmax=weight_absmax,
                weight_rms=weight_rms,
                weight_per_out_channel_absmax=weight_per_out_channel_absmax,
                weight_per_out_channel_rms=weight_per_out_channel_rms,
            )
        )
    return modules


def summarize_linear_modules(model: nn.Module) -> collections.Counter[str]:
    return collections.Counter(info.kind for info in collect_linear_modules(model))


def parse_hook_target_kinds(value: str) -> tuple[str, ...]:
    if value == "all":
        return LINEAR_KINDS
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(parsed) - set(LINEAR_KINDS))
    if invalid:
        raise ValueError(f"Unsupported hook kinds: {invalid}. Valid kinds: {LINEAR_KINDS}")
    return parsed


def print_linear_summary(model: nn.Module, *, max_rows: int = 16) -> None:
    modules = collect_linear_modules(model)
    counts = summarize_linear_modules(model)
    print("linear module counts:")
    for kind in LINEAR_KINDS:
        print(f"  {kind}: {counts.get(kind, 0)}")

    print("sample linear modules:")
    for info in modules[:max_rows]:
        print(
            f"  {info.kind:16s} {info.name} [{info.in_features} -> {info.out_features}] {info.weight_dtype}"
        )


def print_hook_summary(hook_scaffold: LinearHookScaffold, *, max_rows: int = 16) -> None:
    print("hook stats summary:")
    sorted_stats = sorted(
        hook_scaffold.export().items(),
        key=lambda item: float(item[1]["output"]["absmax"]),
        reverse=True,
    )
    for module_name, stats in sorted_stats[:max_rows]:
        print(
            "  "
            f"{stats['kind']:16s} {module_name} "
            f"calls={stats['calls']} "
            f"in_rms={float(stats['input']['rms_mean']):.5f} "
            f"out_rms={float(stats['output']['rms_mean']):.5f} "
            f"in_absmax={float(stats['input']['absmax']):.5f} "
            f"out_absmax={float(stats['output']['absmax']):.5f} "
            f"w_scale={float(stats['fake_quant']['weight_scale_mean_last'] or 0.0):.6f}"
        )


def export_module_infos(module_infos: list[LinearModuleInfo]) -> dict[str, dict[str, object]]:
    exported = {}
    for info in module_infos:
        exported[info.name] = {
            "name": info.name,
            "kind": info.kind,
            "in_features": info.in_features,
            "out_features": info.out_features,
            "weight_dtype": info.weight_dtype,
            "weight_shape": list(info.weight_shape),
            "weight_absmax": info.weight_absmax,
            "weight_rms": info.weight_rms,
            "weight_per_out_channel_absmax": info.weight_per_out_channel_absmax,
            "weight_per_out_channel_rms": info.weight_per_out_channel_rms,
        }
    return exported


def build_kind_aggregates(
    hook_stats: dict[str, dict[str, object]], module_infos: dict[str, dict[str, object]]
) -> dict[str, dict[str, float | int]]:
    aggregates: dict[str, dict[str, float | int]] = {}
    for kind in LINEAR_KINDS:
        selected = [(name, stats) for name, stats in hook_stats.items() if stats["kind"] == kind]
        if not selected:
            continue
        module_count = len(selected)
        calls_total = sum(int(stats["calls"]) for _, stats in selected)
        input_rms_mean = sum(float(stats["input"]["rms_mean"]) for _, stats in selected) / module_count
        output_rms_mean = sum(float(stats["output"]["rms_mean"]) for _, stats in selected) / module_count
        input_absmax_mean = sum(float(stats["input"]["absmax"]) for _, stats in selected) / module_count
        output_absmax_mean = sum(float(stats["output"]["absmax"]) for _, stats in selected) / module_count
        weight_absmax_mean = sum(float(module_infos[name]["weight_absmax"]) for name, _ in selected) / module_count
        weight_rms_mean = sum(float(module_infos[name]["weight_rms"]) for name, _ in selected) / module_count
        aggregates[kind] = {
            "module_count": module_count,
            "calls_total": calls_total,
            "input_rms_mean": input_rms_mean,
            "output_rms_mean": output_rms_mean,
            "input_absmax_mean": input_absmax_mean,
            "output_absmax_mean": output_absmax_mean,
            "weight_absmax_mean": weight_absmax_mean,
            "weight_rms_mean": weight_rms_mean,
        }
    return aggregates


def export_attention_module_infos(module_infos: list[AttentionModuleInfo]) -> dict[str, dict[str, object]]:
    exported = {}
    for info in module_infos:
        exported[info.name] = {
            "name": info.name,
            "kind": info.kind,
            "num_attention_heads": info.num_attention_heads,
            "num_key_value_heads": info.num_key_value_heads,
            "head_dim": info.head_dim,
            "scaling": info.scaling,
        }
    return exported


def build_attention_kind_aggregates(attention_stats: dict[str, dict[str, object]]) -> dict[str, dict[str, float | int]]:
    aggregates: dict[str, dict[str, float | int]] = {}
    for kind in ATTENTION_KINDS:
        selected = [stats for stats in attention_stats.values() if stats["kind"] == kind]
        if not selected:
            continue
        count = len(selected)
        aggregates[kind] = {
            "module_count": count,
            "calls_total": sum(int(stats["calls"]) for stats in selected),
            "logits_std_mean": sum(float(stats["logits_std_mean"]) for stats in selected) / count,
            "logits_absmax_mean": sum(float(stats["logits_absmax"]) for stats in selected) / count,
        }
    return aggregates


def save_hook_summary(
    hook_scaffold: LinearHookScaffold | None,
    output_path: str,
    metadata: dict[str, object],
    *,
    attention_scaffold: AttentionLogitsScaffold | None = None,
) -> None:
    module_infos = {} if hook_scaffold is None else export_module_infos(hook_scaffold.module_infos)
    hook_stats = {} if hook_scaffold is None else hook_scaffold.export()
    attention_module_infos = {} if attention_scaffold is None else export_attention_module_infos(attention_scaffold.module_infos)
    attention_stats = {} if attention_scaffold is None else attention_scaffold.export()
    output = {
        "meta": metadata,
        "module_infos": module_infos,
        "hook_stats": hook_stats,
        "attention_module_infos": attention_module_infos,
        "attention_stats": attention_stats,
        "aggregates": {
            "by_kind": build_kind_aggregates(hook_stats, module_infos),
            "attention_by_kind": build_attention_kind_aggregates(attention_stats),
        },
    }
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_file)


def make_bf16_train_config(config_name: str) -> _config.TrainConfig:
    train_config = _config.get_config(config_name)
    model_config = train_config.model
    model_config = dataclasses.replace(model_config, dtype="bfloat16", pytorch_compile_mode=None)
    return dataclasses.replace(train_config, model=model_config)


def default_pytorch_checkpoint_dir(jax_checkpoint_dir: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(f"{jax_checkpoint_dir}_pytorch_bf16")


def copy_assets_if_needed(source_checkpoint_dir: pathlib.Path, target_checkpoint_dir: pathlib.Path) -> None:
    source_assets = source_checkpoint_dir / "assets"
    target_assets = target_checkpoint_dir / "assets"
    if not source_assets.exists() or target_assets.exists():
        return
    shutil.copytree(source_assets, target_assets)


def ensure_pytorch_checkpoint(
    config_name: str,
    checkpoint: str,
    *,
    pytorch_checkpoint_dir: str | None = None,
    force_convert: bool = False,
) -> tuple[pathlib.Path, pathlib.Path]:
    jax_checkpoint_dir = download.maybe_download(checkpoint)
    jax_checkpoint_dir = pathlib.Path(jax_checkpoint_dir)

    if (jax_checkpoint_dir / "model.safetensors").exists():
        return jax_checkpoint_dir, jax_checkpoint_dir

    target_dir = pathlib.Path(pytorch_checkpoint_dir) if pytorch_checkpoint_dir else default_pytorch_checkpoint_dir(
        jax_checkpoint_dir
    )
    target_dir = target_dir.resolve()

    if force_convert or not (target_dir / "model.safetensors").exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        train_config = make_bf16_train_config(config_name)
        convert_pi0_checkpoint(
            checkpoint_dir=str(jax_checkpoint_dir),
            precision="bfloat16",
            output_path=str(target_dir),
            model_config=train_config.model,
        )

    copy_assets_if_needed(jax_checkpoint_dir, target_dir)
    return jax_checkpoint_dir, target_dir


def load_norm_stats_from_jax_checkpoint(
    train_config: _config.TrainConfig, jax_checkpoint_dir: pathlib.Path
):
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        return None
    return _checkpoints.load_norm_stats(jax_checkpoint_dir / "assets", data_config.asset_id)


def create_example(config_name: str, prompt: str) -> dict:
    if "droid" in config_name:
        example = droid_policy.make_droid_example()
    elif "libero" in config_name:
        example = libero_policy.make_libero_example()
    elif "aloha" in config_name:
        example = aloha_policy.make_aloha_example()
    else:
        raise ValueError(f"Unsupported config for built-in example generation: {config_name}")

    example["prompt"] = prompt
    return example


def create_fixed_noise(train_config: _config.TrainConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(
        (train_config.model.action_horizon, train_config.model.action_dim),
        dtype=np.float32,
    )


def resolve_device(requested_device: str | None) -> str:
    if requested_device is not None:
        return requested_device

    if not torch.cuda.is_available():
        return "cpu"

    capability = torch.cuda.get_device_capability()
    arch = f"sm_{capability[0]}{capability[1]}"
    if arch not in torch.cuda.get_arch_list():
        print(
            f"CUDA arch {arch} is not supported by the current PyTorch build; falling back to CPU for the BF16 baseline."
        )
        return "cpu"

    return "cuda"


def create_bf16_policy(
    config_name: str,
    checkpoint: str,
    *,
    pytorch_checkpoint_dir: str | None = None,
    force_convert: bool = False,
    device: str | None = None,
    num_steps: int = 10,
) -> tuple[_policy.Policy, pathlib.Path, pathlib.Path]:
    train_config = make_bf16_train_config(config_name)
    jax_checkpoint_dir, resolved_pytorch_checkpoint_dir = ensure_pytorch_checkpoint(
        config_name,
        checkpoint,
        pytorch_checkpoint_dir=pytorch_checkpoint_dir,
        force_convert=force_convert,
    )
    norm_stats = load_norm_stats_from_jax_checkpoint(train_config, jax_checkpoint_dir)
    resolved_device = resolve_device(device)
    policy = policy_config.create_trained_policy(
        train_config,
        resolved_pytorch_checkpoint_dir,
        norm_stats=norm_stats,
        sample_kwargs={"num_steps": num_steps},
        pytorch_device=resolved_device,
    )
    return policy, jax_checkpoint_dir, resolved_pytorch_checkpoint_dir


def summarize_model(policy: _policy.Policy) -> dict[str, str]:
    model = policy._model
    first_language_q_proj = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight
    first_expert_mlp = model.paligemma_with_expert.gemma_expert.model.layers[0].mlp.up_proj.weight
    return {
        "device": str(next(model.parameters()).device),
        "language_q_proj_dtype": str(first_language_q_proj.dtype),
        "expert_mlp_dtype": str(first_expert_mlp.dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BF16 PyTorch baseline for OpenPI quantization experiments.")
    parser.add_argument("--config", default="pi05_droid", help="Training config name.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="JAX checkpoint path or gs:// path.")
    parser.add_argument(
        "--pytorch-checkpoint-dir",
        default=None,
        help="Directory containing model.safetensors. If omitted, convert the JAX checkpoint on demand.",
    )
    parser.add_argument("--prompt", default="pick up the fork", help="Prompt used for the example input.")
    parser.add_argument("--device", default=None, help="PyTorch device, for example cuda or cuda:0.")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=OFFICIAL_INFERENCE_NUM_STEPS,
        help="Number of denoising steps. Defaults to the official inference setting.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for the example and fixed noise.")
    parser.add_argument(
        "--fake-quant",
        action="store_true",
        help="Enable W/A fake quantization on the selected linear kinds.",
    )
    parser.add_argument("--weight-bits", type=int, default=8, help="Weight bit-width used for fake quant.")
    parser.add_argument(
        "--activation-bits",
        type=int,
        default=8,
        help="Activation bit-width used for fake quant.",
    )
    parser.add_argument(
        "--attach-attention-hooks",
        action="store_true",
        help="Collect attention logits statistics by patching eager_attention_forward.",
    )
    parser.add_argument(
        "--attention-hook-target-kinds",
        default="expert_attention",
        help="Comma-separated attention kinds to trace, or 'all'.",
    )
    parser.add_argument(
        "--fake-quant-attention",
        action="store_true",
        help="Apply A8 fake quantization to Q/K inside the selected attention modules.",
    )
    parser.add_argument("--query-bits", type=int, default=8, help="Query activation bit-width for attention fake quant.")
    parser.add_argument("--key-bits", type=int, default=8, help="Key activation bit-width for attention fake quant.")
    parser.add_argument(
        "--print-linear-summary",
        action="store_true",
        help="Print nn.Linear module classification summary before inference.",
    )
    parser.add_argument(
        "--attach-hooks",
        action="store_true",
        help="Attach forward hooks to selected linear layers and collect activation statistics.",
    )
    parser.add_argument(
        "--hook-target-kinds",
        default="expert_mlp",
        help="Comma-separated linear kinds to hook, or 'all'.",
    )
    parser.add_argument(
        "--max-summary-rows",
        type=int,
        default=16,
        help="Maximum number of rows printed in module or hook summaries.",
    )
    parser.add_argument(
        "--hook-stats-output",
        default=None,
        help="Optional .pt path used to save collected hook statistics.",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Force regeneration of the PyTorch checkpoint even if model.safetensors already exists.",
    )
    args = parser.parse_args()

    configure_cache_env()
    np.random.seed(args.seed)

    policy, jax_checkpoint_dir, pytorch_checkpoint_dir = create_bf16_policy(
        args.config,
        args.checkpoint,
        pytorch_checkpoint_dir=args.pytorch_checkpoint_dir,
        force_convert=args.force_convert,
        device=args.device,
        num_steps=args.num_steps,
    )

    if args.print_linear_summary:
        print_linear_summary(policy._model, max_rows=args.max_summary_rows)

    hook_scaffold = None
    if args.attach_hooks or args.fake_quant:
        hook_scaffold = LinearHookScaffold(
            policy._model,
            target_kinds=parse_hook_target_kinds(args.hook_target_kinds),
            fake_quant_config=(
                FakeQuantConfig(
                    weight_bits=args.weight_bits,
                    activation_bits=args.activation_bits,
                )
                if args.fake_quant
                else None
            ),
        )
        hook_scaffold.register()

    attention_scaffold = None
    if args.attach_attention_hooks or args.fake_quant_attention:
        attention_scaffold = AttentionLogitsScaffold(
            policy._model,
            target_kinds=parse_hook_target_kinds(args.attention_hook_target_kinds),
            fake_quant_config=(
                AttentionFakeQuantConfig(
                    query_bits=args.query_bits,
                    key_bits=args.key_bits,
                )
                if args.fake_quant_attention
                else None
            ),
        )
        attention_scaffold.register()

    example = create_example(args.config, args.prompt)
    fixed_noise = create_fixed_noise(make_bf16_train_config(args.config), args.seed)
    result = policy.infer(example, noise=fixed_noise)
    model_summary = summarize_model(policy)

    if hook_scaffold is not None:
        hook_scaffold.remove()
    if attention_scaffold is not None:
        attention_scaffold.remove()

    print(f"jax_checkpoint_dir: {jax_checkpoint_dir}")
    print(f"pytorch_checkpoint_dir: {pytorch_checkpoint_dir}")
    print(f"device: {model_summary['device']}")
    print(f"language_q_proj_dtype: {model_summary['language_q_proj_dtype']}")
    print(f"expert_mlp_dtype: {model_summary['expert_mlp_dtype']}")
    print(f"action shape: {result['actions'].shape}")
    print("first action:", result["actions"][0])
    print("policy_timing:", result["policy_timing"])

    if hook_scaffold is not None:
        print_hook_summary(hook_scaffold, max_rows=args.max_summary_rows)
    if attention_scaffold is not None:
        print_attention_summary(attention_scaffold, max_rows=args.max_summary_rows)
    if args.hook_stats_output is not None and (hook_scaffold is not None or attention_scaffold is not None):
        save_hook_summary(
            hook_scaffold,
            args.hook_stats_output,
            metadata={
                "config": args.config,
                "checkpoint": args.checkpoint,
                "jax_checkpoint_dir": str(jax_checkpoint_dir),
                "pytorch_checkpoint_dir": str(pytorch_checkpoint_dir),
                "device": model_summary["device"],
                "num_steps": args.num_steps,
                "seed": args.seed,
                "prompt": args.prompt,
                "fake_quant": {
                    "enabled": args.fake_quant,
                    "weight_bits": args.weight_bits,
                    "activation_bits": args.activation_bits,
                    "target_kinds": list(parse_hook_target_kinds(args.hook_target_kinds)),
                },
                "fake_quant_attention": {
                    "enabled": args.fake_quant_attention,
                    "query_bits": args.query_bits,
                    "key_bits": args.key_bits,
                    "target_kinds": list(parse_hook_target_kinds(args.attention_hook_target_kinds)),
                },
            },
            attention_scaffold=attention_scaffold,
        )
        print(f"hook_stats_output: {args.hook_stats_output}")


if __name__ == "__main__":
    main()