import argparse
import logging
import math
import pathlib
import time
import types
import warnings

warnings.filterwarnings("ignore", message="Deprecation: .*", category=UserWarning, module="torchao.*")
warnings.filterwarnings(
    "ignore",
    message=".*does not implement _stable_hash_for_caching.*",
    category=UserWarning,
    module="torch._functorch.*",
)
warnings.filterwarnings(
    "ignore",
    message="Dynamo does not know how to trace the builtin `builtins.__build_class__.*",
    category=UserWarning,
    module="torch._dynamo.*",
)
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

import numpy as np
import torch
import torch.nn as nn
from torchao.quantization import Int4WeightOnlyConfig
from torchao.quantization import Int8DynamicActivationInt8WeightConfig
from torchao.quantization import Int8WeightOnlyConfig
from torchao.quantization import quantize_
from torchao.quantization.quantize_.workflows import Int4PackingFormat

from openpi.policies import policy_config
from quant_exp.bf16_baseline import DEFAULT_CACHE_ROOT
from quant_exp.bf16_baseline import OFFICIAL_INFERENCE_NUM_STEPS
from quant_exp.bf16_baseline import classify_linear_module
from quant_exp.bf16_baseline import configure_cache_env
from quant_exp.bf16_baseline import create_example
from quant_exp.bf16_baseline import create_fixed_noise
from quant_exp.bf16_baseline import load_norm_stats_from_jax_checkpoint
from quant_exp.bf16_baseline import make_bf16_train_config
from quant_exp.bf16_baseline import resolve_device
from quant_exp.bf16_baseline import summarize_model


LOCAL_CHECKPOINT_ROOT = DEFAULT_CACHE_ROOT / "openpi_data" / "openpi-assets" / "checkpoints"
LOCAL_JAX_CHECKPOINT = LOCAL_CHECKPOINT_ROOT / "pi05_droid"
LOCAL_PYTORCH_CHECKPOINT = LOCAL_CHECKPOINT_ROOT / "pi05_droid_pytorch_bf16"
DEFAULT_BETA_PATH = pathlib.Path("/root/autodl-tmp/openpi/quant_exp/artifacts/expert_mlp_beta_step10.pt")
DEFAULT_ALPHA_PATH = pathlib.Path("/root/autodl-tmp/openpi/quant_exp/artifacts/expert_attention_alpha_step10.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local BF16 vs TorchAO acceleration benchmark without downloading.")
    parser.add_argument("--config", default="pi05_droid", help="Training config name.")
    parser.add_argument("--checkpoint", default=str(LOCAL_JAX_CHECKPOINT), help="Local JAX checkpoint directory.")
    parser.add_argument(
        "--pytorch-checkpoint-dir",
        default=str(LOCAL_PYTORCH_CHECKPOINT),
        help="Local PyTorch checkpoint directory containing model.safetensors.",
    )
    parser.add_argument("--beta-path", default=str(DEFAULT_BETA_PATH), help="Path to the exported beta patch file.")
    parser.add_argument("--alpha-path", default=str(DEFAULT_ALPHA_PATH), help="Path to the exported alpha patch file.")
    parser.add_argument("--prompt", default="pick up the fork", help="Prompt used for the example input.")
    parser.add_argument("--device", default="cuda", help="Target device, for example cuda or cpu.")
    parser.add_argument("--num-steps", type=int, default=OFFICIAL_INFERENCE_NUM_STEPS, help="Inference denoising steps.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic fixed noise.")
    parser.add_argument(
        "--compile-mode",
        default="none",
        choices=["none", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="Optional torch.compile mode applied to sample_actions.",
    )
    parser.add_argument(
        "--compile-target",
        default="accelerated",
        choices=["baseline", "accelerated", "both"],
        help="Which policy to compile when --compile-mode is not 'none'.",
    )
    parser.add_argument("--warmup-runs", type=int, default=2, help="Warmup runs before timing.")
    parser.add_argument("--benchmark-runs", type=int, default=3, help="Timed runs used for averaging.")
    parser.add_argument(
        "--enable-alpha-fold",
        action="store_true",
        help="Fold alpha into attention q_proj/k_proj. Disabled by default for the MLP-only MVP.",
    )
    parser.add_argument(
        "--skip-alpha-fold",
        action="store_true",
        help="Deprecated compatibility flag. Alpha folding is skipped unless --enable-alpha-fold is set.",
    )
    parser.add_argument(
        "--skip-beta-fold",
        action="store_true",
        help="Skip folding beta into expert down_proj.",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip TorchAO INT8 replacement and only test folded weights.",
    )
    parser.add_argument(
        "--opt-mode",
        default="dynamic-w8a8",
        choices=["dynamic-w8a8", "weight-only", "weight-only-4bit"],
        help="TorchAO optimization mode for expert MLP linears.",
    )
    parser.add_argument(
        "--fuse-gate-up",
        action="store_true",
        help="Fuse expert MLP gate_proj and up_proj into one larger linear before quantization.",
    )
    return parser.parse_args()


def ensure_local_artifact(path: str | pathlib.Path, *, expect_file: bool) -> pathlib.Path:
    resolved = pathlib.Path(path).expanduser().resolve()
    if expect_file:
        if not resolved.is_file():
            raise FileNotFoundError(f"Required local file not found: {resolved}")
    else:
        if not resolved.is_dir():
            raise FileNotFoundError(f"Required local directory not found: {resolved}")
    return resolved


def load_patch_map(path: pathlib.Path, key: str) -> dict[str, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if key not in payload or not isinstance(payload[key], dict):
        raise KeyError(f"Patch file {path} does not contain a '{key}' dictionary.")
    return payload[key]


@torch.no_grad()
def fold_quantvla_patches(
    model: nn.Module,
    *,
    beta_patches: dict[str, dict[str, object]],
    alpha_patches: dict[str, dict[str, object]],
    enable_beta: bool,
    enable_alpha: bool,
) -> tuple[int, int]:
    folded_mlp = 0
    folded_attn = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if enable_beta and name in beta_patches:
            beta_val = float(beta_patches[name]["beta"])
            module.weight.data.mul_(beta_val)
            if module.bias is not None:
                module.bias.data.mul_(beta_val)
            folded_mlp += 1

        if not enable_alpha:
            continue

        parent_name = ".".join(name.split(".")[:-1])
        if parent_name not in alpha_patches:
            continue
        if not (name.endswith("q_proj") or name.endswith("k_proj")):
            continue

        alpha_val = float(alpha_patches[parent_name]["alpha"])
        sqrt_alpha = math.sqrt(alpha_val)
        module.weight.data.mul_(sqrt_alpha)
        if module.bias is not None:
            module.bias.data.mul_(sqrt_alpha)
        folded_attn += 1

    return folded_mlp, folded_attn


def quantize_expert_mlp(model: nn.Module, device: str, opt_mode: str) -> int:
    selected_names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and classify_linear_module(name) == "expert_mlp"
    }
    if opt_mode == "dynamic-w8a8":
        config = Int8DynamicActivationInt8WeightConfig(version=2)
    elif opt_mode == "weight-only":
        config = Int8WeightOnlyConfig(version=2)
    elif opt_mode == "weight-only-4bit":
        config = Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
            version=2,
        )
    else:
        raise ValueError(f"Unsupported opt_mode: {opt_mode}")
    quantize_(
        model,
        config,
        filter_fn=lambda module, fqn: isinstance(module, nn.Linear) and fqn in selected_names,
        device=device,
    )
    return len(selected_names)


@torch.no_grad()
def fuse_expert_mlp_gate_up(model: nn.Module) -> int:
    fused_count = 0
    for name, module in model.named_modules():
        if not name.startswith("paligemma_with_expert.gemma_expert.model.layers."):
            continue
        if not name.endswith(".mlp"):
            continue
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            continue
        gate_proj = module.gate_proj
        up_proj = module.up_proj
        if not isinstance(gate_proj, nn.Linear) or not isinstance(up_proj, nn.Linear):
            continue
        if gate_proj.bias is not None or up_proj.bias is not None:
            raise ValueError(f"Cannot fuse biased gate/up projections for {name}.")

        fused_linear = nn.Linear(
            gate_proj.in_features,
            gate_proj.out_features + up_proj.out_features,
            bias=False,
            device=gate_proj.weight.device,
            dtype=gate_proj.weight.dtype,
        )
        fused_linear.weight.copy_(torch.cat([gate_proj.weight, up_proj.weight], dim=0))
        module.fused_gate_up = fused_linear
        delattr(module, "gate_proj")
        delattr(module, "up_proj")

        def fused_forward(self, x):
            gate, up = self.fused_gate_up(x).chunk(2, dim=-1)
            return self.down_proj(self.act_fn(gate) * up)

        module.forward = types.MethodType(fused_forward, module)
        fused_count += 1
    return fused_count


def disable_compile_for_quantized_expert_mlp(model: nn.Module) -> int:
    disabled_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if classify_linear_module(name) != "expert_mlp":
            continue
        module.forward = torch.compiler.disable(module.forward)
        disabled_count += 1
    return disabled_count


def create_local_policy(
    *,
    config_name: str,
    checkpoint_dir: pathlib.Path,
    pytorch_checkpoint_dir: pathlib.Path,
    device: str,
    num_steps: int,
):
    train_config = make_bf16_train_config(config_name)
    norm_stats = load_norm_stats_from_jax_checkpoint(train_config, checkpoint_dir)
    policy = policy_config.create_trained_policy(
        train_config,
        pytorch_checkpoint_dir,
        norm_stats=norm_stats,
        sample_kwargs={"num_steps": num_steps},
        pytorch_device=device,
    )
    return policy, train_config


def compile_policy_expert(policy, compile_mode: str | None, label: str) -> None:
    if compile_mode is None:
        return
    expert = policy._model.paligemma_with_expert.gemma_expert
    policy._model.paligemma_with_expert.gemma_expert = torch.compile(expert, mode=compile_mode)
    print(f"compiled {label} gemma_expert with mode={compile_mode}")


def should_compile_target(compile_mode: str | None, compile_target: str, label: str) -> bool:
    return compile_mode is not None and compile_target in {label, "both"}


def is_cuda_device(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def reset_cuda_peak(device: str) -> None:
    if not is_cuda_device(device):
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(torch.device(device))


def clear_cuda_cache_and_peak(device: str) -> None:
    if not is_cuda_device(device):
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device(device))


def collect_cuda_memory(device: str) -> dict[str, float] | None:
    if not is_cuda_device(device):
        return None
    cuda_device = torch.device(device)
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated(cuda_device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(cuda_device) / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(cuda_device) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(cuda_device) / 1024**2,
    }


def print_cuda_memory(label: str, memory: dict[str, float] | None) -> None:
    if memory is None:
        return
    print(
        f"{label}: "
        f"allocated={memory['allocated_mb']:.2f} MiB, "
        f"reserved={memory['reserved_mb']:.2f} MiB, "
        f"peak_allocated={memory['peak_allocated_mb']:.2f} MiB, "
        f"peak_reserved={memory['peak_reserved_mb']:.2f} MiB"
    )


def run_policy_once(policy, example: dict[str, object], noise: np.ndarray, device: str) -> tuple[dict[str, object], float]:
    if device.startswith("cuda"):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = policy.infer(example, noise=noise)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start_event.elapsed_time(end_event))
        return result, elapsed_ms

    start_time = time.perf_counter()
    result = policy.infer(example, noise=noise)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return result, elapsed_ms


def warmup_policy(policy, example: dict[str, object], noise: np.ndarray, runs: int) -> None:
    for _ in range(runs):
        _ = policy.infer(example, noise=noise)


def benchmark_policy(
    policy,
    example: dict[str, object],
    noise: np.ndarray,
    *,
    device: str,
    runs: int,
) -> tuple[dict[str, object], float]:
    timings: list[float] = []
    last_result = None
    for _ in range(runs):
        last_result, elapsed_ms = run_policy_once(policy, example, noise, device)
        timings.append(elapsed_ms)
    assert last_result is not None
    return last_result, float(sum(timings) / len(timings))


def summarize_delta(baseline_actions: np.ndarray, accelerated_actions: np.ndarray) -> dict[str, float]:
    delta = accelerated_actions - baseline_actions
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "l2": float(np.sqrt(np.mean(np.square(delta)))),
    }


def main() -> None:
    args = parse_args()
    compile_mode = None if args.compile_mode == "none" else args.compile_mode
    configure_cache_env()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_dir = ensure_local_artifact(args.checkpoint, expect_file=False)
    pytorch_checkpoint_dir = ensure_local_artifact(args.pytorch_checkpoint_dir, expect_file=False)
    beta_path = ensure_local_artifact(args.beta_path, expect_file=True)
    alpha_path = ensure_local_artifact(args.alpha_path, expect_file=True)

    requested_device = resolve_device(args.device)
    if args.device.startswith("cuda") and not requested_device.startswith("cuda"):
        raise RuntimeError(f"Requested CUDA execution but resolved device is {requested_device}.")

    beta_patches = load_patch_map(beta_path, "beta")
    alpha_patches = load_patch_map(alpha_path, "alpha")

    print("using local checkpoint:", checkpoint_dir)
    print("using local pytorch checkpoint:", pytorch_checkpoint_dir)
    print("using beta patches:", beta_path)
    print("using alpha patches:", alpha_path)
    print("compile mode:", compile_mode)
    print("compile target:", args.compile_target if compile_mode is not None else "none")
    print("opt mode:", "none" if args.skip_quantize else args.opt_mode)
    print("fuse gate/up:", args.fuse_gate_up)
    enable_alpha_fold = args.enable_alpha_fold and not args.skip_alpha_fold
    print("alpha fold enabled:", enable_alpha_fold)

    example = create_example(args.config, args.prompt)
    fixed_noise = create_fixed_noise(make_bf16_train_config(args.config), args.seed)

    clear_cuda_cache_and_peak(requested_device)
    baseline_policy, _ = create_local_policy(
        config_name=args.config,
        checkpoint_dir=checkpoint_dir,
        pytorch_checkpoint_dir=pytorch_checkpoint_dir,
        device=requested_device,
        num_steps=args.num_steps,
    )
    if should_compile_target(compile_mode, args.compile_target, "baseline"):
        compile_policy_expert(baseline_policy, compile_mode, "baseline")
    print("baseline model summary:", summarize_model(baseline_policy))
    baseline_ready_memory = collect_cuda_memory(requested_device)

    print("\n[1/2] BF16 baseline warmup...")
    reset_cuda_peak(requested_device)
    warmup_policy(baseline_policy, example, fixed_noise, args.warmup_runs)
    baseline_result, baseline_ms = benchmark_policy(
        baseline_policy,
        example,
        fixed_noise,
        device=requested_device,
        runs=args.benchmark_runs,
    )
    baseline_actions = np.asarray(baseline_result["actions"])
    baseline_inference_memory = collect_cuda_memory(requested_device)
    del baseline_policy
    clear_cuda_cache_and_peak(requested_device)

    accelerated_policy, _ = create_local_policy(
        config_name=args.config,
        checkpoint_dir=checkpoint_dir,
        pytorch_checkpoint_dir=pytorch_checkpoint_dir,
        device=requested_device,
        num_steps=args.num_steps,
    )

    print("\n[2/2] applying patches and TorchAO...")
    fused_mlp_count = 0
    if args.fuse_gate_up:
        fused_mlp_count = fuse_expert_mlp_gate_up(accelerated_policy._model)
    print(f"fused expert mlp gate/up layers: {fused_mlp_count}")

    folded_mlp, folded_attn = fold_quantvla_patches(
        accelerated_policy._model,
        beta_patches=beta_patches,
        alpha_patches=alpha_patches,
        enable_beta=not args.skip_beta_fold,
        enable_alpha=enable_alpha_fold,
    )
    print(f"folded mlp layers: {folded_mlp}")
    print(f"folded attention projections: {folded_attn}")

    quantized_count = 0
    if not args.skip_quantize:
        quantized_count = quantize_expert_mlp(accelerated_policy._model, requested_device, args.opt_mode)
    print(f"quantized expert_mlp linears: {quantized_count}")
    disabled_compile_count = 0
    if (
        args.opt_mode == "dynamic-w8a8"
        and quantized_count > 0
        and should_compile_target(compile_mode, args.compile_target, "accelerated")
    ):
        disabled_compile_count = disable_compile_for_quantized_expert_mlp(accelerated_policy._model)
    print(f"disabled compile for quantized expert_mlp linears: {disabled_compile_count}")

    if should_compile_target(compile_mode, args.compile_target, "accelerated"):
        compile_policy_expert(accelerated_policy, compile_mode, "accelerated")
    accelerated_ready_memory = collect_cuda_memory(requested_device)

    reset_cuda_peak(requested_device)
    warmup_policy(accelerated_policy, example, fixed_noise, args.warmup_runs)
    accelerated_result, accelerated_ms = benchmark_policy(
        accelerated_policy,
        example,
        fixed_noise,
        device=requested_device,
        runs=args.benchmark_runs,
    )
    accelerated_actions = np.asarray(accelerated_result["actions"])
    accelerated_inference_memory = collect_cuda_memory(requested_device)
    delta_stats = summarize_delta(baseline_actions, accelerated_actions)

    speedup = baseline_ms / accelerated_ms if accelerated_ms > 0 else float("inf")
    print("\n=== benchmark summary ===")
    print(f"baseline_ms: {baseline_ms:.2f}")
    print(f"accelerated_ms: {accelerated_ms:.2f}")
    print(f"speedup: {speedup:.4f}x")
    print("baseline first action:", baseline_actions[0])
    print("accelerated first action:", accelerated_actions[0])
    print("action delta stats:", delta_stats)
    print("\n=== cuda memory summary ===")
    print_cuda_memory("baseline ready", baseline_ready_memory)
    print_cuda_memory("baseline inference", baseline_inference_memory)
    print_cuda_memory("accelerated ready", accelerated_ready_memory)
    print_cuda_memory("accelerated inference", accelerated_inference_memory)


if __name__ == "__main__":
    main()