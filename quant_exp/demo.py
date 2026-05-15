import argparse
import os

from openpi.policies import droid_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal OpenPI policy load + inference demo.")
    parser.add_argument("--config", default="pi05_droid", help="Training config name.")
    parser.add_argument(
        "--checkpoint",
        default="gs://openpi-assets/checkpoints/pi05_droid",
        help="Checkpoint directory or gs:// path.",
    )
    parser.add_argument("--prompt", default="pick up the fork", help="Prompt used for the random example.")
    args = parser.parse_args()

    os.environ.setdefault("OPENPI_DATA_HOME", "/root/autodl-tmp/openpi_data")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.cache/huggingface")
    os.environ.setdefault("TORCH_HOME", "/root/autodl-tmp/.cache/torch")

    config = _config.get_config(args.config)
    checkpoint_dir = download.maybe_download(args.checkpoint)

    print("creating policy...")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)


    example = droid_policy.make_droid_example()
    example["prompt"] = args.prompt

    print("running inference...")

    result = policy.infer(example)
    action_chunk = result["actions"]

    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"action shape: {action_chunk.shape}")
    print("first action:", action_chunk[0])


if __name__ == "__main__":
    main()