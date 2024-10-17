import torch
import torch_xla
from accelerate import Accelerator
import time
import json
import os
from src.video_crafter import VideoCrafterPipeline
from src.tools import DistController
from src.video_infinity.wrapper import DistWrapper


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Video Infinity Inference")
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    return args


def init_pipeline(config, device):
    # Initialize the pipeline on the TPU device
    pipe = VideoCrafterPipeline.from_pretrained(
        "adamdad/videocrafterv2_diffusers", torch_dtype=torch.float16
    )
    pipe.to(device)  # Move pipeline to the TPU device
    pipe.enable_vae_slicing()
    return pipe


def run_inference(accelerator, config):
    # Initialize accelerator and get the device
    device = accelerator.device

    # Initialize pipeline and distributed wrapper
    dist_controller = DistController(
        accelerator.local_process_index, accelerator.num_processes, config
    )
    pipe = init_pipeline(config, device)
    dist_pipe = DistWrapper(pipe, dist_controller, config)

    pipe_configs = config["pipe_configs"]
    plugin_configs = config["plugin_configs"]

    # Select prompt based on process index
    prompt_id = int(
        accelerator.local_process_index
        / accelerator.num_processes
        * len(pipe_configs["prompts"])
    )
    prompt = pipe_configs["prompts"][prompt_id]

    start = time.time()

    # Perform inference
    dist_pipe.inference(
        prompt,
        config,
        pipe_configs,
        plugin_configs,
        additional_info={"full_config": config},
    )

    accelerator.print(
        f"Process {accelerator.local_process_index} finished. Time: {time.time() - start}"
    )


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Initialize accelerator (accelerate will handle the TPU devices automatically)
    accelerator = Accelerator()

    # Ensure the output path exists
    if not os.path.exists(config["base_path"]):
        os.makedirs(config["base_path"])

    # Run the inference on TPU
    run_inference(accelerator, config)


if __name__ == "__main__":
    # Use accelerate to launch the script across TPU cores
    main()
