import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import time
import json
import os
from src.video_crafter import VideoCrafterPipeline
from src.tools import DistController
from src.video_infinity.wrapper import DistWrapper

# import processpool executor
from concurrent.futures import ProcessPoolExecutor


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


def run_inference(index, config):
    # Get the TPU device for this process
    device = torch_xla.device()

    # Initialize pipeline and distributed wrapper
    dist_controller = DistController(index, xm.xrt_world_size(), config)
    pipe = init_pipeline(config, device)
    dist_pipe = DistWrapper(pipe, dist_controller, config)

    pipe_configs = config["pipe_configs"]
    plugin_configs = config["plugin_configs"]

    # Select prompt based on process index
    prompt_id = int(index / xm.xrt_world_size() * len(pipe_configs["prompts"]))
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

    print(f"Process {index} finished. Time: {time.time() - start}")


def main(rank, args):
    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Ensure the output path exists
    if not os.path.exists(config["base_path"]):
        os.makedirs(config["base_path"])

    # Run the inference on TPU
    run_inference(rank, config)


if __name__ == "__main__":
    args = parse_args()

    # Use torch_xla multiprocessing to launch the script across TPU cores
    torch_xla.launch(main, args=(args,))
