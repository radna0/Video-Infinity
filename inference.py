import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import time
import json
import os
import torch.distributed as dist
from src.video_crafter import VideoCrafterPipeline, UNetVideoCrafter
from diffusers.schedulers import DPMSolverMultistepScheduler
from src.tools import DistController
from src.video_infinity.wrapper import DistWrapper
import numpy as np
from torch_xla import runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
    SpmdFullyShardedDataParallel as FSDPv2,
)

xr.use_spmd()


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


def cleanup():
    dist.destroy_process_group()


def run_inference(rank, size, config):
    # Setup TPU device based on rank, explicitly mapping TPU cores
    device = xm.xla_device()  # Map rank to TPU core

    # Initialize distributed controller and pipeline
    dist_controller = DistController(rank, size, config)
    pipe = init_pipeline(config, device)
    pipe = FSDPv2(pipe)
    dist_pipe = DistWrapper(pipe, dist_controller, config)

    pipe_configs = config["pipe_configs"]
    plugin_configs = config["plugin_configs"]

    # Determine which prompt to use for the given rank
    prompt_id = int(rank / size * len(pipe_configs["prompts"]))
    prompt = pipe_configs["prompts"][prompt_id]

    start = time.time()

    # Perform inference, distribute across TPUs
    dist_pipe.inference(
        prompt,
        config,
        pipe_configs,
        plugin_configs,
        additional_info={"full_config": config},
    )

    print(f"Rank {rank} finished. Time: {time.time() - start}")


def main(index, size, config):
    # Ensure the output path exists
    if not os.path.exists(config["base_path"]):
        os.makedirs(config["base_path"])

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    # To be noted, the mesh must have an axis named 'fsdp', which the weights and activations will be sharded on.
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)

    # Run inference for each rank
    run_inference(index, size, config)


if __name__ == "__main__":
    with open(parse_args().config, "r") as f:
        config = json.load(f)

    # Use xm.spawn to launch the multiprocessing across TPUs
    main(0,1,config)
