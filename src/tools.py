import json
import numpy as np
import imageio
import os

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from pprint import pprint
import torchvision.io as tvio


def export_to_video(video_frames, output_video_path, fps=12):
    assert all(
        isinstance(frame, np.ndarray) for frame in video_frames
    ), "All video frames must be NumPy arrays."
    if not output_video_path.endswith(".mp4"):
        output_video_path += ".mp4"
    with imageio.get_writer(output_video_path, fps=fps, format="mp4") as writer:
        for frame in video_frames:
            writer.append_data((frame * 255).astype(np.uint8))


def save_generation(video_frames, configs, base_path, file_name=None):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    p_config = configs["pipe_configs"]
    frames, steps, fps = p_config["num_frames"], p_config["steps"], p_config["fps"]
    if not file_name:
        index = [int(each.split("_")[0]) for each in os.listdir(base_path)]
        max_index = max(index) if index else 0
        idx_str = str(max_index + 1).zfill(6)

        key_info = "_".join([str(frames), str(steps), str(fps)])
        file_name = f"{idx_str}_{key_info}"

    with open(f"{base_path}/{file_name}.json", "w") as f:
        json.dump(configs, f, indent=4)

    export_to_video(
        video_frames, f"{base_path}/{file_name}.mp4", fps=p_config["export_fps"]
    )

    return file_name


class GlobalState:
    def __init__(self, state={}):
        self.init_state(state)

    def init_state(self, state={}):
        self.state = state

    def set(self, key, value):
        self.state[key] = value

    def get(self, key, default=None):
        return self.state.get(key, default)


class DistController(object):
    def __init__(self, rank, world_size, config) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.is_master = rank == 0
        self.device = xm.xla_device()  # Use XLA device
        self.init_dist()
        self.init_group()

    def init_dist(self):
        print(f"Rank {self.rank}, {self.device} is running on XLA device.")
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(self.config.get("master_port") or "29500")
        dist.init_process_group("xla", rank=self.rank, world_size=self.world_size)

    def init_group(self):
        self.adj_groups = [
            dist.new_group([i, i + 1]) for i in range(self.world_size - 1)
        ]
        pprint(f"Rank {self.rank} initialized groups: {self.adj_groups}")
