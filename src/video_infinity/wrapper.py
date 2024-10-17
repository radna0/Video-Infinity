from ..tools import save_generation, GlobalState, DistController

from .plugins import (
    torch,
    ModulePlugin,
    UNetPlugin,
    GroupNormPlugin,
    ConvLayerPlugin,
    AttentionPlugin,
    Conv3DPligin,
    dist,
)
import torch_xla.core.xla_model as xm
from accelerate.utils.operations import _tpu_gather_one
import torch
import os
import json


def save_tensors_and_config(video_frames, config, base_path, file_name=None):
    # Ensure base_path exists
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    p_config = config["pipe_configs"]
    frames, steps, fps = p_config["num_frames"], p_config["steps"], p_config["fps"]

    # Generate a file name if none is provided
    if not file_name:
        index = [int(each.split("_")[0]) for each in os.listdir(base_path)]
        max_index = max(index) if index else 0
        idx_str = str(max_index + 1).zfill(6)
        key_info = "_".join([str(frames), str(steps), str(fps)])
        file_name = f"{idx_str}_{key_info}"

    # Save the config file
    config_path = os.path.join(base_path, f"{file_name}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Save the TPU tensors to a file
    frames_file_path = os.path.join(base_path, f"{file_name}_frames.pt")
    torch.save(video_frames, frames_file_path)

    print(f"Tensors and config saved to {frames_file_path} and {config_path}")
    return frames_file_path, config_path


class DistWrapper(object):
    def __init__(self, pipe, dist_controller: DistController, config) -> None:
        super().__init__()
        self.pipe = pipe
        self.dist_controller = dist_controller
        self.config = config
        self.global_state = GlobalState({"dist_controller": dist_controller})
        self.plugin_mount()

    def switch_plugin(self, plugin_name, enable):
        if plugin_name not in self.plugins:
            return
        for moudule_id in self.plugins[plugin_name]:
            moudle: ModulePlugin = self.plugins[plugin_name][moudule_id]
            moudle.set_enable(enable)

    def config_plugin(self, plugin_name, config):
        if plugin_name not in self.plugins:
            return
        for moudule_id in self.plugins[plugin_name]:
            moudle: ModulePlugin = self.plugins[plugin_name][moudule_id]
            moudle.update_config(config)

    def plugin_mount(self):
        self.plugins = {}
        self.unet_plugin_mount()
        self.attn_plugin_mount()

        # self.group_norm_plugin_mount()
        # self.conv_3d_plugin_mount()

        # Conv3d and Conv layer can only be used one at a time
        self.conv_plugin_mount()

    def group_norm_plugin_mount(self):
        self.plugins["group_norm"] = {}
        group_norms = []
        for module in self.pipe.unet.named_modules():
            if ("temp_" in module[0] or "transformer_in" in module[0]) and module[
                1
            ].__class__.__name__ == "GroupNorm":
                group_norms.append(module[1])
        if self.dist_controller.is_master:
            print(f"Found {len(group_norms)} group norms")
        for i, group_norm in enumerate(group_norms):
            plugin_id = "group_norm", i
            self.plugins["group_norm"][plugin_id] = GroupNormPlugin(
                group_norm, plugin_id, self.global_state
            )

    def conv_plugin_mount(self):
        self.plugins["conv_layer"] = {}
        convs = []
        for module in self.pipe.unet.named_modules():
            if ("temp_" in module[0] or "transformer_in" in module[0]) and module[
                1
            ].__class__.__name__ == "TemporalConvLayer":
                convs.append(module[1])
        if self.dist_controller.is_master:
            print(f"Found {len(convs)} convs")
        for i, conv in enumerate(convs):
            plugin_id = "conv_layer", i
            self.plugins["conv_layer"][plugin_id] = ConvLayerPlugin(
                conv, plugin_id, self.global_state
            )

    def conv_3d_plugin_mount(self):
        self.plugins["conv_3d"] = {}
        conv3d_s = []
        for module in self.pipe.unet.named_modules():
            if ("temp_" in module[0] or "transformer_in" in module[0]) and module[
                1
            ].__class__.__name__ == "Conv3d":
                conv3d_s.append(module[1])
        if self.dist_controller.is_master:
            print(f"Found {len(conv3d_s)} conv3d_s")
        for i, conv in enumerate(conv3d_s):
            plugin_id = "conv_3d", i
            self.plugins["conv_3d"][plugin_id] = Conv3DPligin(
                conv, plugin_id, self.global_state
            )

    def attn_plugin_mount(self):
        self.plugins["attn"] = {}
        attns = []
        for module in self.pipe.unet.named_modules():
            if ("temp_" in module[0] or "transformer_in" in module[0]) and module[
                1
            ].__class__.__name__ == "Attention":
                attns.append(module[1])
        if self.dist_controller.is_master:
            print(f"Found {len(attns)} attns")
        for i, attn in enumerate(attns):
            plugin_id = "attn", i
            self.plugins["attn"][plugin_id] = AttentionPlugin(
                attn, plugin_id, self.global_state
            )

    def unet_plugin_mount(self):
        self.plugins["unet"] = UNetPlugin(
            self.pipe.unet, ("unet", 0), self.global_state
        )

    def inference(
        self,
        prompts="A beagle wearning diving goggles  swimming in the ocean while the camera is moving, coral reefs in the background",
        config={},
        pipe_configs={
            "steps": 50,
            "guidance_scale": 12,
            "fps": 60,
            "num_frames": 24 * 1,
            "height": 320,
            "width": 512,
            "export_fps": 12,
            "base_path": "./work/output",
            "file_name": None,
        },
        plugin_configs={
            "attn": {
                "padding": 24,
                "top_k": 24,
                "top_k_chunk_size": 24,
                "attn_scale": 1.0,
                "token_num_scale": True,
                "dynamic_scale": True,
            },
            "conv_3d": {
                "padding": 1,
            },
            "conv_layer": {},
        },
        additional_info={},
    ):
        self.plugin_mount()
        generator = torch.Generator("cpu").manual_seed(
            self.config["seed"] + self.dist_controller.rank
        )
        # generator = torch.Generator("cpu").manual_seed(self.config["seed"])

        self.global_state.set("plugin_configs", plugin_configs)

        print(f"Rank {self.dist_controller.rank} started inference.")

        video_frames = self.pipe(
            prompts,
            num_inference_steps=pipe_configs["steps"],
            guidance_scale=pipe_configs["guidance_scale"],
            height=pipe_configs["height"],
            width=pipe_configs["width"],
            num_frames=pipe_configs["num_frames"],
            fps=pipe_configs["fps"],
            generator=generator,
        ).frames[0]

        video_frames = torch.tensor(
            video_frames, dtype=torch.float16, device=self.dist_controller.device
        )

        print(
            f"Rank {self.dist_controller.rank} finished inference. Result: {video_frames.shape}"
        )
        all_frames = (
            [
                torch.zeros_like(video_frames, dtype=torch.float16)
                for _ in range(self.dist_controller.world_size)
            ]
            if self.dist_controller.is_master
            else None
        )
        if self.dist_controller.is_master:
            dist.all_gather(all_frames, video_frames)
            print(f"\n\nall_frames: {all_frames[0].shape}, {all_frames[0].device}\n\n")

            all_frames = torch.cat(all_frames, dim=0).cpu().numpy()
            save_generation(
                all_frames,
                {
                    "prompt": prompts,
                    "pipe_configs": pipe_configs,
                    "plugin_configs": plugin_configs,
                    "additional_info": additional_info,
                },
                config["base_path"],
                pipe_configs["file_name"],
            )
