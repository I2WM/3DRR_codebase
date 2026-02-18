#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import argparse
import os
import warnings

import torch
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from core.data import Blender
from core.libs import ConfigDict
from core.model import Simple3DGS


@torch.no_grad()
def evaluate(checkpoint_path, device="cuda"):
    # load config from checkpoint directory
    ckpt_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(config_path) as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    config_dict["EXP_STR"] = ""
    config_dict["TIME_STR"] = ""
    meta_cfg = ConfigDict(config_path=config_dict)
    cfg = meta_cfg.MODEL

    # load dataset (json only, no images)
    test_dataset = Blender(meta_cfg.DATASET, split="test", load_images=False)
    H, W = test_dataset._data_info["img_h"], test_dataset._data_info["img_w"]

    # build model and load checkpoint
    model = Simple3DGS(cfg, test_dataset._data_info).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for k, v in ckpt.items():
        model.splats[k] = torch.nn.Parameter(v)
    model.sh_degree = model.sh_degree_max
    model.eval()

    # render and save
    output_dir = os.path.join(ckpt_dir, "test")
    os.makedirs(output_dir, exist_ok=True)
    num_test = len(test_dataset._records_keys)
    for i in tqdm(range(num_test), desc="Rendering"):
        data = test_dataset[i]
        camtoworld = data["transforms"].to(device)
        rendered, _, _ = model(camtoworld, H, W)
        frame_name = test_dataset._records_keys[i]
        save_image(
            rendered.permute(2, 0, 1).clamp(0, 1),
            os.path.join(output_dir, f"{frame_name}.png"),
        )
    print(f"Rendered {num_test} images to {output_dir}/ | {model.num_gaussians} Gaussians")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-w", required=True, type=str)
    args = parser.parse_args()
    evaluate(args.checkpoint)
