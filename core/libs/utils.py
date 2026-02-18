#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import os
import random

os.environ["WANDB_SILENT"] = "true"

from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf


def ssim(img1, img2, window_size=11):
    """Compute SSIM between two images.

    Args:
        img1, img2: (H, W, C) float tensors in [0, 1].
    Returns:
        Scalar SSIM value (higher is better, max 1.0).
    """
    C1 = 0.01**2
    C2 = 0.03**2
    img1 = img1.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    channels = img1.shape[1]
    # Gaussian kernel
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    gauss_1d = torch.exp(-(coords**2) / (2 * 1.5**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = (gauss_1d[:, None] * gauss_1d[None, :]).expand(channels, 1, window_size, window_size)
    pad = window_size // 2
    mu1 = F.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_params(model):
    op_para_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_para_num = sum(p.numel() for p in model.parameters())
    return op_para_num, all_para_num


def pretty_dict(input_dict, indent=0):
    out_line = ""
    tab = "    "
    for key, value in input_dict.items():
        out_line += tab * indent + str(key)
        if isinstance(value, dict):
            out_line += ":\n"
            out_line += pretty_dict(value, indent + 1)
        else:
            out_line += ":" + "\t" + str(value) + "\n"
    if indent == 0:
        max_length = 0
        for line in out_line.split("\n"):
            max_length = max(max_length, len(line.split("\t")[0]))
        max_length += 4
        aligned_line = ""
        for line in out_line.split("\n"):
            if "\t" in line:
                aligned_number = max_length - len(line.split("\t")[0])
                line = line.replace("\t", aligned_number * " ")
            aligned_line += line + "\n"
        return aligned_line[:-2]
    return out_line


### CONFIG ###
def read_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} was not found.")
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def merge_a_into_b(a, b):
    # merge dict a into dict b. values in a will overwrite b.
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(b[k], dict), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v


class ConfigDict(dict):
    def __init__(self, config_path=None):
        if isinstance(config_path, str):
            # build new config
            config_dict = read_config(config_path)
            # set output path
            experiment_string = "{}_{}".format(config_dict["MODEL"]["NAME"], config_dict["DATASET"]["NAME"])
            timeInTokyo = datetime.now()
            timeInTokyo = timeInTokyo.astimezone(ZoneInfo("Asia/Tokyo"))
            time_string = timeInTokyo.strftime("%b%d_%H%M_") + "".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 5))
            config_dict["EXP_STR"] = experiment_string
            config_dict["TIME_STR"] = time_string
        elif isinstance(config_path, dict):
            config_dict = config_path
        else:
            raise ValueError("config_path must be a string or a dict")

        _dot_config = OmegaConf.create(config_dict)
        super().__init__(OmegaConf.to_container(_dot_config, resolve=True))
        self._dot_config = _dot_config
        OmegaConf.set_readonly(self._dot_config, True)

    def __getattr__(self, name):
        if name == "_dump":
            return dict(self)
        if name == "_raw_string":
            import re

            ansi_escape = re.compile(
                r"""
                \x1B  # ESC
                (?:   # 7-bit C1 Fe (except CSI)
                    [@-Z\\-_]
                |     # or [ for CSI, followed by a control sequence
                    \[
                    [0-?]*  # Parameter bytes
                    [ -/]*  # Intermediate bytes
                    [@-~]   # Final byte
                )
            """,
                re.VERBOSE,
            )
            result = "\n" + ansi_escape.sub("", pretty_dict(self))
            return result
        return getattr(self._dot_config, name)

    def __str__(self):
        return pretty_dict(self)

    def update(self, key, value):
        OmegaConf.set_readonly(self._dot_config, False)
        self._dot_config[key] = value
        self[key] = value
        OmegaConf.set_readonly(self._dot_config, True)


class ConfigDictWrapper(dict):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self._dot_config = OmegaConf.create(dict(self))
        OmegaConf.set_readonly(self._dot_config, True)

    def __getattr__(self, name):
        return getattr(self._dot_config, name)

    def __str__(self):
        return pretty_dict(self)
