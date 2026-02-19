#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import argparse
import math
import os
import random
import warnings

import gsplat
import torch
import yaml
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from core.data import Blender
from core.libs import ConfigDict, ssim
from core.model import Simple3DGS


# ====================== LOW-LIGHT ENHANCEMENT (GAMMA AUGMENT) ===================
# This helper applies a gamma correction (gamma < 1 brightens dark regions),
# used as a simple *low-light enhancement* during training.
# ================================================================================

def gamma_augment(image, gamma=0.5):
    enhanced = torch.clamp(image, 0, 1).pow(gamma)
    return enhanced

# --------------------------------------------------------------------------------


def train(config_path, device="cuda"):
    # build config
    meta_cfg = ConfigDict(config_path=config_path)
    print(meta_cfg)
    cfg = meta_cfg.MODEL

    # build output directory
    output_dir = os.path.join("outputs", meta_cfg.EXP_STR, meta_cfg.TIME_STR)
    os.makedirs(os.path.join(output_dir, "examples"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(dict(meta_cfg), f, default_flow_style=False)

    # load dataset
    train_dataset = Blender(meta_cfg.DATASET, split="train")
    val_dataset = Blender(meta_cfg.DATASET, split="val", load_images=False)
    num_train = len(train_dataset._records_keys)

    # build model
    model = Simple3DGS(cfg, train_dataset._data_info).to(device)
    print(f"Initialized {model.num_gaussians} Gaussians")

    # per-parameter optimizers (required by gsplat DefaultStrategy)
    lr_map = {
        "means": cfg.LR_MEANS,
        "quats": cfg.LR_QUATS,
        "scales": cfg.LR_SCALES,
        "opacities": cfg.LR_OPACITIES,
        "sh0": cfg.LR_SH0,
        "shN": cfg.LR_SHN,
    }
    optimizers = {}
    for name, param in model.splats.items():
        optimizers[name] = torch.optim.Adam([param], lr=lr_map[name], eps=1e-15)

    # exponential LR decay for means
    total_steps = cfg.TRAIN_TOTAL_STEP
    lr_final_factor = cfg.LR_MEANS_FINAL / cfg.LR_MEANS
    schedulers = {
        "means": torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=lr_final_factor ** (1.0 / total_steps)
        )
    }

    # densification strategy
    strategy = gsplat.DefaultStrategy(
        verbose=True,
        refine_start_iter=cfg.DENSIFY_START_STEP,
        refine_stop_iter=cfg.DENSIFY_STOP_STEP,
        refine_every=cfg.DENSIFY_INTERVAL,
        grow_grad2d=cfg.DENSIFY_GRAD_THRESH,
        reset_every=cfg.OPACITY_RESET_INTERVAL,
    )
    strategy_state = strategy.initialize_state(scene_scale=cfg.SCENE_SCALE)

    # training loop
    train_aug_images = []
    pbar = tqdm(range(total_steps))
    for step in pbar:
        # gradually increase SH degree
        if step > 0 and step % cfg.SH_UPGRADE_INTERVAL == 0:
            model.sh_degree = min(model.sh_degree + 1, model.sh_degree_max)

        # sample random training image
        data = train_dataset[random.randint(0, num_train - 1)]

        # ====================== LOW-LIGHT ENHANCEMENT ======================
        # > TODO:: change this simple implementation
        
        gt_image = gamma_augment(data["images"].to(device))  # (3, H, W)
        # ===================================================================

        camtoworld = data["transforms"].to(device)  # (3, 4)
        H, W = gt_image.shape[1], gt_image.shape[2]

        # forward
        rendered, alphas, info = model(camtoworld, H, W)

        # loss: (1 - lambda) * L1 + lambda * (1 - SSIM)
        gt_hwc = gt_image.permute(1, 2, 0)  # (H, W, 3)
        l1_loss = torch.abs(rendered - gt_hwc).mean()
        ssim_val = ssim(rendered, gt_hwc)
        loss = (1.0 - cfg.LAMBDA_SSIM) * l1_loss + cfg.LAMBDA_SSIM * (1.0 - ssim_val)

        # densification hooks
        strategy.step_pre_backward(model.splats, optimizers, strategy_state, step, info)
        loss.backward()
        strategy.step_post_backward(model.splats, optimizers, strategy_state, step, info, packed=False)

        # optimizer step
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sch in schedulers.values():
            sch.step()

        # logging
        if step % cfg.LOG_INTERVAL_STEP == 0:
            with torch.no_grad():
                mse = ((rendered - gt_hwc) ** 2).mean()
                psnr = -10.0 * math.log10(mse.clamp_min(1e-10).item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}", n_gs=model.num_gaussians)

        # collect augmented train images for visualization (only once)
        if train_aug_images is not None:
            train_aug_images.append(gt_image.clamp(0, 1))
            if len(train_aug_images) >= 4:
                grid = make_grid(train_aug_images[:4], nrow=2)
                save_image(grid, os.path.join(output_dir, "examples", "train_aug.jpg"))
                train_aug_images = None

        # validation
        if step > 0 and step % cfg.VAL_INTERVAL_STEP == 0:
            validate(model, val_dataset, step, device, output_dir)
            torch.save(model.splats.state_dict(), os.path.join(output_dir, "latest.pt"))
            print(f"Model saved to {output_dir}/latest.pt")

    # save model checkpoint
    torch.save(model.splats.state_dict(), os.path.join(output_dir, "latest.pt"))
    print(f"Model saved to {output_dir}/latest.pt")

    # run test evaluation
    test_dataset = Blender(meta_cfg.DATASET, split="test", load_images=False)
    evaluate(model, test_dataset, device, output_dir)


@torch.no_grad()
def validate(model, val_dataset, step, device, output_dir):
    model.eval()
    H, W = val_dataset._data_info["img_h"], val_dataset._data_info["img_w"]
    num_val = len(val_dataset._records_keys)
    val_images = []
    for i in range(num_val):
        data = val_dataset[i]
        camtoworld = data["transforms"].to(device)
        rendered, _, _ = model(camtoworld, H, W)
        # collect first 4 rendered images for visual spot-checking
        if i < 4:
            val_images.append(rendered.permute(2, 0, 1).clamp(0, 1))
    # save 4 views as a 2x2 grid
    if val_images:
        grid = make_grid(val_images, nrow=2)
        save_image(grid, os.path.join(output_dir, "examples", f"val_step{step}.jpg"))
    print(f"\n[Step {step}] {model.num_gaussians} Gaussians")
    model.train()


@torch.no_grad()
def evaluate(model, test_dataset, device, output_dir):
    model.eval()
    H, W = test_dataset._data_info["img_h"], test_dataset._data_info["img_w"]
    num_test = len(test_dataset._records_keys)
    for i in range(num_test):
        data = test_dataset[i]
        camtoworld = data["transforms"].to(device)
        rendered, _, _ = model(camtoworld, H, W)
        frame_name = test_dataset._records_keys[i]
        save_image(
            rendered.permute(2, 0, 1).clamp(0, 1),
            os.path.join(output_dir, "test", f"{frame_name}.png"),
        )
    print(f"Test renders saved to {output_dir}/test/")
    model.train()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", required=True, type=str)
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))

    train(args.config_path)
