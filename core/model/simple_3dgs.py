# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import torch
import torch.nn as nn
from gsplat import rasterization


class Simple3DGS(nn.Module):
    def __init__(self, model_cfg, data_info):
        """
        Args:
            model_cfg: OmegaConf config with NUM_INIT_POINTS, SH_DEGREE, etc.
            data_info: dict with keys "fl_x", "fl_y", "cx", "cy", "bg_color".
        """
        super().__init__()
        self.fl_x = data_info["fl_x"]
        self.fl_y = data_info["fl_y"]
        self.cx = data_info["cx"]
        self.cy = data_info["cy"]
        self.bg_color = data_info["bg_color"]
        self.sh_degree_max = model_cfg.SH_DEGREE
        self.sh_degree = 0  # will be increased during training

        num_points = model_cfg.NUM_INIT_POINTS
        num_sh_bases = (self.sh_degree_max + 1) ** 2

        # Random initialization
        means = (torch.rand(num_points, 3) - 0.5) * 10.0  # uniform in [-3, 3]
        quats = torch.zeros(num_points, 4)
        quats[:, 0] = 1.0  # identity rotation
        scales = torch.log(torch.full((num_points, 3), 0.005))  # small initial size
        opacities = torch.logit(torch.full((num_points,), 0.1))
        sh0 = torch.zeros(num_points, 1, 3)  # DC term (~gray after SH eval + 0.5 bias)
        shN = torch.zeros(num_points, num_sh_bases - 1, 3)

        # Store as ParameterDict for gsplat strategy compatibility
        self.splats = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "quats": nn.Parameter(quats),
                "scales": nn.Parameter(scales),
                "opacities": nn.Parameter(opacities),
                "sh0": nn.Parameter(sh0),
                "shN": nn.Parameter(shN),
            }
        )

    @property
    def num_gaussians(self):
        return self.splats["means"].shape[0]

    def forward(self, camtoworld, img_h, img_w):
        """
        Render an image from the given camera pose.

        Args:
            camtoworld: (3, 4) camera-to-world transformation matrix.
            img_h: image height in pixels.
            img_w: image width in pixels.
        Returns:
            rendered: (H, W, 3) rendered RGB image.
            alphas: (H, W, 1) rendered alpha map.
            info: dict with rasterization metadata (used by densification strategy).
        """
        device = self.splats["means"].device

        # Build world-to-camera view matrix (4x4)
        # NeRF Synthetic uses OpenGL convention (camera looks down -Z),
        # gsplat expects OpenCV convention (camera looks down +Z),
        # so we flip Y and Z axes after inversion.
        c2w = torch.eye(4, device=device, dtype=torch.float32)
        c2w[:3, :] = camtoworld.to(device)
        viewmat = torch.linalg.inv(c2w)
        viewmat[1, :] *= -1  # flip Y
        viewmat[2, :] *= -1  # flip Z
        viewmat = viewmat[None]  # (1, 4, 4)

        # Build intrinsics from calibrated parameters
        K = torch.tensor(
            [
                [self.fl_x, 0.0, self.cx],
                [0.0, self.fl_y, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )[None]  # (1, 3, 3)

        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        bg = torch.full((1, 3), self.bg_color, dtype=torch.float32, device=device)

        renders, alphas, info = rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=img_w,
            height=img_h,
            sh_degree=self.sh_degree,
            backgrounds=bg,
            render_mode="RGB",
            packed=False,
        )

        # renders: (1, H, W, 3), alphas: (1, H, W, 1)
        return renders[0], alphas[0], info
