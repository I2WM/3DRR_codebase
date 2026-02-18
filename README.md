# Simple 3D Gaussian Splatting

A minimal 3DGS implementation built on [gsplat](https://github.com/nerfstudio-project/gsplat).

## Dependencies

- PyTorch
- gsplat
- torchvision
- omegaconf
- tqdm
- PyYAML

## Project Structure

```
simple_gs/
├── config/             # Training configs (YAML)
├── core/
│   ├── data/           # Dataset loaders (Blender format)
│   ├── libs/           # Utilities (SSIM, config, etc.)
│   └── model/          # 3DGS model
├── train.py            # Training + validation + test rendering
├── eval.py             # Load checkpoint and render test set
└── outputs/            # Training outputs
```

## Data Format

Expects datasets in Blender/NeRF-synthetic JSON format with calibrated intrinsics.


## Training

```bash
python train.py -c config/cupcake.yaml
```

Outputs are saved to `outputs/<experiment>/<timestamp>/`:
- `latest.pt` — model checkpoint
- `config.yaml` — copy of training config
- `examples/` — validation render grids (`val_step*.jpg`) and augmented training samples (`train_aug.jpg`)
- `test/` — rendered test images

## Evaluation (Render Only)

Load a checkpoint and render the test set (no metrics, no GT images needed):

```bash
python eval.py -w outputs/.../latest.pt
```

Config is automatically loaded from the checkpoint directory. Rendered images are saved to `{checkpoint_dir}/test/`.
