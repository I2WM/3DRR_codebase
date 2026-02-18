# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import json
import os
import random

import torch
import torchvision


class Blender(torch.utils.data.Dataset):
    def __init__(self, data_cfg, split, load_images=True):
        super().__init__()
        # build path
        assert split in ["train", "val", "test"]
        self._bg_color = data_cfg.BACKGROUND_COLOR / 255.0
        self._load_images = load_images
        # load data
        self._split = split
        self._img_path_base = os.path.join(data_cfg.DATA_PATH, split)
        if split == "val":
            self._split = "test"
            self._meta_path_base = os.path.join(data_cfg.DATA_PATH, "transforms_test.json")
            self._records, self._data_info = self._load_data()
            # keep first 4 records
            self._records = {k: v for k, v in self._records.items() if k in list(self._records.keys())[:4]}
        else:
            self._meta_path_base = os.path.join(data_cfg.DATA_PATH, f"transforms_{split}.json")
            self._records, self._data_info = self._load_data()
        if load_images:
            self._pre_loading_data()
        self._records_keys = list(self._records.keys())

    def __getitem__(self, index):
        frame_key = self._records_keys[index % len(self._records_keys)]
        return self._load_one_record(self._records[frame_key])

    def __len__(self):
        return self._length

    def _load_one_record(self, record):
        one_record_data = {
            "transforms": record["transform_matrix"],
            "infos": {"frame_name": record["frame_name"]},
        }
        if self._load_images:
            if record["img_tensor"] is not None:
                image_tensor = record["img_tensor"]
            else:
                image_tensor = load_img(record["file_path"], channel=3).float() / 255.0
            one_record_data["images"] = image_tensor
        return one_record_data

    def _load_data(self):
        with open(self._meta_path_base, "rb") as f:
            json_data = json.load(f)
        meta_info = {
            "bg_color": self._bg_color,
            "img_h": int(json_data["h"]),
            "img_w": int(json_data["w"]),
            "fl_x": json_data["fl_x"],
            "fl_y": json_data["fl_y"],
            "cx": json_data["cx"],
            "cy": json_data["cy"],
        }
        records = {}
        for frame in json_data["frames"]:
            frame_name = "_".join(frame["file_path"].split("/")[-2:])
            file_path = os.path.join(self._img_path_base, frame["file_path"].split("/")[-1])
            transform_matrix = torch.tensor(frame["transform_matrix"]).float()[:3]
            records[frame_name] = {
                "frame_name": frame_name,
                "file_path": file_path,
                "img_tensor": None,
                "alpha_tensor": None,
                "transform_matrix": transform_matrix,
            }
        return records, meta_info

    def _pre_loading_data(self):
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor

        def _load_img(key, record):
            image_tensor = load_img(record["file_path"], channel=3).float() / 255.0
            image_tensor = image_tensor[:3]
            return (key, image_tensor)

        # for key, record in records.items():
        print(f"Load data [{self._split}]: [{len(self._records.items())}].")
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 16)) as executor:
            all_images = list(executor.map(lambda _rec: _load_img(_rec[0], _rec[1]), self._records.items()))
        for key, image in all_images:
            self._records[key]["img_tensor"] = image


def load_img(file_name, channel=3):
    # load image as [channel(RGB), image_height, image_width]
    if channel == 3:
        _mode = torchvision.io.ImageReadMode.RGB
    elif channel == 4:
        _mode = torchvision.io.ImageReadMode.RGB_ALPHA
    else:
        _mode = torchvision.io.ImageReadMode.GRAY
    image = torchvision.io.read_image(file_name, mode=_mode)
    assert image is not None, file_name
    return image
