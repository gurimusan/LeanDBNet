import csv
import os
from copy import deepcopy
from typing import Callable
from pathlib import Path

from PIL import Image

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


ICDAR_MEAN = np.array([0.485, 0.456, 0.406])
ICDAR_STD = np.array([0.229, 0.224, 0.225])


class ICDAR2015Dataset(Dataset):
    """`ICDAR2015 <https://rrc.cvc.uab.es/?ch=4&com=introduction>`_ Dataset.

    train:
        `images <https://rrc.cvc.uab.es/?com=downloads&action=download&ch=4&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy8/Y29tPWRvd25sb2FkcyZhY3Rpb249ZG93bmxvYWQmZmlsZT1jaDRfdHJhaW5pbmdfaW1hZ2VzLnppcA==>`
        `gts <https://rrc.cvc.uab.es/?com=downloads&action=download&ch=4&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy8/Y29tPWRvd25sb2FkcyZhY3Rpb249ZG93bmxvYWQmZmlsZT1jaDRfdHJhaW5pbmdfbG9jYWxpemF0aW9uX3RyYW5zY3JpcHRpb25fZ3Quemlw>`
    test:
        `images <https://rrc.cvc.uab.es/?com=downloads&action=download&ch=4&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy8/Y29tPWRvd25sb2FkcyZhY3Rpb249ZG93bmxvYWQmZmlsZT1jaDRfdGVzdF9pbWFnZXMuemlw>`
        `gts <https://rrc.cvc.uab.es/?com=downloads&action=download&ch=4&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9kb3dubG9hZHMvQ2hhbGxlbmdlNF9UZXN0X1Rhc2sxX0dULnppcA==>`
    """

    def __init__(
        self,
        img_root: list[str|Path],
        gt_root: list[str|Path],
        transforms: Callable[[any], any] | None = None,
        ):
        """
        Args:
            img_root: directory with all images.
            gt_root: directory with all gts(ground truth).
            transform: Optional transform to be applied on a sample.
        """
        self.img_root = []
        for p in img_root:
            if isinstance(p, str):
                p = os.path.expanduser(p)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"unable to locate {p}"
                )
            self.img_root.append(p)

        self.gt_root = []
        for p in gt_root:
            if isinstance(p, str):
                p = os.path.expanduser(p)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"unable to locate {p}"
                )
            self.gt_root.append(p)

        self.transforms = transforms
        self.datas: list[tuple[Path, list[str], np.ndarray]] = []

        for img_root in self.img_root:
            np_dtype = np.float32
            img_names = os.listdir(img_root)
            for img_name in tqdm(iterable=img_names, desc="Preparing and Loading icdar2015", total=len(img_names)):
                img_path = Path(img_root, img_name)
                img_id = Path(img_name).stem
                for gt_root in self.gt_root:
                    gt_path = Path(gt_root, "gt_" + img_id + ".txt")
                    if os.path.exists(gt_path):
                        break
                    gt_path = None

                polygon_classes: list[str] = []
                polygons: list = []

                with open(gt_path, newline="\n") as f:
                    for line in f:
                        parts = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line.strip().split(',')]

                        polygon_classes.append(parts[-1])
                        polygons.append(np.array(list(map(float, parts[:8]))).reshape((-1, 2)).tolist())

                self.datas.append((img_path, img_id, polygons, polygon_classes))

    def _read_sample(self, index: int):
        img_path, img_id, polygons, polygon_classes = self.datas[index]

        # Read image
        with Image.open(img_path) as pil_img:
            img = np.array(pil_img)

        return img, img_path, img_id, polygons, polygon_classes

    def _format_sample(self, sample):
        img, img_path, img_id, polygons, polygon_classes = sample

        img = TF.to_tensor(img)

        polygons = np.array(polygons)

        target = {
            "img_id": img_id,
            "img_path": str(img_path),
            "polygons": polygons,
            "polygon_classes": polygon_classes,
            "polygon_ignores": [label in ["###"] for label in polygon_classes],
            }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index: int) -> (torch.Tensor, dict):
        """
        Returns:
            image: image as tensor(N, C, H, W).
            target: metadata(e.g. polygons, polygon_classes, ...)
        """
        return self._format_sample(self._read_sample(index))

    def get_by_id(self, id) -> (torch.Tensor, dict):
        for d in self.datas:
            if d[1] == id:
                img_path, img_id, polygons, polygon_classes = d

                # Read image
                with Image.open(img_path) as pil_img:
                    img = np.array(pil_img)

                return self._format_sample((img, img_path, img_id, polygons, polygon_classes))
        raise Exception("no data")
