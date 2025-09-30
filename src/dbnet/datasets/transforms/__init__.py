import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from shapely import affinity
from shapely.geometry import Polygon
from torch import nn
from torchvision.transforms.v2 import Compose, Normalize, RandomApply, RandomChoice

__all__ = [
    "Compose",
    "Normalize",
    "RandomApply",
    "RandomChoice",
    "RandomMinResize",
    "HorizontalFlip",
    "RandomCrop",
    "RandomRotate",
]


class RandomMinResize(nn.Module):
    """無作為に画像をアスペクト比を保持してリサイズ"""

    def __init__(self, min_sizes: list[int], max_size: int):
        """
        Args:
            min_sizes: 短辺の長さの候補、この中から無作為に長さを抽出
            max_size: 長辺の長さの最大値
        """
        super().__init__()
        self.min_sizes = min_sizes
        self.max_size = max_size

    def _get_target_size(self, min_side: int, max_side: int,
                         target: int):
        # アスペクト比を保持して短辺をtargetに合わせる
        max_side = int(max_side * target / min_side)
        min_side = target

        # 長辺がmax_sizeを超えている場合、
        # アスペクト比を保持して長辺をmax_sizeに合わせる
        if max_side > self.max_size:
            min_side = int(min_side * self.max_size / max_side)
            max_side = self.max_size

        return min_side, max_side

    def forward(self, img: torch.Tensor, target: dict) -> tuple[torch.Tensor, dict]:
        """
        Args:
            img: image as tensor (C, H, W)
            target: metadata(e.g. polygons, polygon_classes, ...)
        """
        min_size = random.choice(self.min_sizes)

        orig_shape = img.shape

        _, h, w = img.shape

        # リサイズ後の大きさを取得
        # 幅と高さのどちらが短辺であるかで場合分け
        if w < h:
            rw, rh  = self._get_target_size(w, h, min_size)
        else:
            rh, rw = self._get_target_size(h, w, min_size)

        # 指定した大きさに画像をリサイズ
        img = F.resize(img, (rh, rw))

        # 正解矩形をリサイズ前後のスケールに合わせて変更
        ratio = rw / w
        try:
            target["polygons"] *= ratio
        except Exception as e:
            print(target["polygons"].dtype)
            print(ratio)
            raise e

        # リサイズ後の画像の大きさを保持
        target['shape'] = orig_shape

        return img, target


class HorizontalFlip(nn.Module):
    """画像を水平反転する"""

    def forward(self, img: np.ndarray, target: dict) -> tuple[np.ndarray, dict]:
        """
        Args:
            img: image as tensor (C, H, W)
            target: metadata(e.g. polygons, polygon_classes, ...)
        """
        # 画像の水平反転
        img = F.hflip(img)

        # 正解矩形をx軸方向に反転
        w = img.shape[2]
        target["polygons"][:, :, 0] = w - target["polygons"][:, :, 0]

        return img, target


class RandomCrop(nn.Module):
    """無作為に画像を切り抜く"""

    def __init__(self, size: tuple[int, int] = (512, 512),
                 max_tries: int =50, min_crop_side_ratio: float = 0.1):
        """
        Args:
            size: 切り取るサイズ. (w, h)のようなタプル.
            max_tries: 切り取り位置を決める試行回数.
        """
        super().__init__()
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio

    def forward(self, img: torch.Tensor, target: dict) -> tuple[torch.Tensor, dict]:
        """
        Args:
            img: image as tensor (C, H, W)
            target: metadata(e.g. polygons, polygon_classes, ...)
        """
        all_care_polygons = [p for p, ignore, c in zip(target["polygons"], target["polygon_ignores"], target["polygon_classes"], strict=False) if not ignore]
        crop_x, crop_y, crop_w, crop_h = self._crop_area(img, np.array(all_care_polygons))

        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        padimg = torch.zeros((img.shape[0], self.size[1], self.size[0]), dtype=img.dtype)
        padimg[:, :h, :w] = F.resize(
            img[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (h, w))
        img = padimg

        polygons = []
        polygon_ignores = []
        polygon_classes = []
        for p, ignore, c in zip(target["polygons"], target["polygon_ignores"], target["polygon_classes"], strict=False):
            poly = ((np.array(p) - (crop_x, crop_y)) * scale).tolist()
            if not self._is_poly_outside_rect(poly, 0, 0, w, h):
                polygons.append(poly)
                polygon_ignores.append(ignore)
                polygon_classes.append(c)

        target["polygons"] = np.array(polygons)
        target["polygon_ignores"] = polygon_ignores
        target["polygon_classes"] = polygon_classes
        target['scale_w'] = scale
        target['scale_h'] = scale

        return img, target

    def _crop_area(self, img: torch.Tensor, polygons: np.ndarray):
        _, h, w = img.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in polygons:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self._split_regions(h_axis)
        w_regions = self._split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self._region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self._random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self._region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self._random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in polygons:
                if not self._is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h

    def _is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def _split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i-1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def _random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def _region_wise_random_select(self, regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax


class RandomRotate(nn.Module):
    """無作為に角度を変える"""
    def __init__(self, degrees: tuple[int, int]):
        super().__init__()
        self.degrees = degrees

    def forward(self, img: torch.Tensor, target: dict) -> tuple[torch.Tensor, dict]:
        """
        Args:
            img: image as tensor (C, H, W)
            target: metadata(e.g. polygons, polygon_classes, ...)
        """
        _, h, w = img.shape
        ch = h / 2
        cw = w / 2

        angle = random.randint(*self.degrees)
        rotated_img = F.rotate(img, angle=float(angle))

        polygons = []
        polygon_ignores = []
        polygon_classes = []
        for p, ignore, c in zip(target["polygons"], target["polygon_ignores"], target["polygon_classes"], strict=False):
            rotated_p = self._rotate_poly(p, angle, (cw, ch))
            if not self._is_poly_outside_rect(rotated_p, 0, 0, w, h):
                polygons.append(rotated_p)
                polygon_ignores.append(ignore)
                polygon_classes.append(c)

        target["polygons"] = np.array(polygons)
        target["polygon_ignores"] = polygon_ignores
        target["polygon_classes"] = polygon_classes

        return rotated_img, target

    def _rotate_poly(self, polygon, angle, center):
        radius = np.radians(-angle)
        m = np.array([
            [np.cos(radius), -np.sin(radius)],
            [np.sin(radius), np.cos(radius)],
            ])

        rotated_polygons = polygon.copy()
        rotated_polygons[:, 0] -= center[0]
        rotated_polygons[:, 1] -= center[1]
        rotated_polygons = np.matmul(rotated_polygons, m.T)
        rotated_polygons[:, 0] += center[0]
        rotated_polygons[:, 1] += center[1]

        return rotated_polygons

    def _is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False
