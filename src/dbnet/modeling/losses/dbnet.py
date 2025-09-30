from typing import Any

import cv2
import numpy as np
import pyclipper
import torch
from shapely.geometry import Polygon
from torch import nn

from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
from .dice_loss import DiceLoss
from .l1_loss import MaskL1Loss


class DBNetLoss(nn.Module):
    """
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    """

    shrink_ratio = 0.4
    thresh_min = 0.3
    thresh_max = 0.7
    min_size_box = 8

    def __init__(self, eps=1e-6, alpha=1.0, beta=10):
        """
        Args:
            eps: epsilon for dice loss
            alpha: weight for bce_loss (prob_map loss)
                - Paper formula: L = Ls + α*Lb + β*Lt
                - Implementation formula: L = Lb + α*Ls + β*Lt
                - MhLiao/DB: alpha=5.0, beta=10
                - WenmuZhou/DBNet.pytorch: alpha=1.0, beta=10
                - This implementation: alpha=1.0, beta=10 (following majority)
            beta: weight for l1_loss (threshold_map loss)
                - All implementations use beta=10
        """
        super().__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: dict[str, torch.Tensor], target: list[dict[str, Any]]):
        prob_map = pred.get("prob_map")

        if prob_map is None:
            raise ValueError("prob_map is required in pred dict")

        shape = prob_map.shape
        target_shape = (shape[0], shape[1], shape[2], shape[3])
        seg_target, seg_mask, thresh_target, thresh_mask = self._build_target(target, target_shape)
        seg_target = torch.from_numpy(seg_target).to(prob_map.device)
        seg_mask = torch.from_numpy(seg_mask).to(prob_map.device)
        thresh_target = torch.from_numpy(thresh_target).to(prob_map.device)
        thresh_mask = torch.from_numpy(thresh_mask).to(prob_map.device)

        bce_loss = self.bce_loss(prob_map, seg_target, seg_mask)
        if "thresh_map" in pred:
            l1_loss, l1_metric = self.l1_loss(pred["thresh_map"], thresh_target, thresh_mask)
            dice_loss = self.dice_loss(pred["binary_map"], seg_target, seg_mask)
            # Note: Formula difference between paper and implementation
            # Paper formula: L = Ls + α*Lb + β*Lt
            # Implementation: L = Lb + α*Ls + β*Lt
            # where Ls = probability map loss (bce_loss)
            #       Lb = approximate binary map loss (dice_loss)
            #       Lt = threshold map loss (l1_loss)
            # This follows the original MhLiao/DB implementation
            loss = dice_loss + self.alpha * bce_loss + self.beta * l1_loss
        else:
            loss = bce_loss
        return loss

    def _build_target(self, target: list[dict[str, Any]], target_shape: tuple[int, int, int, int]):
        seg_target: np.ndarray = np.zeros(target_shape, dtype=np.uint8)
        seg_mask: np.ndarray = np.ones((target_shape[0], target_shape[2], target_shape[3]), dtype=np.float32)
        thresh_target: np.ndarray = np.zeros(target_shape, dtype=np.float32)
        thresh_mask: np.ndarray = np.zeros((target_shape[0], target_shape[2], target_shape[3]), dtype=np.uint8)

        for idx, tgt in enumerate(target):
            polygons = tgt.get("polygons")
            polygon_ignores = tgt.get("polygon_ignores")

            if polygons is None or polygon_ignores is None:
                continue

            for class_idx in range(len(polygons)):
                polygon = polygons[class_idx]
                height = max(polygon[:, 1]) - min(polygon[:, 1])
                width = max(polygon[:, 0]) - min(polygon[:, 0])

                if polygon_ignores[class_idx] or min(height, width) < self.min_size_box:
                    cv2.fillPoly(seg_mask[idx], polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    polygon_ignores[class_idx] = True
                else:
                    polygon_shape = Polygon(polygon)
                    distance = polygon_shape.area * \
                        (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                    subject = [tuple(l) for l in polygons[class_idx]]
                    padding = pyclipper.PyclipperOffset()
                    padding.AddPath(subject, pyclipper.JT_ROUND,
                                    pyclipper.ET_CLOSEDPOLYGON)
                    shrinked = padding.Execute(-distance)
                    if shrinked == []:
                        cv2.fillPoly(seg_mask[idx], polygon.astype(np.int32)[np.newaxis, :, :], 0)
                        polygon_ignores[class_idx] = True
                    else:
                        shrinked = np.array(shrinked[0]).reshape(-1, 2)
                        cv2.fillPoly(seg_target[idx, 0], [shrinked.astype(np.int32)], 1)

                if not polygon_ignores[class_idx]:
                    poly, thresh_target[idx, 0], thresh_mask[idx] = self._draw_thresh_map(
                        polygon, thresh_target[idx, 0], thresh_mask[idx]
                    )

        thresh_target = thresh_target.astype(np.float32) * (self.thresh_max - self.thresh_min) + self.thresh_min

        seg_target = seg_target.astype(np.float32)
        seg_mask = seg_mask.astype(bool)
        thresh_target = thresh_target.astype(np.float32)
        thresh_mask = thresh_mask.astype(bool)

        return seg_target, seg_mask, thresh_target, thresh_mask

    def _draw_thresh_map(
        self,
        polygon: np.ndarray,
        canvas: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a polygon treshold map on a canvas, as described in the DB paper

        Args:
            polygon: array of coord., to draw the boundary of the polygon
            canvas: threshold map to fill with polygons
            mask: mask for training on threshold polygons
        """
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise AttributeError("polygon should be a 2 dimensional array of coords")

        polygon = np.array(polygon)

        # Augment polygon by shrink_ratio
        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(coor) for coor in polygon]  # Get coord as list of tuples
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon: np.ndarray = np.array(padding.Execute(distance)[0])

        # Fill the mask with 1 on the new padded polygon
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        # Get min/max to recover polygon after distance computation
        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        # Get absolute polygon for distance computation
        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin
        # Get absolute padded polygon
        xs: np.ndarray = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys: np.ndarray = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        # Compute distance map to fill the padded polygon
        distance_map = np.zeros((polygon.shape[0], height, width), dtype=polygon.dtype)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self._compute_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = np.min(distance_map, axis=0)

        # Clip the padded polygon inside the canvas
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)

        # Fill the canvas with the distances computed inside the valid padded polygon
        canvas[ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1] = np.fmax(
            1
            - distance_map[
                ymin_valid - ymin : ymax_valid - ymax + height, xmin_valid - xmin : xmax_valid - xmax + width
            ],
            canvas[ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1],
        )

        return polygon, canvas, mask

    def _compute_distance(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Compute the distance for each point of the map (xs, ys) to the (a, b) segment

        Args:
            xs : map of x coordinates (height, width)
            ys : map of y coordinates (height, width)
            a: first point defining the [ab] segment
            b: second point defining the [ab] segment
            eps: epsilon to avoid division by zero

        Returns:
            The computed distance

        """
        square_dist_1 = np.square(xs - a[0]) + np.square(ys - a[1])
        square_dist_2 = np.square(xs - b[0]) + np.square(ys - b[1])
        square_dist = np.square(a[0] - b[0]) + np.square(a[1] - b[1])
        cosin = (square_dist - square_dist_1 - square_dist_2) / (2 * np.sqrt(square_dist_1 * square_dist_2) + eps)
        cosin = np.clip(cosin, -1.0, 1.0)
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_dist_1 * square_dist_2 * square_sin / square_dist + eps)
        result[cosin < 0] = np.sqrt(np.fmin(square_dist_1, square_dist_2))[cosin < 0]
        return np.asarray(result)
