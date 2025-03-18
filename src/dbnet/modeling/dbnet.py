import importlib

import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

import torch
from torch import nn

class DBNet(nn.Module):
    thresh = 0.3
    box_thresh = 0.7
    # box_thresh = 0.3
    max_candidates = 100
    min_size = 3

    def __init__(
        self,
        backbone="deformable_resnet50",
        backbone_args: dict[str, any] = {},
        decoder="DBNetDecoder",
        decoder_args: dict[str, any] = {},
        loss="DBNetLoss",
        loss_args: dict[str, any] = {},
        ):
        super().__init__()
        self.backbone = self._load_backbone(backbone, backbone_args)
        self.decoder = self._load_decoder(decoder, {**{
            "in_channels": self.backbone.out_channels,
            }, **decoder_args})
        self.loss = self._load_loss(loss, loss_args)

    def forward(self, x: torch.Tensor, target: list[any] | None = None, return_preds: bool = False):
        y = self.backbone(x)
        y = self.decoder(y)

        result: dict[str, any] = dict(out_map=y["prob_map"])

        if target is None or return_preds:
            # Disable for torch.compile compatibility
            @torch.compiler.disable  # type: ignore[attr-defined]
            def _pred(prob_map: torch.Tensor):
                return self._pred_boxes(prob_map.detach().cpu().numpy())

            # Post-process boxes (keep only text predictions)
            result["preds"] = _pred(y["prob_map"])

        if target is not None:
            loss = self.loss(y, target)
            result["loss"] = loss

        return result

    def _load_backbone(self, backbone: str, backbone_args: dict[str, any]):
        mod = importlib.import_module("src.dbnet.modeling.backbones")
        klass = getattr(mod, backbone)
        instance = klass(**backbone_args)
        return instance

    def _load_decoder(self, decoder: str, decoder_args: dict[str, any]):
        mod = importlib.import_module("src.dbnet.modeling.decoders")
        klass = getattr(mod, decoder)
        instance = klass(**decoder_args)
        return instance

    def _load_loss(self, loss: str, loss_args: dict[str, any]):
        mod = importlib.import_module("src.dbnet.modeling.losses")
        klass = getattr(mod, loss)
        instance = klass(**loss_args)
        return instance

    def _pred_boxes(self, prob_map):
        """
        Args:
            prob_map: probability map of shape (N, C, H, W)
        """
        segmentation = prob_map > self.thresh

        boxes_batch = []
        scores_batch = []
        for pmap, bmap in zip(prob_map, segmentation):
            boxes, scores = self._boxes_from_bitmap(pmap, bmap)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def _boxes_from_bitmap(self, prob_map, bitmap):
        """
        Args:
            prob_map: probability map of shape (1, H, W)
            bitmap: binalized map of shape (1, H, W)
        """
        bitmap = bitmap[0]  # The first channel
        prob_map = prob_map[0]
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap*255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self._get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            points = np.array(points)
            score = self._box_score_fast(prob_map, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self._unclip(points).reshape(-1, 1, 2)
            box, sside = self._get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(box[:, 0], 0, width)
            box[:, 1] = np.clip(box[:, 1], 0, height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score

        return boxes, scores

    def _unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def _get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def _box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int16), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int16), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int16), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int16), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]
