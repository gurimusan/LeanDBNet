import torch
import torch.nn as nn


class MaskL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, gt, mask):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum, dict(l1_loss=mask_sum)
        else:
            loss = (torch.abs(pred[:, 0] - gt) * mask).sum() / mask_sum
            return loss, dict(l1_loss=loss)
