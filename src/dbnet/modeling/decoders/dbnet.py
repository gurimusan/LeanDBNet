from typing import Any

import torch
from torch import nn


class DBNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: list[int]=[64, 128, 256, 512],
        inner_channels: int=256,
        k: int=50,
        smooth: bool=False,
        serial: bool=False
        ):
        """
        :param int k: Controll the gradient.
        :param bool smooth: If true, use bilinear instead of deconv.
        :param bool serial: If true, thresh prediction will combine segmentation result as input.
        """
        super().__init__()

        self.k = k
        self.serial = serial

        # for feature pyramid network
        self.in5 = nn.Conv2d(in_channels[3], inner_channels, kernel_size=1, bias=False)
        self.in4 = nn.Conv2d(in_channels[2], inner_channels, kernel_size=1, bias=False)
        self.in3 = nn.Conv2d(in_channels[1], inner_channels, kernel_size=1, bias=False)
        self.in2 = nn.Conv2d(in_channels[0], inner_channels, kernel_size=1, bias=False)
        self.up5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.Upsample(scale_factor=8, mode="nearest"))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.Upsample(scale_factor=4, mode="nearest"))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode="nearest"))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels // 4, kernel_size=3, padding=1, bias=False)

        # for probability map
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self._init_weights)

        # for threshold map
        self.thresh = self._init_thresh(
            inner_channels, serial=serial, smooth=smooth)
        self.thresh.apply(self._init_weights)

        self.in5.apply(self._init_weights)
        self.in4.apply(self._init_weights)
        self.in3.apply(self._init_weights)
        self.in2.apply(self._init_weights)
        self.out5.apply(self._init_weights)
        self.out4.apply(self._init_weights)
        self.out3.apply(self._init_weights)
        self.out2.apply(self._init_weights)

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels: int, serial: bool=False, smooth: bool=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        return nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth),
            nn.Sigmoid())

    def _init_upsample(self, in_channels, out_channels, smooth=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_channels, inter_out_channels, kernel_size=3, stride=1, padding=1, bias=False)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=0, bias=False))

            return nn.Sequential(*module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def _differentiable_binarization(self, p, t):
        """`B = 1 / (1 + exp(-k * (p - t)))`

        Args:
            p: probability map
            t: threshold map
        """
        return torch.reciprocal(1 + torch.exp(-self.k * (p - t)))

    def forward(self, x: torch.Tensor):
        c2, c3, c4, c5 = x # x2, x3, x4, x5 = 1/4, 1/8, 1/16, 1/32

        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = in4 + self.up5(in5)  # 1/16
        out3 = in3 + self.up4(out4) # 1/8
        out2 = in2 + self.up3(out3) # 1/4

        p5 = self.out5(in5)     # 1/4
        p4 = self.out4(out4)    # 1/4
        p3 = self.out3(out3)    # 1/4
        p2 = self.out2(out2)    # 1/4

        fuse = torch.cat((p5, p4, p3, p2), 1)

        result: dict[str, Any] = dict()

        # pred text segmentation map
        prob_map = self.binarize(fuse)
        result.update(prob_map=prob_map)

        # in inference, return only the text segmentation map
        if not self.training:
            return result

        # in training,
        if self.serial:
            fuse = torch.cat(
                    (fuse, nn.functional.interpolate(
                        prob_map, fuse.shape[2:])), 1)

        # pred threshold map
        thresh_map = self.thresh(fuse)
        result.update(thresh_map=thresh_map)

        # calc a approximate binary map
        binary_map = self._differentiable_binarization(prob_map, thresh_map)
        result.update(binary_map=binary_map)

        return result
