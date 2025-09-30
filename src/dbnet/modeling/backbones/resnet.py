from typing import Any

import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)


def _ovewrite_named_param(kwargs: dict[str, Any], param: str, new_value: Any) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


class BasicBlock(nn.Module):
    """残差ブロック

    1. 畳み込み層
    2. バッチ正規化
    3. 活性化関数
    4. 畳み込み
    5. バッチ正規化
    6. 残差接続 or スキップ接続
    7. 活性化関数
    """
    def __init__(self, in_channel: int, out_channel: int, stride: int, with_dcn: bool = False):
        super().__init__()

        self.with_dcn = with_dcn

        # 畳み込み
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        # バッチ正規化
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 活性化関数
        self.relu = nn.ReLU(inplace=True)
        # 畳み込み
        if not with_dcn:
            self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            from torchvision.ops import DeformConv2d
            # # 3x3の畳み込みであれば、行列方向の2成分×9画素の計18成分のオフセットを計算
            offset_channels = 18
            self.conv2_offset = nn.Conv2d(out_channel, offset_channels, kernel_size=3, padding=1)
            self.conv2 = DeformConv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        # バッチ正規化
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 残差接続
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
                )

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        # 残差接続 or スキップ接続
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    """bottleneckという構造の残差ブロック

    1. 畳み込み層 (1x1, 64, 1)
    2. バッチ正規化
    3. 活性化関数
    4. 畳み込み層 (3x3, 64, 1)
    5. バッチ正規化
    6. 活性化関数
    7. 畳み込み層 (1x1, 256, 1)
    8. バッチ正規化
    9. 残差接続 or スキップ接続
    10. 活性化関数
    """
    def __init__(self, in_channel: int, out_channel: int, stride: int, with_dcn: bool = False):
        super().__init__()

        inner_channel = out_channel // 4

        self.with_dcn = with_dcn

        # 畳み込み
        self.conv1 = nn.Conv2d(in_channel, inner_channel, kernel_size=1, stride=1, bias=False)
        # バッチ正規化
        self.bn1 = nn.BatchNorm2d(inner_channel)
        # 畳み込み
        if not with_dcn:
            self.conv2 = nn.Conv2d(inner_channel, inner_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            from torchvision.ops import DeformConv2d
            # # 3x3の畳み込みであれば、行列方向の2成分×9画素の計18成分のオフセットを計算
            offset_channels = 18
            self.conv2_offset = nn.Conv2d(inner_channel, offset_channels, kernel_size=3, stride=stride, padding=1)
            self.conv2 = DeformConv2d(inner_channel, inner_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        # バッチ正規化
        self.bn2 = nn.BatchNorm2d(inner_channel)
        # 畳み込み
        self.conv3 = nn.Conv2d(inner_channel, out_channel, kernel_size=1, stride=1, bias=False)
        # バッチ正規化
        self.bn3 = nn.BatchNorm2d(out_channel)

        # 活性化関数
        self.relu = nn.ReLU(inplace=True)

        # 残差接続
        self.downsample = None
        if stride > 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
                )

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # 残差接続 or スキップ接続
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet

    1. 畳み込み層 (64, 7x7, 2)
    2. 最大プーリング(, 3x3, 2)
    3. layer1
    4. layer2
    5. layer3
    6. layer4
    7. 平均プーリング
    8. 全結合層
    """
    def __init__(self, layers: list, out_channels: list[int], num_classes: int = 1000):
        super().__init__()

        # 畳み込み層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 最大プーリング
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = layers[0]
        self.layer2 = layers[1]
        self.layer3 = layers[2]
        self.layer4 = layers[3]

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.max_pool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5


def _make_base_block_layer(in_channel: int, out_channel: int, block: int, stride: int, with_dcn: bool = False):
    layers = []
    layers.append(BasicBlock(in_channel, out_channel, stride, with_dcn=with_dcn))
    for _ in range(1, block):
        layers.append(BasicBlock(out_channel, out_channel, 1, with_dcn=with_dcn))
    return nn.Sequential(*layers)


def _make_bottleneck_block_layer(in_channel: int, out_channel: int, block: int, stride: int, with_dcn: bool = False):
    layers = []
    layers.append(BottleneckBlock(in_channel, out_channel, stride, with_dcn=with_dcn))
    for _ in range(1, block):
        layers.append(BottleneckBlock(out_channel, out_channel, 1, with_dcn=with_dcn))
    return nn.Sequential(*layers)


def resnet18(*, weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet18

    1. 畳み込み層 (64, 7x7, 2)
    2. 最大プーリング(, 3x3, 2)
    3. layer1 残差ブロック (64, 64, 1) x2
    4. layer2 残差ブロック (64, 128, 2) x2
    5. layer3 残差ブロック (128, 256, 2) x2
    6. layer4 残差ブロック (256, 512, 2) x2
    7. 平均プーリング
    8. 全結合層
    """
    weights = ResNet18_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet([
        _make_base_block_layer(64, 64, 2, stride=1),
        _make_base_block_layer(64, 128, 2, stride=2),
        _make_base_block_layer(128, 256, 2, stride=2),
        _make_base_block_layer(256, 512, 2, stride=2),
        ], [64, 128, 256, 512], **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True), strict=False)

    return model


def deformable_resnet18(*, weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet18

    1. 畳み込み層 (64, 7x7, 2)
    2. 最大プーリング(, 3x3, 2)
    3. layer1 残差ブロック (64, 64, 1) x2
    4. layer2 残差ブロック (64, 128, 2) x2
    5. layer3 残差ブロック (128, 256, 2) x2
    6. layer4 残差ブロック (256, 512, 2) x2
    7. 平均プーリング
    8. 全結合層
    """
    weights = ResNet18_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet([
        _make_base_block_layer(64, 64, 2, stride=1),
        _make_base_block_layer(64, 128, 2, stride=2, with_dcn=True),
        _make_base_block_layer(128, 256, 2, stride=2, with_dcn=True),
        _make_base_block_layer(256, 512, 2, stride=2, with_dcn=True),
        ], [64, 128, 256, 512], **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True), strict=False)

    return model


def resnet34(*, weights: ResNet34_Weights | None = ResNet34_Weights.DEFAULT, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet34

    1. 畳み込み層 (64, 7x7, 2)
    2. 最大プーリング(, 3x3, 2)
    3. layer1 残差ブロック (64, 64, 1) x3
    4. layer2 残差ブロック (64, 128, 2) x4
    5. layer3 残差ブロック (128, 256, 2) x6
    6. layer4 残差ブロック (256, 512, 2) x3
    7. 平均プーリング
    8. 全結合層
    """
    weights = ResNet34_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet([
        _make_base_block_layer(64, 64, 3, stride=1),
        _make_base_block_layer(64, 128, 4, stride=2),
        _make_base_block_layer(128, 256, 6, stride=2),
        _make_base_block_layer(256, 512, 3, stride=2),
        ], [64, 128, 256, 512], **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True), strict=False)

    return model


def resnet50(*, weights: ResNet50_Weights | None = ResNet50_Weights.DEFAULT, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet50

    1. 畳み込み層 (64, 7x7, 2)
    2. 最大プーリング(, 3x3, 2)
    3. layer1 残差ブロック(64, 256, 1) x3
    4. layer2 残差ブロック (256, 512, 2) x4
    5. layer3 残差ブロック (512, 1024, 2) x6
    6. layer4 残差ブロック (1024, 2048, 2) x3
    7. 平均プーリング
    8. 全結合層
    """
    weights = ResNet50_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet([
        _make_bottleneck_block_layer(64, 256, 3, stride=1),
        _make_bottleneck_block_layer(256, 512, 4, stride=2),
        _make_bottleneck_block_layer(512, 1024, 6, stride=2),
        _make_bottleneck_block_layer(1024, 2048, 3, stride=2),
        ], [256, 512, 1024, 2048], **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True), strict=False)

    return model


def deformable_resnet50(*, weights: ResNet50_Weights | None = ResNet50_Weights.DEFAULT, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet50

    1. 畳み込み層 (64, 7x7, 2)
    2. 最大プーリング(, 3x3, 2)
    3. layer1 残差ブロック(64, 256, 1) x3
    4. layer2 残差ブロック (256, 512, 2) x4
    5. layer3 残差ブロック (512, 1024, 2) x6
    6. layer4 残差ブロック (1024, 2048, 2) x3
    7. 平均プーリング
    8. 全結合層
    """
    weights = ResNet50_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet([
        _make_bottleneck_block_layer(64, 256, 3, stride=1),
        _make_bottleneck_block_layer(256, 512, 4, stride=2, with_dcn=True),
        _make_bottleneck_block_layer(512, 1024, 6, stride=2, with_dcn=True),
        _make_bottleneck_block_layer(1024, 2048, 3, stride=2, with_dcn=True),
        ], [256, 512, 1024, 2048], **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True), strict=False)

    return model


def resnet101(*, weights: ResNet101_Weights | None = ResNet101_Weights.DEFAULT, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet101

    1. 畳み込み層 (64, 7x7, 2)
    2. 最大プーリング(, 3x3, 2)
    3. layer1 残差ブロック(64, 256, 1) x3
    4. layer2 残差ブロック (256, 512, 2) x4
    5. layer3 残差ブロック (512, 1024, 2) x23
    6. layer4 残差ブロック (1024, 2048, 2) x3
    7. 平均プーリング
    8. 全結合層
    """
    weights = ResNet101_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet([
        _make_bottleneck_block_layer(64, 256, 3, stride=1),
        _make_bottleneck_block_layer(256, 512, 4, stride=2),
        _make_bottleneck_block_layer(512, 1024, 23, stride=2),
        _make_bottleneck_block_layer(1024, 2048, 3, stride=2),
        ], [256, 512, 1024, 2048], **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True), strict=False)

    return model


def resnet152(*, weights: ResNet152_Weights | None = ResNet152_Weights.DEFAULT, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet152

    1. 畳み込み層 (64, 7x7, 2)
    2. 最大プーリング(, 3x3, 2)
    3. layer1 残差ブロック(64, 256, 1) x3
    4. layer2 残差ブロック (256, 512, 2) x8
    5. layer3 残差ブロック (512, 1024, 2) x36
    6. layer4 残差ブロック (1024, 2048, 2) x3
    7. 平均プーリング
    8. 全結合層
    """
    weights = ResNet152_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet([
        _make_bottleneck_block_layer(64, 256, 3, stride=1),
        _make_bottleneck_block_layer(256, 512, 8, stride=2),
        _make_bottleneck_block_layer(512, 1024, 36, stride=2),
        _make_bottleneck_block_layer(1024, 2048, 3, stride=2),
        ], [256, 512, 1024, 2048], **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True), strict=False)

    return model
