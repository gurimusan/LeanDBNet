import torch
from src.dbnet.modeling.backbones import deformable_resnet50, deformable_resnet18
from src.dbnet.modeling.decoders import DBNetDecoder


def test_dbnet_decoder_resnet50():
    backbone = deformable_resnet50()
    decoder = DBNetDecoder(in_channels=backbone.out_channels, smooth=True)

    x = torch.zeros(2, 3, 32, 32)
    y = decoder(backbone(x))

    assert y['prob_map'].shape == (2, 1, 32, 32)


def test_dbnet_decoder_resnet18():
    backbone = deformable_resnet18()
    decoder = DBNetDecoder(in_channels=backbone.out_channels)

    x = torch.zeros(2, 3, 32, 32)
    y = decoder(backbone(x))

    assert y['prob_map'].shape == (2, 1, 32, 32)
