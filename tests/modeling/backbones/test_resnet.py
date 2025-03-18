import torch
from src.dbnet.modeling.backbones import (
    resnet18,
    deformable_resnet18,
    resnet34,
    resnet50,
    deformable_resnet50,
    resnet101,
    resnet152,
    )

def test_resnet18():
    x = torch.zeros(2, 3, 32, 32)
    bb_model = resnet18()
    y = bb_model(x)

    assert len(y) == 4
    assert y[0].shape == (2, 64, 8, 8)
    assert y[1].shape == (2, 128, 4, 4)
    assert y[2].shape == (2, 256, 2, 2)
    assert y[3].shape == (2, 512, 1, 1)


def test_deformable_resnet18():
    x = torch.zeros(2, 3, 32, 32)
    bb_model = deformable_resnet18()
    y = bb_model(x)

    assert len(y) == 4
    assert y[0].shape == (2, 64, 8, 8)
    assert y[1].shape == (2, 128, 4, 4)
    assert y[2].shape == (2, 256, 2, 2)
    assert y[3].shape == (2, 512, 1, 1)


def test_resnet34():
    x = torch.zeros(2, 3, 32, 32)
    bb_model = resnet34()
    y = bb_model(x)

    assert len(y) == 4
    assert y[0].shape == (2, 64, 8, 8)
    assert y[1].shape == (2, 128, 4, 4)
    assert y[2].shape == (2, 256, 2, 2)
    assert y[3].shape == (2, 512, 1, 1)


def test_resnet50():
    x = torch.zeros(2, 3, 32, 32)
    bb_model = resnet50()
    y = bb_model(x)

    assert len(y) == 4
    assert y[0].shape == (2, 256, 8, 8)
    assert y[1].shape == (2, 512, 4, 4)
    assert y[2].shape == (2, 1024, 2, 2)
    assert y[3].shape == (2, 2048, 1, 1)


def test_deformable_resnet50():
    x = torch.zeros(2, 3, 32, 32)
    bb_model = deformable_resnet50()
    y = bb_model(x)

    assert len(y) == 4
    assert y[0].shape == (2, 256, 8, 8)
    assert y[1].shape == (2, 512, 4, 4)
    assert y[2].shape == (2, 1024, 2, 2)
    assert y[3].shape == (2, 2048, 1, 1)


def test_resnet101():
    x = torch.zeros(2, 3, 32, 32)
    bb_model = resnet101()
    y = bb_model(x)

    assert len(y) == 4
    assert y[0].shape == (2, 256, 8, 8)
    assert y[1].shape == (2, 512, 4, 4)
    assert y[2].shape == (2, 1024, 2, 2)
    assert y[3].shape == (2, 2048, 1, 1)


def test_resnet152():
    x = torch.zeros(2, 3, 32, 32)
    bb_model = resnet152()
    y = bb_model(x)

    assert len(y) == 4
    assert y[0].shape == (2, 256, 8, 8)
    assert y[1].shape == (2, 512, 4, 4)
    assert y[2].shape == (2, 1024, 2, 2)
    assert y[3].shape == (2, 2048, 1, 1)
