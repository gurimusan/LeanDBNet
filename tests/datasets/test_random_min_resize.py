import math
import random

import torch
import numpy as np
import pytest
from unittest.mock import patch

from src.dbnet.datasets.transforms import RandomMinResize


@pytest.mark.parametrize(
    "img_shape,min_size_choice,max_size,want_h,want_w",
    [
        # Regular case: w < h, no max_size constraint
        [(3, 600, 400), 300, 1000, 450, 300],

        # Regular case: h < w, no max_size constraint
        [(3, 400, 600), 300, 1000, 300, 450],

        # Case where max_size is applied (w < h)
        [(3, 1200, 300), 500, 800, 800, 200],

        # Case where max_size is applied (h < w)
        [(3, 300, 1200), 500, 800, 200, 800],

        # Edge case: square image
        [(3, 500, 500), 300, 1000, 300, 300],

        # Edge case: tiny image
        [(3, 50, 25), 100, 200, 200, 100],
    ],
)
def test_random_min_resize_dimensions(mocker, img_shape, min_size_choice, max_size, want_h, want_w):
    """Test that RandomMinResize produces correct output dimensions"""
    min_sizes = [100, 300, 500]
    model = RandomMinResize(min_sizes, max_size)

    # Mock random.choice to return our parametrized value
    mocker.patch("random.choice", return_value=min_size_choice)

    # Create test image and target
    img = torch.randn(*img_shape)
    c, h, w = img_shape

    # Create test polygons
    polygons = np.array([
        [[10.0, 20.0], [30.0, 30.0], [40.0, 50.0], [10.0, 40.0]],
        [[100.0, 120.0], [130.0, 130.0], [140.0, 150.0], [110.0, 140.0]],
    ])
    target = {"polygons": polygons.copy()}

    # Apply transform
    resized_img, resized_target = model(img, target)

    # Check output dimensions
    assert resized_img.shape == (c, want_h, want_w)

    # Check that original shape is preserved in target
    assert resized_target["shape"] == img.shape

    # Check that polygon array has the same shape (num_polygons, num_points, 2)
    assert resized_target["polygons"].shape == polygons.shape


def test_random_min_resize_polygon_scaling():
    """Test that polygons are properly scaled after resize"""
    min_sizes = [300]
    max_size = 1000
    model = RandomMinResize(min_sizes, max_size)

    # Create a 400x200 image (height x width)
    img = torch.zeros((3, 400, 200))

    # Create a simple polygon with a 100x100 bounding box
    polygons = np.array([
        [[50.0, 50.0], [150.0, 50.0], [150.0, 150.0], [50.0, 150.0]]
    ])
    target = {"polygons": polygons.copy()}

    # After resize, the width should become 300 (min_size) and height 600
    # So scaling ratio is 300/200 = 1.5
    expected_ratio = 1.5

    # Fix random choice to always select 300
    with patch("random.choice", return_value=300):
        _, resized_target = model(img, target)

    # Expected coordinates after scaling by 1.5
    expected_polygons = polygons * expected_ratio

    # Verify polygons were properly scaled
    np.testing.assert_allclose(resized_target["polygons"], expected_polygons)


def test_random_min_resize_preserves_aspect_ratio():
    """Test that aspect ratio is preserved during resize"""
    min_sizes = [400]
    max_size = 1000
    model = RandomMinResize(min_sizes, max_size)

    # Create different shaped images
    images = [
        torch.zeros((3, 300, 600)),  # h:w = 1:2
        torch.zeros((3, 600, 300)),  # h:w = 2:1
        torch.zeros((3, 450, 300)),  # h:w = 3:2
    ]

    for img in images:
        c, h, w = img.shape
        original_aspect = w / h

        target = {"polygons": np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32)}

        with patch("random.choice", return_value=400):
            resized_img, _ = model(img, target)

        # Check that aspect ratio is preserved
        c_new, h_new, w_new = resized_img.shape
        resized_aspect = w_new / h_new

        assert math.isclose(original_aspect, resized_aspect, rel_tol=1e-5)


def test_random_min_resize_max_size_constraint():
    """Test that max_size constraint is properly applied"""
    min_sizes = [300]
    max_size = 400
    model = RandomMinResize(min_sizes, max_size)

    # Create a very wide image, 100x800 (height x width)
    img = torch.zeros((3, 100, 800))
    target = {"polygons": np.array([[[0, 0], [800, 0], [800, 100], [0, 100]]], dtype=np.float32)}

    with patch("random.choice", return_value=300):
        resized_img, _ = model(img, target)

    # The short side (height) should be resized toward 300, but then adjusted down
    # to maintain aspect ratio when width hits max_size=400
    c, h, w = resized_img.shape

    assert w == 400  # Width should be capped at max_size
    assert h == 50   # Height should be adjusted to maintain aspect ratio
