import torch
import numpy as np
import pytest

from src.dbnet.datasets.transforms import HorizontalFlip


def test_horizontal_flip_basic():
    """Test basic functionality of HorizontalFlip transform."""
    transform = HorizontalFlip()

    # Create a test image with distinctive left and right sides
    # Left half (0-15) is 0, right half (16-31) is 1
    img = torch.zeros((3, 32, 32), dtype=torch.float32)
    img[:, :, 16:] = 1.0

    # Create a sample polygon
    polygons = np.array([
        [[3.0, 3.0], [6.0, 2.0], [5.0, 4.0], [1.0, 6.0]]
    ], dtype=np.float32)

    target = {"polygons": polygons.copy()}

    # Apply the transform
    flipped_img, flipped_target = transform(img, target)

    # Check image properties
    assert isinstance(flipped_img, torch.Tensor)
    assert flipped_img.shape == img.shape
    assert flipped_img.dtype == img.dtype

    # Check the image was properly flipped - left half should now be 1, right half 0
    assert torch.all(flipped_img[:, :, :16] == 1.0)
    assert torch.all(flipped_img[:, :, 16:] == 0.0)

    # Check polygon properties
    assert isinstance(flipped_target["polygons"], np.ndarray)
    assert flipped_target["polygons"].dtype == np.float32
    assert flipped_target["polygons"].shape == polygons.shape

    # Check x-coordinates were properly flipped (w - x)
    # For a 32-pixel wide image: new_x = 32 - old_x
    expected_polygons = np.array([
        [[29.0, 3.0], [26.0, 2.0], [27.0, 4.0], [31.0, 6.0]]
    ], dtype=np.float32)

    np.testing.assert_allclose(flipped_target["polygons"], expected_polygons)


def test_horizontal_flip_multiple_polygons():
    """Test HorizontalFlip with multiple polygons."""
    transform = HorizontalFlip()

    # Create test image
    img = torch.zeros((3, 50, 50), dtype=torch.float32)

    # Create multiple polygons
    polygons = np.array([
        [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]],  # Square in top-left
        [[30.0, 30.0], [40.0, 30.0], [40.0, 40.0], [30.0, 40.0]],  # Square in bottom-right
    ], dtype=np.float32)

    target = {"polygons": polygons.copy()}

    # Apply the transform
    _, flipped_target = transform(img, target)

    # Calculate expected flipped polygons (x = width - x)
    expected_polygons = polygons.copy()
    for i in range(len(polygons)):
        expected_polygons[i, :, 0] = 50 - polygons[i, :, 0]

    # Check all polygons were properly flipped
    np.testing.assert_allclose(flipped_target["polygons"], expected_polygons)


def test_horizontal_flip_preserve_other_target_fields():
    """Test that HorizontalFlip preserves other fields in the target dictionary."""
    transform = HorizontalFlip()

    # Create test image
    img = torch.zeros((3, 64, 64), dtype=torch.float32)

    # Create target with polygons and additional fields
    target = {
        "polygons": np.array([[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]], dtype=np.float32),
        "polygon_classes": np.array([1, 2, 3]),
        "additional_field": "preserved",
        "numeric_field": 42
    }

    # Apply the transform
    _, flipped_target = transform(img, target)

    # Check that other fields are preserved
    assert "polygon_classes" in flipped_target
    assert "additional_field" in flipped_target
    assert "numeric_field" in flipped_target

    assert flipped_target["polygon_classes"].tolist() == [1, 2, 3]
    assert flipped_target["additional_field"] == "preserved"
    assert flipped_target["numeric_field"] == 42


def test_horizontal_flip_empty_polygon():
    """Test HorizontalFlip with an empty polygon array."""
    transform = HorizontalFlip()

    # Create test image
    img = torch.zeros((3, 32, 32), dtype=torch.float32)

    # Create target with empty polygon array
    empty_polygons = np.zeros((0, 4, 2), dtype=np.float32)  # 0 polygons, 4 points per polygon, 2 coords per point
    target = {"polygons": empty_polygons}

    # Apply the transform
    _, flipped_target = transform(img, target)

    # Check that empty polygon array is still empty but properly shaped
    assert flipped_target["polygons"].shape == (0, 4, 2)
    assert flipped_target["polygons"].dtype == np.float32
