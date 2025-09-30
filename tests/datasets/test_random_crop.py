from unittest.mock import patch

import numpy as np
import pytest
import torch

from dbnet.datasets.transforms import RandomCrop


class TestRandomCrop:
    @pytest.fixture
    def sample_data(self):
        """Create sample image and target data for testing."""
        # Create a 100x200 sample image
        img = torch.zeros((3, 100, 200), dtype=torch.float32)

        # Add some shapes to the image to make it visually distinguishable
        img[:, 20:40, 50:150] = 1.0  # Rectangle in the center

        # Create polygon annotations
        polygons = np.array([
            # Polygon in the top-left
            [[10, 10], [30, 10], [30, 20], [10, 20]],

            # Polygon in the center
            [[80, 80], [120, 80], [120, 90], [80, 90]],

            # Polygon in the bottom-right
            [[150, 60], [180, 60], [180, 80], [150, 80]]
        ], dtype=np.float32)

        target = {
            "polygons": polygons,
            "polygon_ignores": [False, False, False],
            "polygon_classes": [1, 2, 3]
        }

        return img, target

    def test_random_crop_basic(self, sample_data):
        """Test basic functionality of RandomCrop transform."""
        img, target = sample_data

        # Create transform with fixed size
        transform = RandomCrop(size=(100, 100), max_tries=10)

        # Mock the _crop_area method to return a fixed crop region
        # This makes the test deterministic
        with patch.object(transform, '_crop_area', return_value=(50, 20, 100, 80)):
            cropped_img, cropped_target = transform(img, target)

            # Check output image shape matches the specified crop size
            assert cropped_img.shape == (3, 100, 100)

            # Check that target has been updated
            assert "polygons" in cropped_target
            assert "polygon_ignores" in cropped_target
            assert "polygon_classes" in cropped_target

            # Check scale values were added
            assert "scale_w" in cropped_target
            assert "scale_h" in cropped_target

    def test_random_crop_polygon_adjustment(self, sample_data):
        """Test that polygon coordinates are properly adjusted after cropping."""
        img, target = sample_data

        # Create transform
        transform = RandomCrop(size=(150, 150))

        # Force a specific crop area - cropping from (50,20) with size 100x80
        with patch.object(transform, '_crop_area', return_value=(50, 20, 100, 80)):
            orig_target = target.copy()

            # For a crop from (50,20) with size (100,80) and target size (150,150),
            # the scaling should be min(150/100, 150/80) = min(1.5, 1.875) = 1.5
            _, cropped_target = transform(img, target)

            # Only the polygon in the center and bottom-right should remain
            # The top-left polygon should be outside the crop
            assert len(cropped_target["polygons"]) <= len(target["polygons"])

            # For remaining polygons, check that coordinates were adjusted
            # Polygon coordinates should be shifted by (-50,-20) and scaled by 1.5
            for original_poly, cropped_poly in zip(
                    orig_target["polygons"][1:],
                    cropped_target["polygons"], strict=False):
                expected_poly = (original_poly - np.array([50, 20])) * 1.5
                np.testing.assert_allclose(cropped_poly, expected_poly, rtol=1e-5)

    def test_random_crop_fallback(self, sample_data):
        """Test fallback to original image when no valid crop is found."""
        img, target = sample_data
        _, h, w = img.shape

        # Create transform
        transform = RandomCrop(size=(150, 150), max_tries=1)

        # Mock _crop_area to simulate no valid crop found (returns full image)
        with patch.object(transform, '_crop_area', return_value=(0, 0, w, h)):
            cropped_img, cropped_target = transform(img, target)

            # The output should be the original image resized and padded to (150,150)
            assert cropped_img.shape == (3, 150, 150)

            # The scale should be min(150/200, 150/100) = min(0.75, 1.5) = 0.75
            expected_scale = 0.75
            assert cropped_target["scale_w"] == pytest.approx(expected_scale)
            assert cropped_target["scale_h"] == pytest.approx(expected_scale)

            # Verify that all polygons are still present
            assert len(cropped_target["polygons"]) == len(target["polygons"])

    def test_crop_area_method(self, sample_data):
        """Test the _crop_area method directly."""
        img, target = sample_data

        transform = RandomCrop()

        # Create a simpler test case with one polygon
        test_polygon = np.array([[[50, 50], [150, 50], [150, 70], [50, 70]]], dtype=np.float32)

        # Use a mock for random selection to make the test deterministic
        with patch('numpy.random.choice', side_effect=lambda arr, size=None: np.array([arr[0]])):
            with patch.object(transform, '_random_select', return_value=(10, 40)):
                with patch.object(transform, '_region_wise_random_select', return_value=(10, 40)):
                    # The crop area should avoid the text region (50-150, 50-70)
                    x, y, w, h = transform._crop_area(img, test_polygon)

                    # Crop should be in a valid area
                    assert x >= 0 and y >= 0
                    assert x + w <= img.shape[2]
                    assert y + h <= img.shape[1]

    def test_is_poly_outside_rect(self):
        """Test the _is_poly_outside_rect method."""
        transform = RandomCrop()

        # Polygon completely outside rectangle (left)
        poly1 = [[1, 5], [5, 5], [5, 10], [1, 10]]
        assert transform._is_poly_outside_rect(poly1, 10, 0, 50, 50) is True

        # Polygon completely outside rectangle (right)
        poly2 = [[60, 5], [70, 5], [70, 10], [60, 10]]
        assert transform._is_poly_outside_rect(poly2, 0, 0, 50, 50) is True

        # Polygon completely outside rectangle (top)
        poly3 = [[20, 1], [30, 1], [30, 5], [20, 5]]
        assert transform._is_poly_outside_rect(poly3, 0, 10, 50, 50) is True

        # Polygon completely outside rectangle (bottom)
        poly4 = [[20, 60], [30, 60], [30, 70], [20, 70]]
        assert transform._is_poly_outside_rect(poly4, 0, 0, 50, 50) is True

        # Polygon partially inside rectangle
        poly5 = [[40, 40], [60, 40], [60, 60], [40, 60]]
        assert transform._is_poly_outside_rect(poly5, 0, 0, 50, 50) is False

        # Polygon completely inside rectangle
        poly6 = [[10, 10], [20, 10], [20, 20], [10, 20]]
        assert transform._is_poly_outside_rect(poly6, 0, 0, 50, 50) is False
