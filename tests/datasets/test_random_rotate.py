import torch
import numpy as np
import pytest
from unittest.mock import patch
import math

from src.dbnet.datasets.transforms import RandomRotate


class TestRandomRotate:
    @pytest.fixture
    def sample_data(self):
        """Create sample image and target data for testing."""
        # Create a 100x100 sample image
        img = torch.zeros((3, 100, 100), dtype=torch.float32)

        # Add some shapes to the image to make it visually distinguishable
        img[:, 40:60, 40:60] = 1.0  # Square in the center

        # Create polygon annotations
        polygons = np.array([
            # Square in the center
            [[40, 40], [60, 40], [60, 60], [40, 60]],
            # Rectangle near the edge
            [[10, 10], [30, 10], [30, 20], [10, 20]],
        ], dtype=np.float32)

        target = {
            "polygons": polygons,
            "polygon_ignores": [False, False, False],
            "polygon_classes": [1, 2, 3]
        }

        return img, target

    def test_random_rotate_basic(self, sample_data):
        """Test basic functionality of RandomRotate transform."""
        img, target = sample_data

        # Create transform with fixed angle range
        transform = RandomRotate(degrees=(-30, 30))

        # Mock random.randint to return a fixed angle (15 degrees)
        with patch('random.randint', return_value=15):
            rotated_img, rotated_target = transform(img, target)

            # Check output image shape matches the input shape
            assert rotated_img.shape == img.shape

            # Check that target has been updated
            assert "polygons" in rotated_target
            assert "polygon_ignores" in rotated_target
            assert "polygon_classes" in rotated_target

    def test_rotate_poly_method(self):
        """Test the _rotate_poly method directly."""
        transform = RandomRotate(degrees=(-30, 30))

        # Create a square polygon centered at the origin
        polygon = np.array([
            [10, 10], [20, 10], [20, 20], [10, 20]
        ], dtype=np.float32)

        # Rotate 90 degrees around center (15, 15)
        angle = 90
        center = (15, 15)
        rotated_poly = transform._rotate_poly(polygon, angle, center)

        # Expected result after 90 degree rotation around (15, 15)
        # [10, 10] -> [10, 20]
        # [20, 10] -> [10, 10]
        # [20, 20] -> [20, 10]
        # [10, 20] -> [20, 20]
        expected_poly = np.array([
            [10, 20], [10, 10], [20, 10], [20, 20]
        ], dtype=np.float32)

        # Allow for small floating point errors
        np.testing.assert_allclose(rotated_poly, expected_poly, atol=1e-5)

    def test_random_rotate_polygon_preservation(self, sample_data):
        """Test that polygons remain in the image boundary after rotation."""
        img, target = sample_data

        # Create transform with small angle range to ensure polygons stay in the image
        transform = RandomRotate(degrees=(-10, 10))

        # Mock random.randint to return a fixed angle (5 degrees)
        with patch('random.randint', return_value=5):
            _, rotated_target = transform(img, target)

            # Center polygon should still be inside the image
            assert len(rotated_target["polygons"]) >= 1

            # Check that the center polygon is rotated correctly
            # For a small angle like 5 degrees, the center square should still be present
            center_square_found = False
            for polygon in rotated_target["polygons"]:
                # Check if this polygon is roughly in the center
                x_center = np.mean(polygon[:, 0])
                y_center = np.mean(polygon[:, 1])

                if 45 <= x_center <= 55 and 45 <= y_center <= 55:
                    center_square_found = True
                    break

            assert center_square_found

    def test_is_poly_outside_rect(self):
        """Test the _is_poly_outside_rect method."""
        transform = RandomRotate(degrees=(-30, 30))

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

    def test_random_rotate_large_angle(self, sample_data):
        """Test rotation with a large angle that might move polygons outside the image."""
        img, target = sample_data

        # Create transform with large angle range
        transform = RandomRotate(degrees=(-90, 90))

        # Mock random.randint to return a fixed large angle (75 degrees)
        with patch('random.randint', return_value=75):
            _, rotated_target = transform(img, target)

            # Check that some polygons may be filtered out if they go outside the image
            assert len(rotated_target["polygons"]) <= len(target["polygons"])

            # For each remaining polygon, check it's inside the image boundaries
            for polygon in rotated_target["polygons"]:
                assert not transform._is_poly_outside_rect(polygon, 0, 0, 100, 100)
