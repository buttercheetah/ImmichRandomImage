"""Tests for image processing utilities."""
import io
import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

# Stub homeassistant modules before importing our code
_ha_stubs = [
    "homeassistant",
    "homeassistant.config_entries",
    "homeassistant.const",
    "homeassistant.core",
    "homeassistant.exceptions",
    "homeassistant.helpers",
    "homeassistant.helpers.aiohttp_client",
    "homeassistant.helpers.config_validation",
    "homeassistant.helpers.entity_platform",
    "homeassistant.helpers.selector",
    "homeassistant.helpers.update_coordinator",
    "homeassistant.components",
    "homeassistant.components.image",
    "homeassistant.components.sensor",
    "homeassistant.data_entry_flow",
]
for stub_name in _ha_stubs:
    if stub_name not in sys.modules:
        sys.modules[stub_name] = MagicMock()

# Now we can import our module
from custom_components.immich_random.image_processing import (
    compute_face_focus_point,
    crop_to_aspect_with_focus,
    resize_image,
)


def create_test_image(width: int, height: int, color: str = "red") -> bytes:
    """Create a test image as bytes."""
    image = Image.new("RGB", (width, height), color)
    output = io.BytesIO()
    image.save(output, format="JPEG")
    output.seek(0)
    return output.read()


def test_compute_face_focus_point_no_faces():
    """Test focus point computation with no faces."""
    focus_x, focus_y = compute_face_focus_point([], 1000, 1000)
    assert focus_x == 0.5
    assert focus_y == 0.5


def test_compute_face_focus_point_single_face():
    """Test focus point computation with a single face."""
    faces = [
        {
            "boundingBoxX1": 100,
            "boundingBoxY1": 100,
            "boundingBoxX2": 200,
            "boundingBoxY2": 200,
        }
    ]
    focus_x, focus_y = compute_face_focus_point(faces, 1000, 1000)
    # Center of box is at (150, 150)
    assert focus_x == pytest.approx(0.15, abs=0.01)
    assert focus_y == pytest.approx(0.15, abs=0.01)


def test_compute_face_focus_point_multiple_faces():
    """Test focus point computation with multiple faces."""
    faces = [
        {
            "boundingBoxX1": 100,
            "boundingBoxY1": 100,
            "boundingBoxX2": 200,
            "boundingBoxY2": 200,
        },
        {
            "boundingBoxX1": 300,
            "boundingBoxY1": 150,
            "boundingBoxX2": 400,
            "boundingBoxY2": 250,
        },
    ]
    focus_x, focus_y = compute_face_focus_point(faces, 1000, 1000)
    # Union box: (100, 100) to (400, 250)
    # Center: (250, 175)
    assert focus_x == pytest.approx(0.25, abs=0.01)
    assert focus_y == pytest.approx(0.175, abs=0.01)


def test_crop_to_aspect_landscape_to_portrait():
    """Test cropping a landscape image to portrait aspect ratio."""
    # Create a 1600x900 landscape image
    image_bytes = create_test_image(1600, 900)
    
    # Crop to 9:16 portrait aspect ratio (e.g., 480x800)
    cropped_bytes, width, height = crop_to_aspect_with_focus(
        image_bytes,
        target_aspect_width=480,
        target_aspect_height=800,
        max_width=480,
        max_height=800,
        focus_x=0.5,
        focus_y=0.5,
    )
    
    assert isinstance(cropped_bytes, bytes)
    assert width <= 480
    assert height <= 800
    # Should be portrait
    assert height > width


def test_crop_to_aspect_portrait_to_landscape():
    """Test cropping a portrait image to landscape aspect ratio."""
    # Create a 900x1600 portrait image
    image_bytes = create_test_image(900, 1600)
    
    # Crop to 5:3 landscape aspect ratio (800x480)
    cropped_bytes, width, height = crop_to_aspect_with_focus(
        image_bytes,
        target_aspect_width=800,
        target_aspect_height=480,
        max_width=800,
        max_height=480,
        focus_x=0.5,
        focus_y=0.4,  # Focus higher for portrait
    )
    
    assert isinstance(cropped_bytes, bytes)
    assert width <= 800
    assert height <= 480
    # Should be landscape
    assert width > height


def test_crop_with_off_center_focus():
    """Test cropping with focus point off-center."""
    # Create a 2000x2000 square image
    image_bytes = create_test_image(2000, 2000)
    
    # Crop to landscape 16:9 with focus in upper-left
    cropped_bytes, width, height = crop_to_aspect_with_focus(
        image_bytes,
        target_aspect_width=1600,
        target_aspect_height=900,
        max_width=1600,
        max_height=900,
        focus_x=0.3,
        focus_y=0.3,
    )
    
    assert isinstance(cropped_bytes, bytes)
    assert width <= 1600
    assert height <= 900


def test_resize_image_large_to_small():
    """Test resizing a large image to fit within max dimensions."""
    # Create a 3000x2000 image
    image_bytes = create_test_image(3000, 2000)
    
    # Resize to fit within 1440x1440
    resized_bytes, width, height = resize_image(image_bytes, 1440, 1440)
    
    assert isinstance(resized_bytes, bytes)
    assert width <= 1440
    assert height <= 1440
    # Aspect ratio should be preserved (3:2)
    assert abs((width / height) - 1.5) < 0.1


def test_resize_image_already_small():
    """Test that small images are not upscaled."""
    # Create a 800x600 image
    image_bytes = create_test_image(800, 600)
    
    # Resize with max 1440x1440 (should not upscale)
    resized_bytes, width, height = resize_image(image_bytes, 1440, 1440)
    
    assert isinstance(resized_bytes, bytes)
    assert width == 800
    assert height == 600


def test_crop_output_size_limit():
    """Test that cropped images respect max output dimensions."""
    # Create a very large 6000x4000 image
    image_bytes = create_test_image(6000, 4000)
    
    # Crop to 800x480 max
    cropped_bytes, width, height = crop_to_aspect_with_focus(
        image_bytes,
        target_aspect_width=800,
        target_aspect_height=480,
        max_width=800,
        max_height=480,
    )
    
    assert isinstance(cropped_bytes, bytes)
    assert width <= 800
    assert height <= 480
    # File size should be reasonable (less than 500 KB for 800x480 JPEG)
    assert len(cropped_bytes) < 500 * 1024


def test_crop_rgba_to_rgb_conversion():
    """Test that RGBA images are converted to RGB."""
    # Create an RGBA image
    image = Image.new("RGBA", (1000, 1000), (255, 0, 0, 128))
    output = io.BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    image_bytes = output.read()
    
    # Crop it
    cropped_bytes, width, height = crop_to_aspect_with_focus(
        image_bytes,
        target_aspect_width=800,
        target_aspect_height=600,
        max_width=800,
        max_height=600,
    )
    
    # Should successfully convert and crop
    assert isinstance(cropped_bytes, bytes)
    assert len(cropped_bytes) > 0
    
    # Verify it's a valid JPEG
    cropped_image = Image.open(io.BytesIO(cropped_bytes))
    assert cropped_image.format == "JPEG"
    assert cropped_image.mode == "RGB"
