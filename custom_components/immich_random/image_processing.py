"""Image processing utilities for face-aware cropping."""
from __future__ import annotations

import io
import logging
from typing import Any

from PIL import Image

_LOGGER = logging.getLogger(__name__)


def compute_face_focus_point(
    faces: list[dict[str, Any]], image_width: int, image_height: int
) -> tuple[float, float]:
    """Compute the focus point based on detected faces.

    Returns (focus_x, focus_y) as fractions 0.0-1.0 of image dimensions.
    If no faces, returns center (0.5, 0.5).
    """
    if not faces:
        return (0.5, 0.5)

    # Compute union bounding box of all faces
    min_x = min(face["boundingBoxX1"] for face in faces)
    max_x = max(face["boundingBoxX2"] for face in faces)
    min_y = min(face["boundingBoxY1"] for face in faces)
    max_y = max(face["boundingBoxY2"] for face in faces)

    # Center of the union box
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    # Convert to 0-1 fractions
    focus_x = center_x / image_width if image_width > 0 else 0.5
    focus_y = center_y / image_height if image_height > 0 else 0.5

    return (focus_x, focus_y)


def crop_to_aspect_with_focus(
    image_bytes: bytes,
    target_aspect_width: int,
    target_aspect_height: int,
    max_width: int,
    max_height: int,
    focus_x: float = 0.5,
    focus_y: float = 0.5,
) -> tuple[bytes, int, int]:
    """Crop and resize an image to target aspect ratio, centered on focus point.

    Args:
        image_bytes: Input image as bytes
        target_aspect_width: Target aspect ratio width (e.g., 800)
        target_aspect_height: Target aspect ratio height (e.g., 480)
        max_width: Maximum output width in pixels
        max_height: Maximum output height in pixels
        focus_x: Focus point X as fraction 0.0-1.0 (0.5 = center)
        focus_y: Focus point Y as fraction 0.0-1.0 (0.5 = center, slightly higher for portraits)

    Returns:
        Tuple of (cropped_bytes, final_width, final_height)
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        orig_width, orig_height = image.size

        # Target aspect ratio
        target_aspect = target_aspect_width / target_aspect_height
        orig_aspect = orig_width / orig_height

        # Determine crop box
        if orig_aspect > target_aspect:
            # Image is wider than target: crop horizontally
            crop_height = orig_height
            crop_width = int(crop_height * target_aspect)
        else:
            # Image is taller than target: crop vertically
            crop_width = orig_width
            crop_height = int(crop_width / target_aspect)

        # Clamp crop dimensions
        crop_width = min(crop_width, orig_width)
        crop_height = min(crop_height, orig_height)

        # Center the crop on the focus point
        center_x = int(focus_x * orig_width)
        center_y = int(focus_y * orig_height)

        left = center_x - crop_width // 2
        top = center_y - crop_height // 2

        # Clamp to image bounds
        if left < 0:
            left = 0
        elif left + crop_width > orig_width:
            left = orig_width - crop_width

        if top < 0:
            top = 0
        elif top + crop_height > orig_height:
            top = orig_height - crop_height

        # Crop
        cropped = image.crop((left, top, left + crop_width, top + crop_height))

        # Resize to max dimensions while preserving aspect ratio
        cropped.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        final_width, final_height = cropped.size

        # Convert to bytes
        output = io.BytesIO()
        # Save as JPEG with reasonable quality
        if cropped.mode in ("RGBA", "LA", "P"):
            cropped = cropped.convert("RGB")
        cropped.save(output, format="JPEG", quality=85, optimize=True)
        output.seek(0)

        _LOGGER.debug(
            "Cropped image from %dx%d to %dx%d (focus: %.2f, %.2f)",
            orig_width,
            orig_height,
            final_width,
            final_height,
            focus_x,
            focus_y,
        )

        return (output.read(), final_width, final_height)

    except Exception as err:
        _LOGGER.error("Failed to process image: %s", err)
        raise


def resize_image(
    image_bytes: bytes, max_width: int = 1440, max_height: int = 1440
) -> tuple[bytes, int, int]:
    """Resize an image to fit within max dimensions, preserving aspect ratio.

    Args:
        image_bytes: Input image as bytes
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels

    Returns:
        Tuple of (resized_bytes, final_width, final_height)
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        orig_width, orig_height = image.size

        # Resize if needed
        if orig_width > max_width or orig_height > max_height:
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            final_width, final_height = image.size
        else:
            final_width, final_height = orig_width, orig_height

        # Convert to bytes
        output = io.BytesIO()
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        image.save(output, format="JPEG", quality=85, optimize=True)
        output.seek(0)

        _LOGGER.debug(
            "Resized image from %dx%d to %dx%d", orig_width, orig_height, final_width, final_height
        )

        return (output.read(), final_width, final_height)

    except Exception as err:
        _LOGGER.error("Failed to resize image: %s", err)
        raise
