"""Data update coordinator for the Immich Random Image integration."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    DEFAULT_FACE_CROP_ASPECT_HEIGHT,
    DEFAULT_FACE_CROP_ASPECT_WIDTH,
    DEFAULT_FACE_CROP_ENABLED,
    DEFAULT_FACE_CROP_MAX_HEIGHT,
    DEFAULT_FACE_CROP_MAX_WIDTH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_REVERSE_GEOCODE,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    IMAGE_SIZE_PREVIEW,
    OPT_FACE_CROP_ASPECT_HEIGHT,
    OPT_FACE_CROP_ASPECT_WIDTH,
    OPT_FACE_CROP_ENABLED,
    OPT_FACE_CROP_MAX_HEIGHT,
    OPT_FACE_CROP_MAX_WIDTH,
    OPT_IMAGE_SIZE,
    OPT_REVERSE_GEOCODE,
)
from .hub import CannotConnect, ImmichRandomHub
from .image_processing import (
    compute_face_focus_point,
    crop_to_aspect_with_focus,
    resize_image,
)

_LOGGER = logging.getLogger(__name__)


class ImmichCoordinator(DataUpdateCoordinator):
    """Coordinator that fetches random images and shares data across entities.

    Exposes:
      - last_updated: when the last image was pulled
      - media_filename: the original filename
      - media_localdatetime: when the photo was taken
      - media_width / media_height: image dimensions
      - image_bytes: the raw image data (for the image entity)
      - album_ids: current album selection (mutable from select entity)
      - albums: the list of available albums (for the select entity options)
    """

    def __init__(
        self,
        hass: HomeAssistant,
        hub: ImmichRandomHub,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the coordinator."""
        self.hub = hub
        self.config_entry = config_entry
        self.image_bytes: bytes | None = None
        self.albums: list[dict] = []

        scan_interval = config_entry.options.get(
            "scan_interval", config_entry.data.get("scan_interval", DEFAULT_SCAN_INTERVAL)
        )

        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_{config_entry.entry_id[:8]}",
            config_entry=config_entry,
            update_interval=timedelta(seconds=int(scan_interval)) if scan_interval else None,
            update_method=self._async_update_data,
        )
        self._scan_interval_seconds = int(scan_interval)

    @property
    def scan_interval_seconds(self) -> int:
        """Return the configured scan interval in seconds."""
        return self._scan_interval_seconds

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch a new random image from Immich.

        Returns a dict with the latest image metadata.
        The raw image bytes are stored in self.image_bytes.
        """
        _LOGGER.debug("Coordinator fetching new random image from %s", self.hub.host)

        try:
            return await self._async_fetch_random_image()
        except CannotConnect as err:
            raise UpdateFailed(f"Error communicating with Immich: {err}") from err

    async def _async_fetch_random_image(self) -> dict[str, Any]:
        """Fetch and download one random image; keep the old data on a miss."""
        asset = await self.hub.get_random_image()
        if not asset:
            _LOGGER.warning("No random image returned from Immich")
            return self.data or {}

        asset_id = asset.get("id")
        if not asset_id:
            _LOGGER.warning("Random image returned without an asset ID")
            return self.data or {}

        _LOGGER.debug(
            "Got random image: id=%s, file=%s",
            asset_id,
            asset.get("originalFileName", "?"),
        )

        # Get options
        image_size = self.config_entry.options.get(OPT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        face_crop_enabled = self.config_entry.options.get(OPT_FACE_CROP_ENABLED, DEFAULT_FACE_CROP_ENABLED)
        reverse_geocode = self.config_entry.options.get(OPT_REVERSE_GEOCODE, DEFAULT_REVERSE_GEOCODE)

        # Download image (original or preview)
        use_preview = image_size == IMAGE_SIZE_PREVIEW
        image_bytes = await self.hub.download_asset(asset_id, use_preview=use_preview)
        if not image_bytes:
            _LOGGER.warning("Failed to download image %s from Immich", asset_id)
            return self.data or {}

        # Fetch detailed asset info for location and faces
        asset_info = await self.hub.get_asset_info(asset_id)

        # Extract location data
        location_data = await self._extract_location_data(asset_info, reverse_geocode)

        # Extract date taken
        date_taken = ""
        if asset_info and "exifInfo" in asset_info:
            exif_info = asset_info["exifInfo"]
            date_taken = exif_info.get("dateTimeOriginal", "") or asset.get("fileCreatedAt", "")

        # Process faces if face cropping is enabled
        focus_x, focus_y = 0.5, 0.5
        faces_detected = []

        if face_crop_enabled and asset_info:
            # Try to get faces from asset info (people array)
            people = asset_info.get("people", [])
            if people:
                for person in people:
                    person_faces = person.get("faces", [])
                    faces_detected.extend(person_faces)

            # If no faces in asset info, try faces endpoint
            if not faces_detected:
                faces_from_api = await self.hub.get_asset_faces(asset_id)
                if faces_from_api:
                    faces_detected = faces_from_api

            # Compute focus point from faces
            if faces_detected:
                orig_width = asset.get("width", 0)
                orig_height = asset.get("height", 0)
                if orig_width and orig_height:
                    focus_x, focus_y = compute_face_focus_point(faces_detected, orig_width, orig_height)
                    _LOGGER.debug("Computed focus point from %d faces: (%.2f, %.2f)", len(faces_detected), focus_x, focus_y)
            else:
                # No faces: use center, or slightly above center for portrait images
                orig_width = asset.get("width", 0)
                orig_height = asset.get("height", 0)
                if orig_height > orig_width:
                    # Portrait: focus slightly above center
                    focus_y = 0.45
                    _LOGGER.debug("Portrait mode, no faces: using focus (0.5, 0.45)")

        # Apply face-aware cropping if enabled
        final_width = asset.get("width", "")
        final_height = asset.get("height", "")

        if face_crop_enabled:
            aspect_width = self.config_entry.options.get(OPT_FACE_CROP_ASPECT_WIDTH, DEFAULT_FACE_CROP_ASPECT_WIDTH)
            aspect_height = self.config_entry.options.get(OPT_FACE_CROP_ASPECT_HEIGHT, DEFAULT_FACE_CROP_ASPECT_HEIGHT)
            max_width = self.config_entry.options.get(OPT_FACE_CROP_MAX_WIDTH, DEFAULT_FACE_CROP_MAX_WIDTH)
            max_height = self.config_entry.options.get(OPT_FACE_CROP_MAX_HEIGHT, DEFAULT_FACE_CROP_MAX_HEIGHT)

            try:
                # Run image processing in executor to avoid blocking event loop
                image_bytes, final_width, final_height = await self.hass.async_add_executor_job(
                    crop_to_aspect_with_focus,
                    image_bytes,
                    aspect_width,
                    aspect_height,
                    max_width,
                    max_height,
                    focus_x,
                    focus_y,
                )
            except Exception as err:
                _LOGGER.error("Face-aware cropping failed: %s, using original", err)
        elif use_preview:
            # If using preview but not cropping, still resize to reasonable dimensions
            try:
                image_bytes, final_width, final_height = await self.hass.async_add_executor_job(
                    resize_image,
                    image_bytes,
                    1440,
                    1440,
                )
            except Exception as err:
                _LOGGER.error("Image resize failed: %s, using original preview", err)

        self.image_bytes = image_bytes

        data: dict[str, Any] = {
            "asset_id": asset_id,
            "media_filename": asset.get("originalFileName", ""),
            "media_localdatetime": asset.get("localDateTime", ""),
            "media_width": final_width,
            "media_height": final_height,
            "last_pulled": datetime.now().isoformat(),
            "date_taken": date_taken,
            "focus_x": focus_x,
            "focus_y": focus_y,
            **location_data,
        }

        _LOGGER.info(
            "Updated Immich random image: %s (%sx%s, %s)",
            data["media_filename"],
            data["media_width"],
            data["media_height"],
            data["media_localdatetime"],
        )

        return data

    async def _extract_location_data(
        self, asset_info: dict | None, reverse_geocode: bool
    ) -> dict[str, Any]:
        """Extract location data from asset EXIF info.

        Returns a dict with city, state, country, latitude, longitude.
        """
        location_data = {
            "city": "",
            "state": "",
            "country": "",
            "latitude": "",
            "longitude": "",
        }

        if not asset_info:
            return location_data

        exif_info = asset_info.get("exifInfo", {})
        if not exif_info:
            return location_data

        # Extract GPS coordinates
        latitude = exif_info.get("latitude")
        longitude = exif_info.get("longitude")
        if latitude is not None and longitude is not None:
            location_data["latitude"] = latitude
            location_data["longitude"] = longitude

        # Extract location names from EXIF
        location_data["city"] = exif_info.get("city", "")
        location_data["state"] = exif_info.get("state", "")
        location_data["country"] = exif_info.get("country", "")

        # Optional: reverse geocode if GPS exists but no city
        # (Not implemented in this version - requires external API)
        if reverse_geocode and latitude and longitude and not location_data["city"]:
            _LOGGER.debug("Reverse geocoding not yet implemented")

        return location_data

    async def async_refresh_albums(self) -> None:
        """Fetch the list of available albums from Immich."""
        try:
            self.albums = await self.hub.list_all_albums()
            _LOGGER.debug("Fetched %d albums from Immich", len(self.albums))
        except Exception as err:
            _LOGGER.warning("Failed to fetch albums: %s", err)
            self.albums = []

    async def async_set_album_ids(self, album_ids: list[str]) -> None:
        """Update the active album selection (called by the select entity)."""
        self.hub.album_ids = album_ids
        _LOGGER.info("Album selection updated to %d albums, refreshing", len(album_ids))
        await self.async_request_refresh()
