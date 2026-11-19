"""Constants for the Immich Random Image integration."""

DOMAIN = "immich_random"

# Config entry data keys
CONF_ALBUM_IDS = "album_ids"
CONF_VERIFY_SSL = "verify_ssl"

# Options keys
OPT_ALBUM_IDS = "album_ids"
OPT_VERIFY_SSL = "verify_ssl"
OPT_SCAN_INTERVAL = "scan_interval"
OPT_IMAGE_SIZE = "image_size"
OPT_FACE_CROP_ENABLED = "face_crop_enabled"
OPT_FACE_CROP_ASPECT_WIDTH = "face_crop_aspect_width"
OPT_FACE_CROP_ASPECT_HEIGHT = "face_crop_aspect_height"
OPT_FACE_CROP_MAX_WIDTH = "face_crop_max_width"
OPT_FACE_CROP_MAX_HEIGHT = "face_crop_max_height"
OPT_REVERSE_GEOCODE = "reverse_geocode"

# Default scan interval in seconds (5 minutes)
DEFAULT_SCAN_INTERVAL = 300

# Recommended range for scan interval
MIN_SCAN_INTERVAL = 0  # Manual-only rotation
MAX_SCAN_INTERVAL = 86400  # 24 hours
RECOMMENDED_MIN_SCAN_INTERVAL = 60
RECOMMENDED_MAX_SCAN_INTERVAL = 300

# Image size options
IMAGE_SIZE_ORIGINAL = "original"
IMAGE_SIZE_PREVIEW = "preview"
DEFAULT_IMAGE_SIZE = IMAGE_SIZE_PREVIEW

# Face crop defaults
DEFAULT_FACE_CROP_ENABLED = False
DEFAULT_FACE_CROP_ASPECT_WIDTH = 800
DEFAULT_FACE_CROP_ASPECT_HEIGHT = 480
DEFAULT_FACE_CROP_MAX_WIDTH = 800
DEFAULT_FACE_CROP_MAX_HEIGHT = 480
DEFAULT_REVERSE_GEOCODE = False

# Service: manual refresh
SERVICE_REFRESH = "refresh"
ATTR_ENTRY_ID = "entry_id"
