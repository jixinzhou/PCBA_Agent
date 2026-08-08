from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pt"

API_TITLE = "PCBA Defect Classification API"
API_VERSION = "v1"
TOOL_NAME = "pcba_defect_classification"
TOOL_VERSION = "0.1.0"
MODEL_NAME = "efficientnet_b0"
MODEL_VERSION = "0.1.0"
LABEL_SCHEMA_VERSION = "0.1.0"

DEVICE = "auto"
CONFIDENCE_THRESHOLD = 0.6
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}

