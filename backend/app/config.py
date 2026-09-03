"""
ARA OCR — Application Configuration

Environment-based configuration using pydantic-settings.
All settings can be overridden via environment variables or a .env file.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/ara_ocr"

    # Data directories (relative to project root or absolute)
    DATA_DIR: Path = Path("./data")
    MODELS_DIR: Path = Path("./models")

    # Model versions — mandatory for traceability (§30)
    OCR_MODEL_VERSION: str = "paddleocr_mr_en_v1"
    CLASSIFIER_MODEL_VERSION: str = "layoutxlm_ara_v1"
    PIPELINE_VERSION: str = "0.1.0"

    # Classification confidence threshold (§19)
    # Below this threshold -> UNKNOWN / OUT_OF_SCOPE
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.55
    CONFIDENCE_THRESHOLD: float = 0.55

    # File upload limits
    MAX_FILE_SIZE_MB: int = 50

    # CORS origins for frontend dev server
    CORS_ORIGINS: str = "http://localhost:5173"

    # Hardware & GPU Acceleration
    USE_GPU: bool = True
    GPU_DEVICE_ID: int = 0

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = {
        "env_file": "../.env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def model_post_init(self, __context) -> None:
        """Resolve DATA_DIR and MODELS_DIR relative to the project root."""
        # Project root is the parent of the backend/ directory
        project_root = Path(__file__).resolve().parent.parent.parent
        if not self.DATA_DIR.is_absolute():
            object.__setattr__(self, "DATA_DIR", project_root / self.DATA_DIR)
        if not self.MODELS_DIR.is_absolute():
            object.__setattr__(self, "MODELS_DIR", project_root / self.MODELS_DIR)

    @property
    def originals_dir(self) -> Path:
        """Directory for original, unmodified documents."""
        return self.DATA_DIR / "originals"

    @property
    def processed_dir(self) -> Path:
        """Directory for preprocessed images."""
        return self.DATA_DIR / "processed"

    @property
    def demo_dir(self) -> Path:
        """Directory for demo/synthetic documents."""
        return self.DATA_DIR / "demo"

    @property
    def layoutxlm_dir(self) -> Path:
        """Directory for LayoutXLM model weights."""
        return self.MODELS_DIR / "layoutxlm"

    @property
    def max_file_size_bytes(self) -> int:
        """Maximum file size in bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS origins as a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        for directory in [
            self.originals_dir,
            self.processed_dir,
            self.demo_dir,
            self.layoutxlm_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
