import pytest
from app.services.file_handler import FileHandler


def test_valid_jpg(tmp_image):
    result = FileHandler.validate_file(tmp_image)
    assert result.status == "VALID"


def test_valid_png(tmp_path):
    from PIL import Image
    file_path = tmp_path / "valid_image.png"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(file_path)
    result = FileHandler.validate_file(str(file_path))
    assert result.status == "VALID"


def test_valid_pdf(tmp_pdf):
    result = FileHandler.validate_file(tmp_pdf)
    assert result.status == "VALID"


def test_multipage_pdf(tmp_multipage_pdf):
    result = FileHandler.validate_file(tmp_multipage_pdf)
    assert result.status == "VALID"


def test_corrupt_file(tmp_corrupt_file):
    result = FileHandler.validate_file(tmp_corrupt_file)
    assert result.status in ("UNREADABLE", "INVALID")


def test_empty_file(tmp_empty_file):
    result = FileHandler.validate_file(tmp_empty_file)
    assert result.status == "EMPTY"


def test_unsupported_type(tmp_text_file):
    result = FileHandler.validate_file(tmp_text_file)
    assert result.status == "INVALID"


def test_nonexistent_file():
    result = FileHandler.validate_file("does_not_exist.pdf")
    assert result.status == "INVALID"
