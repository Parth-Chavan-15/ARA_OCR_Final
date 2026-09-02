import pytest
import os
import random
import fitz  # PyMuPDF
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from types import SimpleNamespace

@pytest.fixture
def tmp_image(tmp_path):
    file_path = tmp_path / "valid_image.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(file_path)
    return str(file_path)

@pytest.fixture
def tmp_pdf(tmp_path):
    file_path = tmp_path / "valid_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test PDF")
    doc.save(file_path)
    doc.close()
    return str(file_path)

@pytest.fixture
def tmp_multipage_pdf(tmp_path):
    file_path = tmp_path / "multipage_doc.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((50, 50), f"Test PDF Page {i+1}")
    doc.save(file_path)
    doc.close()
    return str(file_path)

@pytest.fixture
def tmp_corrupt_file(tmp_path):
    file_path = tmp_path / "corrupt.pdf"
    with open(file_path, "wb") as f:
        f.write(os.urandom(1024))
    return str(file_path)

@pytest.fixture
def tmp_empty_file(tmp_path):
    file_path = tmp_path / "empty.pdf"
    file_path.touch()
    return str(file_path)

@pytest.fixture
def tmp_text_file(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a text file.")
    return str(file_path)

@pytest.fixture
def db_session():
    # Setup an in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()

@pytest.fixture
def sample_ocr_tokens():
    return [
        SimpleNamespace(text="Test", page_number=1, x1=10, y1=10, x2=50, y2=20, confidence=0.9),
        SimpleNamespace(text="Token", page_number=1, x1=60, y1=10, x2=100, y2=20, confidence=0.85)
    ]
