import pytest
import requests
from io import BytesIO


@pytest.fixture(scope="session")
def mock_image_stream():
    url = "https://miro.medium.com/max/3349/1*mk1-6aYaf_Bes1E3Imhc0A.jpeg"
    return requests.get(url).content


@pytest.fixture(scope="session")
def mock_image_path(mock_image_stream, tmpdir_factory):
    file = BytesIO(mock_image_stream)
    folder = tmpdir_factory.mktemp("images")
    fn = folder.join("mock_image_file.jpeg")
    with open(fn, "wb") as f:
        f.write(file.getbuffer())
    return str(fn)
