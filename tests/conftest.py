import os, pytest, cv2

@pytest.fixture(scope="session")
def assets():
    base = "tests/assets"
    return { k: os.path.join(base,k) for k in os.listdir(base) }

@pytest.fixture(scope="session")
def maybe_img():
    def _load(path):
        if not os.path.exists(path):
            pytest.skip(f"missing asset: {path}")
        return cv2.imread(path)
    return _load
