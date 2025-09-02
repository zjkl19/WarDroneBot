# tests/conftest.py
import pytest
from war_drone.state_detector import TemplateStateDetector

@pytest.fixture(scope="session")
def detector():
    # 与生产一致：掩码 + 可选边缘 + CCORR_NORMED
    return TemplateStateDetector(
        cfg_path="configs/config.json5",
        templates_dir="templates",
        use_edges=True,
        use_mask=True,
        method="CCORR_NORMED",
        default_thresh=0.85,  # 这里对烟雾测试无影响
    )
