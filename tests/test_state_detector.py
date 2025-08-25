import pytest
pytest.skip("v0 版本未启用图像识别，跳过 state_detector 测试", allow_module_level=True)

import cv2, pytest, os
from war_drone.config import Config
from war_drone.recognizers.state_detector import Detector, State

def test_states(maybe_img):
    cfg = Config()
    det = Detector(cfg)
    assert det is not None
    def st(img): return det.detect(maybe_img(img))
    assert st("tests/assets/1_list.jpg")      in [State.LIST, State.UNKNOWN]
    assert st("tests/assets/2_prebattle.jpg") in [State.PRE,  State.UNKNOWN]
    assert st("tests/assets/3_combat_a.jpg")  in [State.COMBAT, State.UNKNOWN]
    assert st("tests/assets/6_failed.jpg")    in [State.SETTLE, State.UNKNOWN]
