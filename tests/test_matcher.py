import cv2, os, pytest
from war_drone.recognizers.matcher import match_once

def test_matcher_basic(maybe_img):
    big = maybe_img("tests/assets/2_prebattle.jpg")
    tpl = maybe_img("templates/start_btn_prebattle.png")
    ok, _, score = match_once(big, tpl, threshold=0.75)
    assert ok, f"score={score}"
