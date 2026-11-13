import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crowd_monitor import (HOLD, TO_LEFT, TO_RIGHT, CrowdAnalyser, DecisionLock, DensityDetector,
                           Detection, Ground, classify, density_from_detections, redirect_decision,
                           resize_density)

ARGS = SimpleNamespace(low=5, mid=12, warn_density=2.0, crit_density=4.0, hysteresis=0.1,
                       min_hold=5.0)


def det(x, y=50):
    return Detection(x, y, 10, 20, 0.9)


# a 10 x 6 m rectangle seen in perspective: wide at the bottom (near), narrow at the top (far)
TRAPEZOID = [[40, 380], [600, 380], [460, 120], [180, 120]]
SHAPE = (480, 640)


def test_classify_thresholds():
    assert classify(5, 5, 12) == "LOW"
    assert classify(6, 5, 12) == "MEDIUM"
    assert classify(12, 5, 12) == "MEDIUM"
    assert classify(13, 5, 12) == "HIGH"
    assert classify(4.5, 2.0, 4.0) == "HIGH"


def test_redirect_away_from_busy_side():
    assert redirect_decision(6, 1, 2) == TO_RIGHT
    assert redirect_decision(1, 6, 2) == TO_LEFT
    assert redirect_decision(3, 1, 2) == HOLD  # within margin


def test_decision_lock_holds_for_cooldown():
    t = [0.0]
    lock = DecisionLock(cooldown=2.0, clock=lambda: t[0])
    t[0] = 2.5
    assert lock.update(TO_LEFT) == TO_LEFT
    t[0] = 3.0
    assert lock.update(TO_RIGHT) == TO_LEFT  # too soon to flip
    t[0] = 4.6
    assert lock.update(TO_RIGHT) == TO_RIGHT


def test_density_from_detections_sums_to_count():
    d = density_from_detections([det(50), det(100), det(150, 60)], (100, 200))
    assert d.sum() == pytest.approx(3, abs=0.05)
    assert d[50, 50] > d[50, 75]  # mass sits on the people


def test_density_mass_sits_on_the_head_not_the_box_centre():
    d = density_from_detections([Detection(100, 200, 40, 100, 0.9)], (400, 300))
    assert int(d[:, 100].argmax()) == 160


def test_analyser_redirect_margin_scales_with_crowd_size():
    g = Ground(SHAPE)
    left, right = g.zone_masks()
    uniform = np.ones(SHAPE, dtype=np.float32) / (SHAPE[0] * SHAPE[1] / 2)
    t = [100.0]
    an = CrowdAnalyser(g, ARGS, clock=lambda: t[0])
    t[0] += 3
    # 160 vs 140 differs by far more than 2 people, but is only ~7% of the crowd
    assert an.update(uniform * (160 * left + 140 * right))["decision"] == HOLD


def test_resize_density_preserves_count():
    d = np.random.default_rng(0).random((30, 40)).astype(np.float32)
    assert resize_density(d, (320, 240)).sum() == pytest.approx(d.sum(), rel=1e-4)


def test_calibrated_ground_area_matches_real_area():
    g = Ground(SHAPE, TRAPEZOID, 10, 6)
    assert g.total_area == pytest.approx(60, rel=0.03)
    # perspective: a far pixel covers more ground than a near one
    assert g.pixel_area[130, 320] > g.pixel_area[370, 320]
    left, right = g.zone_masks()
    # the centre line halves the real ground, not the image
    assert (g.pixel_area * left).sum() == pytest.approx(30, rel=0.05)
    assert (g.pixel_area * right).sum() == pytest.approx(30, rel=0.05)


def test_uniform_crowd_reads_its_true_density():
    g = Ground(SHAPE, TRAPEZOID, 10, 6)
    density = g.pixel_area * 3.0  # exactly 3 people per m^2 everywhere in the ROI
    local = g.local_density(density, sigma=12)
    inner = local[(g.roi > 0) & (local > 0)]
    assert np.median(inner) == pytest.approx(3.0, rel=0.02)


def test_ground_load_rescales_points(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"image_points": [[v / 2 for v in pt] for pt in TRAPEZOID],
                             "width_m": 10, "length_m": 6, "frame_size": [320, 240]}))
    assert Ground.load(p, SHAPE).total_area == pytest.approx(60, rel=0.03)


def test_analyser_calibrated_flags_dense_side():
    g = Ground(SHAPE, TRAPEZOID, 10, 6)
    left, _ = g.zone_masks()
    t = [100.0]
    an = CrowdAnalyser(g, ARGS, clock=lambda: t[0])
    t[0] += 3  # past the decision cooldown
    stats = an.update(g.pixel_area * left * 5.0)  # 5 people/m^2 on the left half only
    assert stats["people"] == pytest.approx(150, rel=0.05)
    assert stats["left_d"] == pytest.approx(5.0, rel=0.05)
    assert stats["right_d"] == pytest.approx(0, abs=0.05)
    assert stats["density"] == "HIGH"
    assert stats["decision"] == TO_RIGHT


def test_analyser_uncalibrated_uses_counts():
    g = Ground(SHAPE)
    an = CrowdAnalyser(g, ARGS, clock=lambda: 100.0)
    stats = an.update(density_from_detections([det(100, 200)] * 3 + [det(500, 200)], SHAPE))
    assert stats["people"] == pytest.approx(4, abs=0.1)
    assert stats["density"] == "LOW"
    assert stats["decision"] == HOLD  # 3 vs 1 is within the margin of 2


class FakeNet(torch.nn.Module):
    """Stands in for CSRNet: 1/8-resolution output with a known total."""

    def forward(self, x):
        b, _, h, w = x.shape
        return torch.full((b, 1, h // 8, w // 8), 7.0 / ((h // 8) * (w // 8)))


def test_density_detector_returns_full_res_map_with_count():
    det_ = DensityDetector(model=FakeNet(), max_side=320)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t = det_.preprocess(frame)
    assert t.shape[2] % 16 == 0 and t.shape[3] % 16 == 0 and max(t.shape[2:]) <= 320
    boxes, density = det_.analyse(frame)
    assert boxes == [] and density.shape == (480, 640)
    assert density.sum() == pytest.approx(7.0, rel=1e-3)
