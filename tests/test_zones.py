import json
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crowd_monitor
from crowd_monitor import (DENSITY_MARGIN, CrowdAnalyser, ExitChooser, Ground, Zones, draw_overlay,
                           exit_margin, redirect_decision, REDIRECT_FRACTION, REDIRECT_MARGIN)

ARGS = SimpleNamespace(low=5, mid=12, warn_density=2.0, crit_density=4.0, hysteresis=0.1,
                       min_hold=5.0)
SHAPE = (480, 640)
TRAPEZOID = [[40, 380], [600, 380], [460, 120], [180, 120]]  # 10 x 6 m, as in test_crowd_monitor
LEFT_BOX = [[0, 0], [319, 0], [319, 479], [0, 479]]
RIGHT_BOX = [[320, 0], [639, 0], [639, 479], [320, 479]]


def write_zones(tmp_path, zones, exits=(), frame_size=None):
    cfg = {"zones": [{"name": n, "polygon": p} for n, p in zones],
           "exits": [{"name": e, "zone": z} for e, z in exits]}
    if frame_size:
        cfg["frame_size"] = frame_size
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(cfg))
    return path


# ---------------- loading ----------------

def test_load_scales_points_from_frame_size(tmp_path):
    small = [[10, 20], [150, 20], [150, 200], [10, 200]]  # drawn on a 320x240 frame
    zones = Zones.load(write_zones(tmp_path, [("A", small)], frame_size=[320, 240]), Ground(SHAPE))
    assert zones.zones[0].polygon.tolist() == [[20, 40], [300, 40], [300, 400], [20, 400]]


def test_load_scales_each_axis_separately(tmp_path):
    zones = Zones.load(write_zones(tmp_path, [("A", [[0, 0], [100, 0], [100, 100]])],
                                   frame_size=[1280, 240]), Ground(SHAPE))
    assert zones.zones[0].polygon.tolist() == [[0, 0], [50, 0], [50, 200]]


def test_load_without_frame_size_uses_points_as_given(tmp_path):
    zones = Zones.load(write_zones(tmp_path, [("A", LEFT_BOX)]), Ground(SHAPE))
    assert zones.zones[0].polygon.tolist() == LEFT_BOX


def test_load_keeps_exits_in_file_order(tmp_path):
    path = write_zones(tmp_path, [("A", LEFT_BOX), ("B", RIGHT_BOX)],
                       [("Gate 2", "B"), ("Gate 1", "A")])
    assert Zones.load(path, Ground(SHAPE)).exits == [("Gate 2", "B"), ("Gate 1", "A")]


@pytest.mark.parametrize("zones,exits,message", [
    ([], [], "no zones"),
    ([("A", LEFT_BOX), ("A", RIGHT_BOX)], [], "used twice"),
    ([("A", [[0, 0], [10, 10]])], [], "at least 3"),
    ([("", LEFT_BOX)], [], "has no name"),
    ([("A", LEFT_BOX)], [("Gate 1", "Nowhere")], "unknown zone 'Nowhere'"),
    ([("A", LEFT_BOX)], [("Gate 1", "A"), ("Gate 1", "A")], "exit name 'Gate 1' is used twice"),
])
def test_load_rejects_bad_files(tmp_path, zones, exits, message):
    with pytest.raises(SystemExit, match=message):
        Zones.load(write_zones(tmp_path, zones, exits), Ground(SHAPE))


def test_example_zones_file_loads():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "zones_example.json")
    zones = Zones.load(path, Ground((720, 1280)))
    assert [z.name for z in zones.zones] == ["Ghat A", "Ghat B", "Ghat C"]
    assert [e for e, _ in zones.exits] == ["Gate 1", "Gate 2", "Gate 3"]


# ---------------- masks ----------------

def test_uncalibrated_masks_cover_their_polygons_in_blocks():
    zones = Zones(Ground(SHAPE), [("A", LEFT_BOX), ("B", RIGHT_BOX)])
    a, b = zones.zones
    assert a.mask.sum() == 320 * 480 and b.mask.sum() == 320 * 480
    assert (a.mask * b.mask).sum() == 0           # the halves don't overlap
    assert a.area == pytest.approx(320 * 480 / 100 ** 2)  # 100x100 px blocks


def test_calibrated_zone_area_is_in_square_metres():
    g = Ground(SHAPE, TRAPEZOID, 10, 6)
    zones = Zones(g, [("All", TRAPEZOID)])
    assert zones.zones[0].area == pytest.approx(60, rel=0.03)


def test_calibrated_zone_is_clipped_to_the_calibrated_area(capsys):
    g = Ground(SHAPE, TRAPEZOID, 10, 6)
    zones = Zones(g, [("Whole frame", [[0, 0], [639, 0], [639, 479], [0, 479]])])
    assert zones.zones[0].area == pytest.approx(60, rel=0.03)  # nothing outside counts
    assert (zones.zones[0].mask * (1 - g.roi)).sum() == 0
    assert "inside the calibrated area" in capsys.readouterr().out  # and the user is told


def test_zone_outside_the_calibrated_area_is_rejected():
    g = Ground(SHAPE, TRAPEZOID, 10, 6)
    with pytest.raises(SystemExit, match="no area inside the calibrated area"):
        Zones(g, [("Sky", [[0, 0], [100, 0], [100, 50], [0, 50]])])


def test_measure_reads_people_and_density_per_zone():
    g = Ground(SHAPE)
    zones = Zones(g, [("A", LEFT_BOX), ("B", RIGHT_BOX)])
    heat = np.zeros(SHAPE, np.float32)
    heat[:, :320] = g.pixel_area[:, :320] * 3.0  # 3 people per block on the left only
    a, b = zones.measure(heat, heat)
    assert a["people"] == pytest.approx(3.0 * 320 * 480 / 100 ** 2, rel=1e-4)
    assert a["density"] == pytest.approx(3.0, rel=1e-4) and b["density"] == 0


# ---------------- choosing an exit ----------------

def zone(name, density, area=10.0):
    return dict(name=name, density=density, area=area, people=density * area, peak=density)


EXITS = [("Gate 1", "A"), ("Gate 2", "B"), ("Gate 3", "C")]


def chooser(calibrated=True):
    t = [0.0]
    return ExitChooser(EXITS, calibrated, cooldown=2.0, clock=lambda: t[0]), t


def test_starts_on_the_least_dense_exit():
    c, _ = chooser()
    assert c.update([zone("A", 3.0), zone("B", 1.0), zone("C", 2.0)]) == "Gate 2"


def test_ties_go_to_the_exit_listed_first():
    c, _ = chooser()
    assert c.update([zone("A", 1.0), zone("B", 1.0), zone("C", 1.0)]) == "Gate 1"


def test_does_not_switch_for_a_difference_within_the_margin():
    c, t = chooser()
    c.update([zone("A", 1.0), zone("B", 2.0), zone("C", 2.0)])
    t[0] = 10
    # B is now lower, but only by 0.4/m^2, under the 0.5 margin
    assert c.update([zone("A", 1.4), zone("B", 1.0), zone("C", 2.0)]) == "Gate 1"


def test_switches_when_clearly_better_after_the_cooldown():
    c, t = chooser()
    c.update([zone("A", 1.0), zone("B", 2.0), zone("C", 2.0)])
    t[0] = 1.0  # too soon
    assert c.update([zone("A", 3.0), zone("B", 1.0), zone("C", 2.0)]) == "Gate 1"
    t[0] = 2.0
    assert c.update([zone("A", 3.0), zone("B", 1.0), zone("C", 2.0)]) == "Gate 2"
    t[0] = 3.0  # cooldown restarts at each switch
    assert c.update([zone("A", 3.0), zone("B", 3.0), zone("C", 0.0)]) == "Gate 2"
    t[0] = 4.0
    assert c.update([zone("A", 3.0), zone("B", 3.0), zone("C", 0.0)]) == "Gate 3"


def test_switches_to_the_least_dense_of_the_clearly_better_exits():
    c, t = chooser()
    c.update([zone("A", 1.0), zone("B", 2.0), zone("C", 2.0)])
    t[0] = 5
    assert c.update([zone("A", 4.0), zone("B", 2.0), zone("C", 1.0)]) == "Gate 3"


def test_two_exits_on_one_zone_never_flip_between_themselves():
    t = [0.0]
    c = ExitChooser([("North", "A"), ("South", "A")], True, cooldown=0, clock=lambda: t[0])
    assert c.update([zone("A", 5.0)]) == "North"
    t[0] = 10
    assert c.update([zone("A", 1.0)]) == "North"


def test_calibrated_margin_is_the_density_margin():
    assert exit_margin(zone("A", 3), zone("B", 1), calibrated=True) == DENSITY_MARGIN


@pytest.mark.parametrize("left,right", [(160, 140), (6, 1), (3, 1), (1, 6), (400, 300), (100, 100)])
def test_uncalibrated_margin_is_the_left_right_rule_for_equal_zones(left, right):
    area = 15.36  # one half of a 640x480 frame, in blocks
    a, b = zone("L", left / area, area), zone("R", right / area, area)
    margin = max(REDIRECT_MARGIN, REDIRECT_FRACTION * (left + right))
    left_is_worse = redirect_decision(left, right, margin) == crowd_monitor.TO_RIGHT
    assert (a["density"] - b["density"] > exit_margin(a, b, calibrated=False)) == left_is_worse


def test_uncalibrated_margin_compares_density_not_people():
    # the big zone holds more people, but the small one is far more crowded
    big, small = zone("Big", 1.0, area=100), zone("Small", 5.0, area=10)
    c = ExitChooser([("To big", "Big"), ("To small", "Small")], False, clock=lambda: 0)
    assert c.update([big, small]) == "To big"


# ---------------- analyser and drawing ----------------

def test_analyser_recommends_the_emptier_zone_by_name():
    g = Ground(SHAPE)
    zones = Zones(g, [("A", LEFT_BOX), ("B", RIGHT_BOX)], [("Gate 1", "A"), ("Gate 2", "B")])
    t = [100.0]
    an = CrowdAnalyser(g, ARGS, clock=lambda: t[0], zones=zones)
    left = np.zeros(SHAPE, np.float32)
    left[:, :320] = 1e-3
    stats = an.update(left)
    assert stats["exit"] == "Gate 2" and stats["decision"] == "USE Gate 2"
    assert [z["name"] for z in stats["zones"]] == ["A", "B"]
    assert stats["zones"][0]["people"] == pytest.approx(320 * 480 * 1e-3, rel=1e-3)


def test_zones_without_exits_keep_the_left_right_decision():
    g = Ground(SHAPE)
    an = CrowdAnalyser(g, ARGS, clock=lambda: 100.0, zones=Zones(g, [("A", LEFT_BOX)]))
    stats = an.update(np.zeros(SHAPE, np.float32))
    assert "zones" in stats and "exit" not in stats and stats["decision"] == crowd_monitor.HOLD


def test_without_zones_stats_are_unchanged():
    an = CrowdAnalyser(Ground(SHAPE), ARGS, clock=lambda: 100.0)
    assert set(an.update(np.zeros(SHAPE, np.float32))) == {
        "people", "left", "right", "left_d", "right_d", "peak", "unit", "density", "avg_density",
        "decision"}


def test_overlay_draws_zones_and_highlights_the_chosen_one():
    g = Ground(SHAPE)
    zones = Zones(g, [("A", [[20, 20], [300, 20], [300, 460], [20, 460]]), ("B", RIGHT_BOX)],
                  [("Gate 1", "A"), ("Gate 2", "B")])
    an = CrowdAnalyser(g, ARGS, clock=lambda: 100.0, zones=zones)
    stats = an.update(np.zeros(SHAPE, np.float32))
    black = np.zeros((*SHAPE, 3), np.uint8)
    out = draw_overlay(black.copy(), black, [], g, stats, 25.0, 1.0, zones)
    assert tuple(out[200, 20]) == (0, 255, 0)      # chosen zone's outline is green
    assert tuple(out[200, 320]) == (255, 255, 255)  # the other is white
    plain = draw_overlay(black.copy(), black, [], g, an.update(np.zeros(SHAPE, np.float32)), 25.0)
    assert not np.array_equal(out, plain)


class Instant:
    def analyse(self, frame):
        d = np.zeros(frame.shape[:2], np.float32)
        d[:, : frame.shape[1] // 2] = 1e-2  # left half crowded
        return [], d


def test_main_logs_zone_stats_and_the_exit(monkeypatch, tmp_path, capsys):
    video = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for _ in range(3):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    path = write_zones(tmp_path, [("West", [[0, 0], [31, 0], [31, 47], [0, 47]]),
                                  ("East", [[32, 0], [63, 0], [63, 47], [32, 47]])],
                       [("Gate W", "West"), ("Gate E", "East")], frame_size=[64, 48])
    monkeypatch.setattr(crowd_monitor, "build_detector", lambda args: Instant())
    crowd_monitor.main(["--source", str(video), "--headless", "--heatmap-out", "",
                        "--zones", str(path)])
    out = capsys.readouterr().out
    assert "Zones: West" in out and "Gate W -> West" in out
    assert "zones[West=" in out and "decision=USE Gate E" in out
