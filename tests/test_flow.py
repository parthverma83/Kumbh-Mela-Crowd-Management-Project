import json
import os
import sqlite3
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crowd_monitor
from crowd_monitor import (CountLine, Detection, FlowMonitor, Ground, LineCounter, StallDetector,
                           TrackSpeeds, Zones, head_point, lapjv_standin)

FPS = 25


def hline(y=100, x0=0, x1=200, name="L"):
    """A line drawn left to right: forward = moving down the image."""
    return CountLine(name, np.float64([x0, y]), np.float64([x1, y]), ("down", "up"))


def walk(counter, tracks, fps=FPS):
    """tracks: {id: [(x, y) per frame or None when missed]}. Feeds frame by frame; returns events."""
    events = []
    n = max(len(v) for v in tracks.values())
    for i in range(n):
        pts = {tid: p[i] for tid, p in tracks.items() if i < len(p) and p[i] is not None}
        events += counter.update(pts, i / fps)
    return events


def path(start, end, frames):
    return [tuple(v) for v in np.linspace(start, end, frames)]


# ---------------- line crossings ----------------

def test_each_direction_is_counted_separately():
    c = LineCounter([hline()])
    ev = walk(c, {1: path((50, 50), (50, 150), 20),       # down
                  2: path((80, 160), (80, 40), 20),       # up
                  3: path((120, 30), (120, 180), 10)})    # down
    assert sorted(ev) == [("L", "down", 1), ("L", "down", 3), ("L", "up", 2)]
    assert c.counts == {"L": {"down": 2, "up": 1}}


def test_a_track_counts_once_per_direction_however_often_it_jitters():
    c = LineCounter([hline()], band=0)
    jitter = [(50, 100 + (3 if i % 2 else -3)) for i in range(40)]  # straddles the line 40 times
    walk(c, {7: jitter})
    assert c.counts == {"L": {"down": 1, "up": 1}}


def test_jitter_inside_the_band_is_not_a_crossing_but_walking_through_is():
    c = LineCounter([hline()], band=15)
    walk(c, {7: [(50, 100 + (14 if i % 2 else -14)) for i in range(40)]})  # +-14 px: inside the band
    assert c.counts == {"L": {"down": 0, "up": 0}}
    walk(c, {8: [(50, 80), (50, 95), (50, 105), (50, 112), (50, 116)]})   # 20 px above -> 16 px below
    assert c.counts == {"L": {"down": 1, "up": 0}}


def test_the_band_is_measured_perpendicular_to_a_slanted_line():
    ln = CountLine("D", np.float64([0, 0]), np.float64([100, 100]))
    assert ln.side((65, 35), band=15) == -1 and ln.side((60, 40), band=15) == 0  # 21.2 px vs 14.1 px


def test_standing_on_the_line_and_stepping_back_is_not_a_crossing():
    c = LineCounter([hline()], band=0)
    walk(c, {1: [(50, 90), (50, 100), (50, 100), (50, 90)]})
    assert c.counts == {"L": {"down": 0, "up": 0}}
    walk(c, {1: [(50, 100), (50, 110)]})  # ...then on through it: counts, from the last side
    assert c.counts == {"L": {"down": 1, "up": 0}}


def test_passing_beside_the_line_ends_is_not_a_crossing():
    c = LineCounter([hline(x0=0, x1=200)])
    ev = walk(c, {1: path((250, 50), (250, 150), 20),     # beyond the right end
                  2: path((-30, 50), (-30, 150), 20)})    # beyond the left end
    assert ev == []


def test_a_fast_diagonal_move_that_clips_the_line_counts():
    c = LineCounter([hline(x0=0, x1=200)])
    assert walk(c, {1: [(190, 80), (230, 120)]}) == []                   # at y=100, x=210: past the end
    c2 = LineCounter([hline(x0=0, x1=200)])
    assert walk(c2, {1: [(170, 80), (210, 120)]}) == [("L", "down", 1)]  # at y=100, x=190: inside


def test_a_track_missed_for_a_few_frames_still_counts():
    c = LineCounter([hline()])
    p = path((50, 50), (50, 150), 20)
    p[8:13] = [None] * 5                     # lost right where it crosses
    assert walk(c, {1: p}) == [("L", "down", 1)]


def test_forgotten_tracks_free_their_state():
    c = LineCounter([hline()], forget=1.0)
    c.update({1: (50, 50)}, 0.0)
    c.update({2: (60, 50)}, 2.0)             # track 1 unseen for 2 s
    assert (1, "L") not in c.state and 1 not in c.seen
    assert c.update({1: (50, 150)}, 2.1) == []  # an id seen again after being forgotten starts over


def test_two_lines_count_independently():
    lines = [hline(y=100, name="A"), CountLine("B", np.float64([100, 0]), np.float64([100, 200]),
                                               ("left", "right"))]
    c = LineCounter(lines)
    walk(c, {1: path((50, 50), (150, 150), 20)})   # down through A, rightwards through B
    # B runs top to bottom, so its right side (forward) is the image's left: moving right is backward
    assert c.counts == {"A": {"down": 1, "up": 0}, "B": {"left": 0, "right": 1}}


# ---------------- speeds and stalls ----------------

def test_speed_is_displacement_over_the_window():
    s = TrackSpeeds(Ground((200, 200)), window=1.0)
    for i in range(51):
        out = s.update({1: (10 + 2 * i, 50), 2: (100 + (1 if i % 2 else -1), 50)}, i / FPS)
    assert out[1] == pytest.approx(50.0)      # 2 px/frame at 25 fps
    assert out[2] < 3                          # jittering in place: ~0, not 50 px/s of path


def test_no_speed_until_half_a_window_has_been_seen():
    s = TrackSpeeds(Ground((200, 200)), window=1.0)
    assert s.update({1: (0, 0)}, 0.0) == {}
    assert s.update({1: (10, 0)}, 0.4) == {}
    assert s.update({1: (20, 0)}, 0.5) == {1: pytest.approx(40.0)}


def test_calibrated_speed_is_in_metres_per_second():
    # 10 x 10 m seen top-down in 100 x 100 px: 10 px per metre
    g = Ground((100, 100), [[0, 99], [99, 99], [99, 0], [0, 0]], 10, 10)
    s = TrackSpeeds(g, window=1.0)
    for i in range(26):
        out = s.update({1: (10 + 0.4 * i, 50)}, i / FPS)  # 10 px/s
    assert out[1] == pytest.approx(10 * 10 / 99, rel=0.01)  # ~1 m/s


def test_stall_alert_after_the_hold_and_not_before():
    st = StallDetector(threshold=0.2, seconds=10, min_tracks=3)
    slow = [0.1, 0.05, 0.15, 0.3]              # median 0.125
    out = [st.update("Z", True, slow, t) for t in np.arange(0, 12.01, 0.5)]
    fired = [(t, a) for t, a in zip(np.arange(0, 12.01, 0.5), out) if a]
    assert len(fired) == 1 and fired[0][0] == 10.0
    assert fired[0][1][:3] == ("stall", "flowing", "stalled")
    end = st.update("Z", True, [1.0, 1.1, 0.9], 12.5)
    assert end[:3] == ("stall", "stalled", "flowing")


def test_no_stall_when_moving_sparse_or_too_few_tracks():
    for dense, speeds in [(True, [1.0, 1.2, 0.9]),      # moving
                          (False, [0.0, 0.0, 0.0]),     # not dense: people standing in a quiet zone
                          (True, [0.0, 0.0])]:          # only 2 tracks
        st = StallDetector(threshold=0.2, seconds=10, min_tracks=3)
        assert all(st.update("Z", dense, speeds, t) is None for t in range(30))


def test_a_brief_movement_restarts_the_stall_timer():
    st = StallDetector(threshold=0.2, seconds=10, min_tracks=3)
    slow, fast = [0.1] * 3, [1.0] * 3
    for t in range(9):
        assert st.update("Z", True, slow, t) is None
    assert st.update("Z", True, fast, 9) is None
    assert all(st.update("Z", True, slow, t) is None for t in range(10, 20))
    assert st.update("Z", True, slow, 20) is not None


def test_zones_stall_independently():
    st = StallDetector(threshold=0.2, seconds=2, min_tracks=1)
    alerts = [st.update(z, True, [0.0] if z == "A" else [1.0], t) for t in range(4) for z in "AB"]
    assert [a[3].split(":")[0] for a in alerts if a] == ["A"]


# ---------------- FlowMonitor on synthetic tracks ----------------

FLOW_ARGS = SimpleNamespace(low=5, mid=12, warn_density=2.0, crit_density=4.0, stall_speed=None,
                            stall_seconds=10.0, stall_min_tracks=3, line_band=15.0)


def det(tid, x, head_y, h=50):
    """A detection whose head point is (x, head_y)."""
    return Detection(int(x), int(head_y + h * crowd_monitor.HEAD_OFFSET), 20, h, 0.9, tid)


def test_head_point_matches_the_density_map():
    d = det(1, 40, 30)
    assert head_point(d) == (40.0, 30.0)
    m = crowd_monitor.density_from_detections([d], (100, 100), sigma=1)
    assert np.unravel_index(m.argmax(), m.shape) == (30, 40)


def flow_setup(calibrated=False, lines=()):
    if calibrated:
        g = Ground((200, 200), [[0, 199], [199, 199], [199, 0], [0, 0]], 20, 20)  # 10 px/m
    else:
        g = Ground((200, 200))
    zones = Zones(g, [("Left", [[0, 0], [99, 0], [99, 199], [0, 199]]),
                      ("Right", [[100, 0], [199, 0], [199, 199], [100, 199]])], lines=lines)
    return FlowMonitor(g, zones, FLOW_ARGS)


def test_flow_monitor_alerts_on_a_dense_standing_crowd_only():
    fm = flow_setup()
    rng = np.random.default_rng(0)
    standing = [(10 + 12 * (i % 7), 20 + 25 * (i // 7)) for i in range(8)]   # 8 people, Left
    alerts = []
    for f in range(int(12 * FPS)):
        t = f / FPS
        dets = [det(i, x + rng.normal(0, 1), y + rng.normal(0, 1)) for i, (x, y) in enumerate(standing)]
        dets += [det(100 + i, 120 + 2 * f % 70, 30 + 40 * i) for i in range(8)]  # 8 walking, Right
        zone_stats = [dict(name="Left", people=8.0, density=0.4), dict(name="Right", people=8.0, density=0.4)]
        _, a = fm.update(dets, t, zone_stats)
        alerts += [(t, x) for x in a]
    assert len(alerts) == 1
    t, (kind, old, new, detail) = alerts[0]
    assert (kind, old, new) == ("stall", "flowing", "stalled") and detail.startswith("Left:")
    assert 10.0 <= t < 10.0 + SPEED_WINDOW_PLUS
    assert zone_stats[0]["tracks"] == 8 and zone_stats[0]["speed"] < 15
    assert zone_stats[1]["speed"] > 15


SPEED_WINDOW_PLUS = crowd_monitor.SPEED_WINDOW + 0.1  # the stall timer starts once speeds exist


def test_flow_monitor_uses_metres_and_density_when_calibrated():
    fm = flow_setup(calibrated=True)
    assert fm.speed_unit == "m/s" and fm.stalls.threshold == crowd_monitor.STALL_SPEED
    assert fm.dense(dict(people=1, density=2.5)) and not fm.dense(dict(people=50, density=1.0))
    assert flow_setup().stalls.threshold == crowd_monitor.STALL_SPEED_PX
    assert not flow_setup().dense(dict(people=5, density=99)) and flow_setup().dense(dict(people=6, density=0))


def test_flow_monitor_counts_line_crossings_from_head_points():
    fm = flow_setup(lines=[hline(y=100, x0=0, x1=200)])
    crossings = []
    for f in range(30):
        zs = [dict(name="Left", people=1.0, density=0.1), dict(name="Right", people=0.0, density=0.0)]
        c, _ = fm.update([det(5, 50, 60 + 3 * f)], f / FPS, zs)
        crossings += c
    assert crossings == [("L", "down", 5)]


def test_detections_without_track_ids_are_ignored():
    fm = flow_setup(lines=[hline()])
    zs = [dict(name="Left", people=1.0, density=0.1), dict(name="Right", people=0.0, density=0.0)]
    assert fm.update([Detection(50, 50, 20, 50, 0.9)], 0.0, zs) == ([], [])
    assert zs[0]["tracks"] == 0 and zs[0]["speed"] is None


# ---------------- zones file lines ----------------

def write(tmp_path, cfg):
    p = tmp_path / "z.json"
    p.write_text(json.dumps(cfg))
    return str(p)


ZONE = {"name": "A", "polygon": [[0, 0], [99, 0], [99, 99], [0, 99]]}


def test_lines_load_scaled_with_default_directions(tmp_path):
    z = Zones.load(write(tmp_path, {"frame_size": [200, 200], "zones": [ZONE],
                                    "lines": [{"name": "Gate", "points": [[0, 100], [200, 100]]}]}),
                   Ground((100, 100)))
    (ln,) = z.lines
    assert ln.name == "Gate" and ln.directions == ("forward", "backward")
    assert ln.a.tolist() == [0, 50] and ln.b.tolist() == [100, 50]
    assert "lines: Gate (forward / backward)" in z.describe()


@pytest.mark.parametrize("line, message", [
    ({"points": [[0, 0], [9, 9]]}, "has no name"),
    ({"name": "G", "points": [[0, 0]]}, "needs 2 different [x, y] points"),
    ({"name": "G", "points": [[5, 5], [5, 5]]}, "needs 2 different [x, y] points"),
    ({"name": "G", "points": [[0, 0], [9, 9]], "directions": ["in", "in"]}, "2 different direction names"),
    ({"name": "G", "points": [[0, 0], [9, 9]], "directions": ["in"]}, "2 different direction names"),
])
def test_bad_lines_are_rejected(tmp_path, line, message):
    with pytest.raises(SystemExit, match=message.replace("[", r"\[").replace("]", r"\]")):
        Zones.load(write(tmp_path, {"zones": [ZONE], "lines": [line]}), Ground((100, 100)))


def test_duplicate_line_names_are_rejected(tmp_path):
    ln = {"name": "G", "points": [[0, 0], [9, 9]]}
    with pytest.raises(SystemExit, match="used twice"):
        Zones.load(write(tmp_path, {"zones": [ZONE], "lines": [ln, ln]}), Ground((100, 100)))


# ---------------- assignment stand-in for lap ----------------

def test_lapjv_standin_matches_an_optimal_assignment_with_a_cost_limit():
    cost = np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.9]])
    total, x, y = lapjv_standin(cost, extend_cost=True, cost_limit=0.8)
    assert x.tolist() == [0, 1] and y.tolist() == [0, 1, -1] and total == pytest.approx(0.3)
    _, x, y = lapjv_standin(np.array([[0.95]]), cost_limit=0.8)   # over the limit: unmatched
    assert x.tolist() == [-1] and y.tolist() == [-1]
    _, x, y = lapjv_standin(np.empty((0, 3)), cost_limit=0.8)
    assert x.tolist() == [] and y.tolist() == [-1, -1, -1]


# ---------------- main loop ----------------

class FakeTracker:
    """One person walking down through the line, then 4 people standing still in the zone."""

    def __init__(self):
        self.f = 0

    def analyse(self, frame):
        f, self.f = self.f, self.f + 1
        dets = [det(1, 32, 4 + f)] if f < 40 else [det(10 + i, 10 + 10 * i, 30, h=10) for i in range(4)]
        return dets, crowd_monitor.density_from_detections(dets, frame.shape, sigma=2)


def test_main_logs_crossings_and_a_stall(monkeypatch, tmp_path, capsys):
    video = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for _ in range(40 + 4 * FPS):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    zones = write(tmp_path, {"zones": [{"name": "Z", "polygon": [[0, 0], [63, 0], [63, 47], [0, 47]]}],
                             "lines": [{"name": "Mid", "points": [[0, 24], [63, 24]],
                                        "directions": ["in", "out"]}]})
    monkeypatch.setattr(crowd_monitor, "build_detector", lambda args: FakeTracker())
    db = tmp_path / "run.db"
    crowd_monitor.main(["--source", str(video), "--headless", "--heatmap-out", "", "--log", str(db),
                        "--flow", "--zones", zones, "--stall-seconds", "1", "--low", "3"])
    con = sqlite3.connect(db)
    assert con.execute("SELECT line, direction, track_id FROM crossings").fetchall() == [("Mid", "in", 1)]
    stalls = con.execute("SELECT old_level, new_level FROM alerts WHERE kind='stall'").fetchall()
    assert stalls == [("flowing", "stalled")]
    out = capsys.readouterr().out
    assert "CROSS Mid: in (track 1)" in out and "Crossings Mid: in 1, out 0" in out


def test_flow_needs_zones_and_yolo(monkeypatch):
    with pytest.raises(SystemExit, match="--flow needs --zones"):
        crowd_monitor.main(["--source", "x.mp4", "--flow"])
    args = crowd_monitor.parse_args(["--flow", "--detector", "dmcount"])
    with pytest.raises(SystemExit, match="--flow needs --detector yolo"):
        crowd_monitor.build_detector(args)
