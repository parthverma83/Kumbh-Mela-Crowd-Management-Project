import os
import sqlite3
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crowd_monitor
from crowd_log import CrowdLog
from crowd_monitor import CrowdAnalyser, Ground, LevelTracker, change_alerts, classify


def tracker(margin=0.10, hold=5.0, low=2.0, mid=4.0):
    t = [0.0]
    return LevelTracker(low, mid, margin, hold, clock=lambda: t[0]), t


def run(tr, t, values, step=0.5):
    """Feed values at `step` s apart; return the level after each."""
    out = []
    for v in values:
        out.append(tr.update(v))
        t[0] += step
    return out


# ---------------- hysteresis ----------------

def test_no_flicker_while_hovering_around_a_threshold():
    hover = [2.05, 1.95, 2.1, 1.9, 2.0, 1.85, 2.02, 1.95] * 10  # 40 s around 2.0/m^2
    tr, t = tracker()
    levels = run(tr, t, hover)
    assert levels[0] == "MEDIUM" and set(levels) == {"MEDIUM"}  # one rise, no fall
    # without hysteresis the same signal changes level dozens of times
    plain = [classify(v, 2.0, 4.0) for v in hover]
    assert sum(a != b for a, b in zip(plain, plain[1:])) > 30


def test_rises_as_soon_as_the_threshold_is_passed():
    tr, t = tracker()
    assert tr.update(1.0) == "LOW"
    assert tr.update(2.0) == "LOW"       # classify's rule: LOW up to and including the threshold
    assert tr.update(2.01) == "MEDIUM"
    t[0] += 0.1                           # rising ignores the hold time
    assert tr.update(4.5) == "HIGH"


def test_falls_only_below_the_margin_and_after_the_hold():
    tr, t = tracker()
    tr.update(3.0)                        # MEDIUM at t=0
    t[0] = 3
    assert tr.update(1.0) == "MEDIUM"     # well below, but held only 3 s
    t[0] = 6
    assert tr.update(1.85) == "MEDIUM"    # held long enough, but 1.85 > 2.0 - 10%
    assert tr.update(1.8) == "LOW"        # at threshold - margin, after 6 s: falls


def test_falls_one_or_two_levels_depending_on_the_value():
    tr, t = tracker()
    tr.update(5.0)
    t[0] = 10
    assert tr.update(3.0) == "MEDIUM"     # below 4.0 - 10%, above 1.8
    tr2, t2 = tracker()
    tr2.update(5.0)
    t2[0] = 10
    assert tr2.update(0.5) == "LOW"       # straight from HIGH to LOW


def test_a_rise_restarts_the_hold_time():
    tr, t = tracker()
    tr.update(3.0)                        # MEDIUM at 0
    t[0] = 4
    tr.update(4.5)                        # HIGH at 4
    t[0] = 6
    assert tr.update(0.5) == "HIGH"       # MEDIUM was held long, but HIGH only 2 s
    t[0] = 9
    assert tr.update(0.5) == "LOW"


def test_uncalibrated_counts_use_the_same_rule():
    tr, t = tracker(low=5, mid=12)
    tr.update(13)                         # HIGH
    t[0] = 10
    assert tr.update(11) == "HIGH"        # 11 > 12 - 1.2
    assert tr.update(10.8) == "MEDIUM"
    t[0] = 20
    assert tr.update(4.6) == "MEDIUM"     # 4.6 > 5 - 0.5
    assert tr.update(4.5) == "LOW"


def test_zero_margin_and_hold_is_plain_classify():
    values = np.random.default_rng(0).uniform(0, 6, 300)
    tr, t = tracker(margin=0, hold=0)
    assert run(tr, t, values) == [classify(v, 2.0, 4.0) for v in values]


ARGS = SimpleNamespace(low=5, mid=12, warn_density=2.0, crit_density=4.0, hysteresis=0.1,
                       min_hold=5.0)


def test_analyser_holds_its_level_through_a_short_dip():
    g = Ground((100, 100), [[0, 99], [99, 99], [99, 0], [0, 0]], 10, 10)  # 10 x 10 m, top-down
    t = [0.0]
    an = CrowdAnalyser(g, ARGS, clock=lambda: t[0])
    crowded = g.pixel_area * 3.0          # 3/m^2 everywhere: MEDIUM
    assert an.update(crowded)["density"] == "MEDIUM"
    t[0] = 1
    an.heat = None                         # drop the heatmap memory to test the level alone
    assert an.update(g.pixel_area * 1.0)["density"] == "MEDIUM"
    t[0] = 6
    an.heat = None
    assert an.update(g.pixel_area * 1.0)["density"] == "LOW"


# ---------------- alert rows ----------------

def stats(level, decision="HOLD / BALANCED", people=10.0, peak=1.0, zones=()):
    return dict(density=level, decision=decision, people=people, peak=peak, zones=list(zones))


def test_first_stats_record_the_starting_level():
    assert change_alerts(None, stats("LOW", people=3), calibrated=False) == [
        ("level", None, "LOW", "people 3.0")]


def test_level_and_decision_changes_become_alerts():
    zones = [dict(name="Ghat A", density=1.5), dict(name="Ghat B", density=3.25)]
    alerts = change_alerts(stats("LOW"), stats("HIGH", "USE Gate 1", peak=4.2, zones=zones),
                           calibrated=True)
    assert alerts == [("level", "LOW", "HIGH", "peak 4.20/m2"),
                      ("decision", "HOLD / BALANCED", "USE Gate 1", "Ghat A 1.50, Ghat B 3.25")]
    assert change_alerts(stats("LOW"), stats("LOW"), calibrated=True) == []


# ---------------- SQLite log ----------------

def make_log(path, interval=1.0):
    t = [1000.0]
    return CrowdLog(str(path), "dmcount/torch", interval, clock=lambda: t[0]), t


def test_log_writes_at_most_one_reading_per_interval(tmp_path):
    log, t = make_log(tmp_path / "c.db")
    s = stats("MEDIUM", zones=[dict(name="A", people=4.0, density=0.5)])
    written = []
    for dt in (0, 0.4, 0.99, 1.0, 1.5, 2.2):
        t[0] = 1000 + dt
        written.append(log.reading(s, 1.25))
    assert written == [True, False, False, True, False, True]
    rows = log.db.execute("SELECT ts, people, level, result_age_s, detector FROM readings").fetchall()
    assert rows == [(1000.0, 10.0, "MEDIUM", 1.25, "dmcount/torch"),
                    (1001.0, 10.0, "MEDIUM", 1.25, "dmcount/torch"),
                    (1002.2, 10.0, "MEDIUM", 1.25, "dmcount/torch")]
    assert log.db.execute("SELECT COUNT(*), MIN(zone) FROM zone_readings").fetchone() == (3, "A")


def test_log_schema_matches_the_spec(tmp_path):
    log, _ = make_log(tmp_path / "c.db")
    cols = {t: [r[1] for r in log.db.execute(f"PRAGMA table_info({t})")]
            for t in ("readings", "zone_readings", "alerts")}
    assert cols == {"readings": ["ts", "people", "peak", "level", "decision", "result_age_s", "detector"],
                    "zone_readings": ["ts", "zone", "people", "density"],
                    "alerts": ["ts", "kind", "old_level", "new_level", "detail"]}
    assert log.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_alerts_are_not_throttled_and_runs_append(tmp_path):
    path = tmp_path / "c.db"
    log, t = make_log(path)
    log.alert("start", None, None, "run 1")
    log.alert("level", None, "LOW", "people 3.0")  # same instant: still written
    log.close()
    log, t = make_log(path)                          # a second run appends to the same file
    log.alert("start", None, None, "run 2")
    log.close()
    db = sqlite3.connect(path)
    assert [r[0] for r in db.execute("SELECT detail FROM alerts ORDER BY rowid")] == [
        "run 1", "people 3.0", "run 2"]


# ---------------- main loop ----------------

class Steps:
    """Density whose total follows `counts`, one value per frame."""

    def __init__(self, counts):
        self.counts = iter(counts)

    def analyse(self, frame):
        d = np.zeros(frame.shape[:2], np.float32)
        d[10, 10] = next(self.counts)
        return [], d


def test_main_logs_start_level_changes_and_readings(monkeypatch, tmp_path, capsys):
    video = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for _ in range(6):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    monkeypatch.setattr(crowd_monitor, "build_detector", lambda args: Steps([1, 1, 200, 200, 200, 200]))
    db = tmp_path / "run.db"
    crowd_monitor.main(["--source", str(video), "--headless", "--heatmap-out", "", "--log", str(db)])
    con = sqlite3.connect(db)
    alerts = con.execute("SELECT kind, old_level, new_level FROM alerts ORDER BY rowid").fetchall()
    assert alerts[0] == ("start", None, None)
    assert alerts[1:] == [("level", None, "LOW"), ("level", "LOW", "HIGH")]
    # 6 frames in well under a second: one reading
    assert con.execute("SELECT COUNT(*), MIN(detector) FROM readings").fetchone() == (1, "yolo")
    assert "ALERT level: LOW -> HIGH" in capsys.readouterr().out
