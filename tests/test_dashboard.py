"""Tests for dashboard.py's query functions, on logs written by crowd_log.CrowdLog."""
import math
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard as dash  # noqa: E402
from crowd_log import CrowdLog  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def stats(people, level, decision="HOLD / BALANCED", zones=(("A", 1.0, 0.1), ("B", 2.0, 0.2))):
    return {"people": people, "peak": people / 10, "density": level, "decision": decision,
            "zones": [{"name": n, "people": p, "density": d} for n, p, d in zones]}


def write_two_runs(path):
    """Run 1: start at 100, readings at 101-103 (LOW, LOW, MEDIUM), 2 crossings, a level alert.
    Run 2: start at 200, one reading at 201."""
    clock = Clock(100.0)
    log = CrowdLog(path, "yolo", interval=1.0, clock=clock)
    log.alert("start", None, None, "source=walk.mp4 detector=yolo calib=- zones=z.json")
    for t, people, level in [(101, 3.0, "LOW"), (102, 4.0, "LOW"), (103, 9.0, "MEDIUM")]:
        clock.t = t
        if t == 103:
            log.alert("level", "LOW", "MEDIUM", "people 9.0")
        log.reading(stats(people, level, zones=(("A", people / 3, 0.1), ("B", people / 2, 0.2))),
                    0.05)
    clock.t = 102.5
    log.crossing("Gate", "in", 7)
    clock.t = 102.7
    log.crossing("Gate", "in", 8)
    clock.t = 102.9
    log.crossing("Gate", "out", 7)
    clock.t = 200.0
    log.alert("start", None, None, "source=0 detector=dmcount calib=cal.json")
    clock.t = 201.0
    log.reading(stats(20.0, "HIGH", "USE Gate 2"), 0.2)
    log.close()


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "run.db"
    write_two_runs(path)
    con = dash.open_log(path)
    yield con
    con.close()


def test_open_log_is_read_only_and_never_creates(tmp_path):
    missing = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        dash.open_log(missing)
    assert not missing.exists()
    path = tmp_path / "run.db"
    write_two_runs(path)
    con = dash.open_log(path)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM readings")
    con.close()


def test_runs_split_at_start_alerts(db):
    r = dash.runs(db)
    assert list(r.run) == [1, 2]
    assert list(r.start) == [100.0, 200.0]
    assert r.lo[0] == -math.inf and r.hi[0] == 200.0
    assert r.lo[1] == 200.0 and r.hi[1] == math.inf
    assert dash.run_info(r.detail[0])["source"] == "walk.mp4"


def test_readings_and_zones_stay_inside_their_run(db):
    r1 = dash.runs(db).iloc[0]
    assert list(dash.readings(db, r1.lo, r1.hi).people) == [3.0, 4.0, 9.0]
    z = dash.zone_readings(db, r1.lo, r1.hi)
    assert len(z) == 6 and set(z.zone) == {"A", "B"}
    r2 = dash.runs(db).iloc[1]
    assert list(dash.readings(db, r2.lo, r2.hi).level) == ["HIGH"]


def test_since_trims_the_window(db):
    r1 = dash.runs(db).iloc[0]
    assert list(dash.readings(db, r1.lo, r1.hi, since=102.0).ts) == [102.0, 103.0]
    assert len(dash.zone_readings(db, r1.lo, r1.hi, since=103.0)) == 2
    assert dash.window_start(103.0, 60) == 43.0
    assert dash.window_start(103.0, None) is None
    assert dash.window_start(None, 60) is None


def test_latest_reading_and_its_zones(db):
    r1, r2 = (row for _, row in dash.runs(db).iterrows())
    now, zones = dash.latest(db, r1.lo, r1.hi)
    assert now["ts"] == 103.0 and now["level"] == "MEDIUM" and now["people"] == 9.0
    assert list(zones.zone) == ["A", "B"]
    assert list(zones.people) == [3.0, 4.5]
    now, _ = dash.latest(db, r2.lo, r2.hi)
    assert now["decision"] == "USE Gate 2" and now["detector"] == "yolo"


def test_latest_of_a_run_without_readings(tmp_path):
    path = tmp_path / "empty.db"
    log = CrowdLog(path, "yolo", clock=Clock(5.0))
    log.alert("start", None, None, "source=0")
    log.close()
    con = dash.open_log(path)
    r = dash.runs(con).iloc[0]
    now, zones = dash.latest(con, r.lo, r.hi)
    assert now is None and zones.empty
    assert dash.readings(con, r.lo, r.hi).empty
    assert dash.crossing_counts(con, r.lo, r.hi).empty
    con.close()


def test_alert_history_newest_first(db):
    r1 = dash.runs(db).iloc[0]
    a = dash.alerts(db, r1.lo, r1.hi)
    assert list(a.kind) == ["level", "start"]
    assert a.new_level[0] == "MEDIUM"


def test_crossing_counts_per_line_and_direction(db):
    r1, r2 = (row for _, row in dash.runs(db).iterrows())
    c = dash.crossing_counts(db, r1.lo, r1.hi)
    assert c.to_dict("records") == [{"line": "Gate", "direction": "in", "count": 2},
                                    {"line": "Gate", "direction": "out", "count": 1}]
    assert dash.crossing_counts(db, r2.lo, r2.hi).empty


def test_log_without_start_rows_or_crossings_table(tmp_path):
    """Logs written before task 5 have no crossings table; hand-made logs may lack start rows."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE readings (ts REAL NOT NULL, people REAL, peak REAL, level TEXT,
                               decision TEXT, result_age_s REAL, detector TEXT);
        CREATE TABLE alerts (ts REAL NOT NULL, kind TEXT NOT NULL, old_level TEXT,
                             new_level TEXT, detail TEXT);
        INSERT INTO readings VALUES (50, 1, 0.1, 'LOW', 'HOLD', 0.1, 'yolo');
        INSERT INTO readings VALUES (51, 2, 0.2, 'LOW', 'HOLD', 0.1, 'yolo');""")
    con.commit()
    con.close()
    con = dash.open_log(path)
    r = dash.runs(con)
    assert list(r.run) == [1] and r.start[0] == 50.0
    run = r.iloc[0]
    assert len(dash.readings(con, run.lo, run.hi)) == 2
    assert dash.crossings(con, run.lo, run.hi).empty
    assert dash.zone_readings(con, run.lo, run.hi).empty
    now, zones = dash.latest(con, run.lo, run.hi)
    assert now["people"] == 2.0 and zones.empty
    con.close()


def test_level_spans_merge_consecutive_levels():
    r = pd.DataFrame({"ts": [1.0, 2.0, 3.0, 4.0, 5.0],
                      "level": ["LOW", "LOW", "HIGH", "HIGH", "LOW"]})
    s = dash.level_spans(r)
    assert s.to_dict("records") == [
        {"level": "LOW", "start": 1.0, "end": 3.0, "readings": 2},
        {"level": "HIGH", "start": 3.0, "end": 5.0, "readings": 2},
        {"level": "LOW", "start": 5.0, "end": 6.0, "readings": 1}]  # + median interval
    assert dash.level_spans(r.iloc[:0]).empty
    one = dash.level_spans(r.iloc[:1])
    assert one.start[0] == one.end[0] == 1.0


def test_density_unit_from_the_start_row():
    assert dash.density_unit("source=0 calib=-") == "people per 100×100 px"
    assert dash.density_unit("source=0 calib=cal.json") == "people/m²"
    assert dash.density_unit("") == "people per 100×100 px"


def test_zone_colours_follow_the_name_not_the_order():
    a = dash.zone_colours(["B", "A", "B"])
    b = dash.zone_colours(["A", "B"])
    assert a == b and a["A"] == dash.SERIES_COLOURS[0]
    many = dash.zone_colours([f"z{i:02d}" for i in range(10)])
    assert many["z09"] == "#8a8984"


def test_pick_run_latest_and_specific(db):
    r = dash.runs(db)
    assert dash.pick_run(r, 0).run == 2
    assert dash.pick_run(r, 1).run == 1
    assert dash.pick_run(r, 99).run == 2   # a run that no longer exists falls back to latest


def test_dashboard_never_imports_opencv_or_the_monitor():
    code = ("import sys, dashboard; "
            "print('cv2' in sys.modules, 'crowd_monitor' in sys.modules, 'ultralytics' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["False", "False", "False"]


def run_app(db_path, monkeypatch, refresh=0):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("CROWD_LOG_DB", str(db_path))
    monkeypatch.setenv("CROWD_LOG_REFRESH", str(refresh))
    at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=60)
    at.run()
    return at


def test_app_renders_the_latest_run(tmp_path, monkeypatch):
    path = tmp_path / "run.db"
    write_two_runs(path)
    at = run_app(path, monkeypatch, refresh=2)
    assert not at.exception
    assert [m.value for m in at.metric][:1] == ["20.0"]
    assert any("HIGH" in m.value for m in at.markdown)
    assert at.sidebar.selectbox[1].options[0].startswith("Latest run")
    at.sidebar.selectbox[1].select_index(2).run()   # run 1, the older one
    assert not at.exception
    assert [m.value for m in at.metric][:1] == ["9.0"]
    assert any("Line crossings" in s.value for s in at.subheader)


def test_save_upload_writes_a_readable_log_once(tmp_path):
    src = tmp_path / "run.db"
    write_two_runs(src)
    data = src.read_bytes()
    a = dash.save_upload(data, tmp_path / "up")
    b = dash.save_upload(data, tmp_path / "up")
    assert a == b and len(list((tmp_path / "up").iterdir())) == 1
    con = dash.open_log(a)
    assert list(dash.runs(con).run) == [1, 2]
    con.close()


def test_sample_logs_are_readable_without_a_wal_file():
    """The hosted demo reads these; rollback-journal mode means no -wal/-shm file is needed."""
    for path in dash.SAMPLE_LOGS.values():
        assert path.is_file(), path
        con = dash.open_log(path)
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert not dash.runs(con).empty
        con.close()


def test_hosted_entry_point_demo_mode(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv("CROWD_LOG_DB", raising=False)
    monkeypatch.setenv("CROWD_DASHBOARD_DEMO", "1")
    monkeypatch.setenv("CROWD_LOG_REFRESH", "0")
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert "not live or Kumbh data" in at.info[0].value
    assert not at.sidebar.text_input    # no server paths in the hosted demo
    sample = at.sidebar.selectbox[0]
    for i in range(len(sample.options)):
        at.sidebar.selectbox[0].select_index(i).run()
        assert not at.exception and at.metric, sample.options[i]
    at.sidebar.radio[0].set_value("Upload a log").run()
    assert not at.exception
    assert any("Upload a log" in c.value for c in at.caption)


def test_app_reports_a_missing_log(tmp_path, monkeypatch):
    at = run_app(tmp_path / "nope.db", monkeypatch)
    assert not at.exception
    assert "Can't open the log" in at.error[0].value
    assert not (tmp_path / "nope.db").exists()
