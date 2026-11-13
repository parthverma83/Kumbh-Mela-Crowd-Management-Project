"""
SQLite log written by crowd_monitor.py --log, and read by the dashboard. Standard library only.

Tables (ts is Unix time in seconds):
  readings(ts, people, peak, level, decision, result_age_s, detector)
  zone_readings(ts, zone, people, density)
  alerts(ts, kind, old_level, new_level, detail)
  crossings(ts, line, direction, track_id)        -- --flow only

Readings are written at most once per `interval` seconds. Alerts (level and decision changes,
stall start/end, and a "start" row per run) and crossings are written when they happen;
hysteresis, the decision cooldown, the stall timer and once-per-track counting already limit
how often that can be.
"""
import sqlite3
import time

LOG_INTERVAL = 1.0  # s; at most one reading per interval

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    ts REAL NOT NULL, people REAL, peak REAL, level TEXT, decision TEXT,
    result_age_s REAL, detector TEXT);
CREATE TABLE IF NOT EXISTS zone_readings (
    ts REAL NOT NULL, zone TEXT NOT NULL, people REAL, density REAL);
CREATE TABLE IF NOT EXISTS alerts (
    ts REAL NOT NULL, kind TEXT NOT NULL, old_level TEXT, new_level TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS crossings (
    ts REAL NOT NULL, line TEXT NOT NULL, direction TEXT NOT NULL, track_id INTEGER);
CREATE INDEX IF NOT EXISTS readings_ts ON readings(ts);
CREATE INDEX IF NOT EXISTS zone_readings_ts ON zone_readings(ts);
CREATE INDEX IF NOT EXISTS alerts_ts ON alerts(ts);
CREATE INDEX IF NOT EXISTS crossings_ts ON crossings(ts);
"""


def connect(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")  # readers (the dashboard) don't block the writer
    db.executescript(SCHEMA)
    return db


class CrowdLog:
    def __init__(self, path, detector, interval=LOG_INTERVAL, clock=time.time):
        self.db = connect(path)
        self.detector = detector
        self.interval = interval
        self.clock = clock
        self.last = None  # time of the last reading written

    def reading(self, stats, result_age):
        """Write a reading and its zones, unless one was written under `interval` s ago."""
        now = self.clock()
        if self.last is not None and now - self.last < self.interval:
            return False
        self.last = now
        with self.db:
            self.db.execute("INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (now, stats["people"], stats["peak"], stats["density"],
                             stats["decision"], result_age, self.detector))
            self.db.executemany("INSERT INTO zone_readings VALUES (?, ?, ?, ?)",
                                [(now, z["name"], z["people"], z["density"])
                                 for z in stats.get("zones", [])])
        return True

    def alert(self, kind, old, new, detail=""):
        with self.db:
            self.db.execute("INSERT INTO alerts VALUES (?, ?, ?, ?, ?)",
                            (self.clock(), kind, old, new, detail))

    def crossing(self, line, direction, track_id):
        with self.db:
            self.db.execute("INSERT INTO crossings VALUES (?, ?, ?, ?)",
                            (self.clock(), line, direction, track_id))

    def close(self):
        self.db.close()
