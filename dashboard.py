"""
Streamlit dashboard for the crowd monitor's SQLite log (see crowd_log.py). It only reads the log,
opened read-only; it never opens a camera or a video, so it can run on another machine or while
crowd_monitor.py is writing.

    python crowd_monitor.py --source 0 --zones zones.json --log run.db     # writer
    streamlit run dashboard.py -- --db run.db                             # reader

The database can also be set with the CROWD_LOG_DB environment variable or in the sidebar, and
the refresh period (default 2 s, 0 = off) with --refresh or CROWD_LOG_REFRESH. If the log doesn't
exist yet (the dashboard was started first), the page retries on the same period.

A log can hold several runs: each run of crowd_monitor.py writes a "start" alert first, so the
dashboard splits the log at those rows and shows the latest run unless you pick another.

Hosted demo (CROWD_DASHBOARD_DEMO=1, set by app/streamlit_app.py): instead of a path on the server,
the sidebar offers the recorded sample logs in eval_results/ and an upload box for a log of your own.

The query functions take an sqlite3 connection and return plain values or pandas DataFrames, so
they are tested without Streamlit (tests/test_dashboard.py).
"""
import argparse
import hashlib
import math
import os
import sqlite3
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = "crowd_log.db"
# Recorded runs shipped with the repository, offered by the hosted demo.
SAMPLE_LOGS = {
    "Street CCTV, 63 s, with --flow line crossings": ROOT / "eval_results" / "task5_run.db",
    "Street CCTV, 60 s, played at real-time pace": ROOT / "eval_results" / "task6_live.db",
    "ShanghaiTech Part A slideshow, YOLO then DM-Count": ROOT / "eval_results" / "task4_run.db",
}
UPLOAD_DIR = Path(tempfile.gettempdir()) / "crowd_dashboard_uploads"
REFRESH_SECONDS = 2.0   # auto-refresh period; 0 turns it off
WINDOWS = {"Last minute": 60, "Last 5 minutes": 300, "Last 15 minutes": 900,
           "Last hour": 3600, "Whole run": None}
DEFAULT_WINDOW = "Last 5 minutes"
STALE_SECONDS = 10.0    # a run whose last reading is older than this is shown as not live

LEVELS = ["LOW", "MEDIUM", "HIGH"]
# Status colours (good / warning / critical) from the dataviz reference palette; always shown with
# the level's name, never colour alone.
LEVEL_COLOURS = {"LOW": "#0ca30c", "MEDIUM": "#fab219", "HIGH": "#d03b3b"}
LEVEL_BADGE = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
# Categorical slots in fixed order; zones get them in order of name, so a zone keeps its colour.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
                  "#e34948"]


# ---------------------------------------------------------------- queries (no Streamlit)

def open_log(path):
    """Read-only connection; raises FileNotFoundError rather than creating an empty database."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no log at {p.resolve()}")
    return sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)


def save_upload(data, folder=UPLOAD_DIR):
    """Write an uploaded log to a file named by its hash, so open_log can read it like any other
    (SQLite needs a file). Identical uploads share one file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (hashlib.sha256(data).hexdigest()[:20] + ".db")
    if not path.exists():
        path.write_bytes(data)
    return path


def tables(db):
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def runs(db):
    """One row per run: run (1-based), start, detail, and the [lo, hi) ts bounds of its rows.
    Runs are split at "start" alerts. Rows logged before the first start belong to run 1, and a
    log with no start rows is a single run."""
    cols = ["run", "start", "detail", "lo", "hi"]
    if "alerts" not in tables(db):
        return pd.DataFrame(columns=cols)
    starts = db.execute("SELECT ts, detail FROM alerts WHERE kind='start' ORDER BY ts").fetchall()
    if not starts:
        first = db.execute("SELECT MIN(ts) FROM readings").fetchone()[0] \
            if "readings" in tables(db) else None
        starts = [(first, "")]
    rows = []
    for i, (ts, detail) in enumerate(starts):
        lo = -math.inf if i == 0 else ts
        hi = starts[i + 1][0] if i + 1 < len(starts) else math.inf
        rows.append((i + 1, ts, detail or "", lo, hi))
    return pd.DataFrame(rows, columns=cols)


def run_info(detail):
    """The key=value pairs of a start row's detail, e.g. {'source': ..., 'calib': '-'}."""
    out = {}
    for part in (detail or "").split():
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def density_unit(detail):
    """Zone density unit for a run: people/m2 with a calibration, else people per 100 px block."""
    calib = run_info(detail).get("calib", "-")
    return "people/m²" if calib not in ("-", "", "None") else "people per 100×100 px"


def _query(db, sql, lo, hi, since=None, order="ts"):
    lower = lo if since is None else max(lo, since)
    return pd.read_sql_query(f"{sql} WHERE ts >= ? AND ts < ? ORDER BY {order}", db,
                             params=(_bound(lower), _bound(hi)))


def _bound(x):
    # SQLite has no infinity literal in parameters on every build; use the float range instead
    return -1e308 if x == -math.inf else 1e308 if x == math.inf else x


def readings(db, lo, hi, since=None):
    """Overall readings of a run, oldest first."""
    return _query(db, "SELECT ts, people, peak, level, decision, result_age_s, detector "
                      "FROM readings", lo, hi, since)


def zone_readings(db, lo, hi, since=None):
    """Per-zone readings of a run, oldest first."""
    if "zone_readings" not in tables(db):
        return pd.DataFrame(columns=["ts", "zone", "people", "density"])
    return _query(db, "SELECT ts, zone, people, density FROM zone_readings", lo, hi, since,
                  order="ts, zone")


def alerts(db, lo, hi):
    """Alert history of a run, newest first."""
    return _query(db, "SELECT ts, kind, old_level, new_level, detail FROM alerts", lo, hi,
                  order="ts DESC")


def crossings(db, lo, hi, since=None):
    """Line crossings of a run (--flow), oldest first; empty for logs without the table."""
    if "crossings" not in tables(db):
        return pd.DataFrame(columns=["ts", "line", "direction", "track_id"])
    return _query(db, "SELECT ts, line, direction, track_id FROM crossings", lo, hi, since)


def crossing_counts(db, lo, hi):
    """Crossings per line and direction for the whole run."""
    c = crossings(db, lo, hi)
    if c.empty:
        return pd.DataFrame(columns=["line", "direction", "count"])
    return (c.groupby(["line", "direction"], sort=True).size()
             .reset_index(name="count"))


def latest(db, lo, hi):
    """(latest reading as a dict or None, that reading's zones as a DataFrame)."""
    r = pd.read_sql_query(
        "SELECT ts, people, peak, level, decision, result_age_s, detector FROM readings "
        "WHERE ts >= ? AND ts < ? ORDER BY ts DESC LIMIT 1", db, params=(_bound(lo), _bound(hi)))
    if r.empty:
        return None, pd.DataFrame(columns=["zone", "people", "density"])
    row = r.iloc[0].to_dict()
    z = pd.DataFrame(columns=["zone", "people", "density"])
    if "zone_readings" in tables(db):
        z = pd.read_sql_query("SELECT zone, people, density FROM zone_readings WHERE ts = ? "
                              "ORDER BY zone", db, params=(row["ts"],))
    return row, z


def window_start(last_ts, seconds):
    """Lower ts bound for a trailing window ending at the run's last reading (None = whole run).
    Anchored to the last reading, not the clock, so a finished run still shows its final minutes."""
    if seconds is None or last_ts is None:
        return None
    return last_ts - seconds


def level_spans(r):
    """Consecutive readings with the same level merged into spans: level, start, end, readings.
    Each span ends where the next one starts. The last ends one median reading interval after
    the last reading, so a level held for a single reading still shows."""
    cols = ["level", "start", "end", "readings"]
    if r.empty:
        return pd.DataFrame(columns=cols)
    r = r.sort_values("ts")
    group = (r["level"] != r["level"].shift()).cumsum()
    spans = r.groupby(group).agg(level=("level", "first"), start=("ts", "first"),
                                 readings=("ts", "size")).reset_index(drop=True)
    step = r["ts"].diff().median() if len(r) > 1 else 0.0
    spans["end"] = list(spans["start"].iloc[1:]) + [r["ts"].iloc[-1] + step]
    return spans[cols]


def zone_colours(zone_names):
    """Fixed colour per zone (sorted by name), folding zones past the palette into grey."""
    names = sorted(set(zone_names))
    return {z: SERIES_COLOURS[i] if i < len(SERIES_COLOURS) else "#8a8984"
            for i, z in enumerate(names)}


# ---------------------------------------------------------------- Streamlit app

def parse_args(argv):
    p = argparse.ArgumentParser(description="Dashboard for a crowd_monitor.py --log database")
    p.add_argument("--db", default=os.environ.get("CROWD_LOG_DB", DEFAULT_DB),
                   help="SQLite log written by crowd_monitor.py --log (env CROWD_LOG_DB)")
    p.add_argument("--refresh", type=float,
                   default=float(os.environ.get("CROWD_LOG_REFRESH", REFRESH_SECONDS)),
                   help="auto-refresh period in seconds, 0 for off (default %(default)s, "
                        "env CROWD_LOG_REFRESH)")
    args, _ = p.parse_known_args(argv)
    return args


def run_label(run):
    when = (pd.Timestamp.fromtimestamp(run.start).strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(run.start) else "no start row")
    source = run_info(run.detail).get("source")
    return f"Run {int(run.run)} · {when}" + (f" · {source}" if source else "")


def pick_run(all_runs, choice):
    """The run row for a sidebar choice: 0 means the latest run, which may be newer than the page."""
    if choice and (all_runs.run == choice).any():
        return all_runs[all_runs.run == choice].iloc[0]
    return all_runs.iloc[-1]


def local_times(ts):
    """Unix seconds -> naive local datetimes (what the person reading the dashboard expects)."""
    return pd.to_datetime(pd.Series(ts, dtype="float64").map(
        lambda t: pd.Timestamp.fromtimestamp(t)))


def ago(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h ago"
    return f"{seconds / 86400:.0f} days ago"


def level_badge(level):
    colour = LEVEL_BADGE.get(level, "gray")
    icon = {"LOW": ":material/check_circle:", "MEDIUM": ":material/warning:",
            "HIGH": ":material/error:"}.get(level, "")
    return f":{colour}-badge[{icon} {level}]"


def time_axis(alt):
    """24-hour clock labels. Vega's default multi-format uses a 12-hour clock without am/pm
    (17:31 showed as 05:31) and labels a short window in milliseconds (.500)."""
    return alt.Axis(format="%H:%M:%S", labelOverlap=True)


def line_chart(alt, df, y, title, colour=None, colours=None, y_title=None, step=False):
    """Time series with a hover rule and tooltip. `colour` names the series field; `colours`
    maps series to fixed colours; `step` draws counts that change at events (cumulative totals)."""
    base = alt.Chart(df).encode(x=alt.X("time:T", title=None, axis=time_axis(alt)))
    tip = [alt.Tooltip("time:T", title="Time", format="%H:%M:%S"),
           alt.Tooltip(f"{y}:Q", title=y_title or y, format=".2f")]
    if colour:
        enc = alt.Color(f"{colour}:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=list(colours), range=list(colours.values())))
        tip.insert(1, alt.Tooltip(f"{colour}:N", title=colour.capitalize()))
    shape = {"interpolate": "step-after"} if step else {}
    if colour:
        lines = base.mark_line(strokeWidth=2, **shape).encode(y=alt.Y(f"{y}:Q", title=y_title),
                                                              color=enc)
    else:
        lines = base.mark_line(strokeWidth=2, color=SERIES_COLOURS[0], **shape).encode(
            y=alt.Y(f"{y}:Q", title=y_title))
    hover = alt.selection_point(fields=["time"], nearest=True, on="pointerover", empty=False,
                                clear="pointerout")
    points = base.mark_circle(size=64, opacity=0).encode(
        y=f"{y}:Q", tooltip=tip).add_params(hover)
    rule = base.mark_rule(color="#8a8984").encode(
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)))
    dots = (base.mark_circle(size=64).encode(y=f"{y}:Q", **({"color": enc} if colour else {}))
            .transform_filter(hover))
    return (lines + rule + points + dots).properties(title=title, height=240)


def level_strip(alt, spans):
    """One bar per level span. No height or title on the chart: inside Streamlit, a fixed height
    plus a title or top legend left no room for the bar (checked in a browser), so the page puts
    the title above it and the legend goes below."""
    df = spans.assign(start_t=local_times(spans["start"]), end_t=local_times(spans["end"]),
                      row="Level")
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("start_t:T", title=None, axis=time_axis(alt)), x2="end_t:T",
        y=alt.Y("row:N", title=None, axis=None),
        color=alt.Color("level:N", title="Level",
                        legend=alt.Legend(orient="bottom", direction="horizontal"),
                        scale=alt.Scale(domain=LEVELS, range=[LEVEL_COLOURS[l] for l in LEVELS])),
        tooltip=[alt.Tooltip("level:N", title="Level"),
                 alt.Tooltip("start_t:T", title="From", format="%H:%M:%S"),
                 alt.Tooltip("end_t:T", title="To", format="%H:%M:%S"),
                 alt.Tooltip("readings:Q", title="Readings")],
    )


def app():
    import altair as alt
    import streamlit as st

    args = parse_args(sys.argv[1:])
    st.set_page_config(page_title="Crowd monitor", page_icon=":material/groups:", layout="wide")

    demo = os.environ.get("CROWD_DASHBOARD_DEMO") == "1"
    with st.sidebar:
        st.header("Log")
        if demo:
            # No free-text path on a public server: bundled samples or an uploaded file only.
            mode = st.radio("Show", ["Sample run", "Upload a log"], horizontal=True)
            if mode == "Sample run":
                path = SAMPLE_LOGS[st.selectbox("Sample", list(SAMPLE_LOGS))]
            else:
                up = st.file_uploader("Log from `crowd_monitor.py --log run.db`",
                                      type=["db", "sqlite", "sqlite3"])
                path = save_upload(up.getvalue()) if up is not None else None
        else:
            path = st.text_input("Database", value=args.db,
                                 help="SQLite file written by crowd_monitor.py --log")
        refresh = st.number_input("Refresh every (s, 0 = off)", min_value=0.0, max_value=600.0,
                                  value=float(args.refresh), step=1.0, format="%g")
        window_name = st.selectbox("Charts show", list(WINDOWS),
                                   index=list(WINDOWS).index(DEFAULT_WINDOW))
    if demo:
        st.info("Demo: recorded runs of `crowd_monitor.py` on test videos (a street CCTV clip and "
                "ShanghaiTech images), not live or Kumbh data. Run the monitor with `--log` and "
                "upload the file in the sidebar to see your own.", icon=":material/info:")
        if path is None:
            st.title("Crowd monitor")
            st.caption("Upload a log in the sidebar.")
            return
    # The log is opened for each read and closed straight after. A connection held between
    # refreshes stops the writer folding its WAL file back into the database when it exits, and
    # on Windows it stops the log being moved or deleted.
    try:
        with closing(open_log(path)) as db:
            known = runs(db)
    except (FileNotFoundError, sqlite3.Error) as e:
        st.title("Crowd monitor")
        st.error(f"Can't open the log: {e}")
        st.caption("Start a run with `python crowd_monitor.py ... --log <file>` "
                   "and point the sidebar at that file."
                   + (f" Retrying every {refresh:g} s." if refresh > 0 else ""))
        if refresh > 0:  # the monitor may not have created the log yet
            time.sleep(refresh)
            st.rerun()
        return
    labels = {"Latest run (follows new runs)": 0}
    labels.update({run_label(r): int(r.run) for r in list(known.itertuples())[::-1]})
    choice = labels[st.sidebar.selectbox("Run", list(labels))]

    # Only this part re-runs on the timer (sidebar widgets can't live inside a fragment).
    @st.fragment(run_every=refresh if refresh > 0 else None)
    def body():
        try:
            with closing(open_log(path)) as db:
                show(db)
        except (FileNotFoundError, sqlite3.Error) as e:
            st.error(f"Can't read the log: {e}")

    def show(db):
        all_runs = runs(db)
        if all_runs.empty:
            st.title("Crowd monitor")
            st.info("The log has no runs yet.")
            return
        run = pick_run(all_runs, choice)
        chosen = int(run.run)
        lo, hi = run.lo, run.hi
        info = run_info(run.detail)
        unit = density_unit(run.detail)

        st.title("Crowd monitor")
        st.caption(" · ".join(f"{k}: {v}" for k, v in info.items()) or "No run details logged")

        now, zones_now = latest(db, lo, hi)
        if now is None:
            st.info("This run has no readings yet.")
        else:
            age = time.time() - now["ts"]
            live = age <= STALE_SECONDS and chosen == all_runs.run.max()
            st.markdown(f"**Level** {level_badge(now['level'])} &nbsp; **Decision** "
                        f"`{now['decision']}` &nbsp; "
                        + (f":green-badge[:material/sensors: live]" if live
                           else f":gray-badge[last reading {ago(age)}]"))
            c = st.columns(4)
            c[0].metric("People in view", f"{now['people']:.1f}")
            c[1].metric("Peak local density", f"{now['peak']:.2f}", help=unit)
            c[2].metric("Result age", f"{now['result_age_s']:.2f} s",
                        help="How old the detector's result was when this reading was logged")
            c[3].metric("Detector", str(now["detector"]))
            if not zones_now.empty:
                st.dataframe(zones_now.rename(columns={"zone": "Zone", "people": "People",
                                                       "density": f"Density ({unit})"}),
                             hide_index=True, width="stretch",
                             column_config={"People": st.column_config.NumberColumn(format="%.1f"),
                                            f"Density ({unit})":
                                                st.column_config.NumberColumn(format="%.3f")})

        last_ts = None if now is None else now["ts"]
        since = window_start(last_ts, WINDOWS[window_name])
        r = readings(db, lo, hi, since)
        if not r.empty:
            r = r.assign(time=local_times(r["ts"]))
            st.subheader("Trends")
            st.markdown("**Level over time**")
            st.altair_chart(level_strip(alt, level_spans(r)), width="stretch")
            st.altair_chart(line_chart(alt, r, "people", "People in view",
                                       y_title="People"), width="stretch")
        z = zone_readings(db, lo, hi, since)
        if not z.empty:
            z = z.assign(time=local_times(z["ts"]))
            colours = zone_colours(z["zone"])
            left, right = st.columns(2)
            left.altair_chart(line_chart(alt, z, "people", "People per zone", colour="zone",
                                         colours=colours, y_title="People"), width="stretch")
            right.altair_chart(line_chart(alt, z, "density", "Density per zone", colour="zone",
                                          colours=colours, y_title=unit), width="stretch")

        counts = crossing_counts(db, lo, hi)
        if not counts.empty:
            st.subheader("Line crossings")
            st.dataframe(counts.rename(columns={"line": "Line", "direction": "Direction",
                                                "count": "Crossings"}),
                         hide_index=True, width="content")
            cx = crossings(db, lo, hi, since)
            if not cx.empty:
                cx = cx.sort_values("ts", ignore_index=True)
                cx = cx.assign(time=local_times(cx["ts"]),
                               series=cx["line"] + ": " + cx["direction"])
                cx["total"] = cx.groupby("series").cumcount() + 1
                st.altair_chart(line_chart(alt, cx, "total", "Cumulative crossings in window",
                                           colour="series", colours=zone_colours(cx["series"]),
                                           y_title="Crossings", step=True), width="stretch")

        st.subheader("Alert history")
        a = alerts(db, lo, hi)
        if a.empty:
            st.caption("No alerts in this run.")
        else:
            a = a.assign(time=local_times(a["ts"]))[["time", "kind", "old_level", "new_level",
                                                    "detail"]]
            st.dataframe(a.rename(columns={"time": "Time", "kind": "Kind", "old_level": "From",
                                           "new_level": "To", "detail": "Detail"}),
                         hide_index=True, width="stretch",
                         column_config={"Time": st.column_config.DatetimeColumn(
                             format="YYYY-MM-DD HH:mm:ss")})
        # times are the dashboard server's local time, which is UTC on most hosts
        st.caption(f"Read {time.strftime('%H:%M:%S')}"
                   + (f", refreshing every {refresh:g} s" if refresh > 0 else ", auto-refresh off")
                   + f". Times are {time.strftime('%Z')}.")

    body()


if __name__ == "__main__":
    app()
