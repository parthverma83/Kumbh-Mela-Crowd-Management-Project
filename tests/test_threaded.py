import os
import sys
import threading
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crowd_monitor
from crowd_monitor import InferenceWorker, RealtimeReader, gap_summary, use_threads

WAIT = 5  # s; generous upper bound so a slow machine doesn't fail a correct test


class GatedDetector:
    """analyse() blocks until released, and records the first pixel of every frame it sees."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.seen = []

    def analyse(self, frame):
        self.seen.append(int(frame[0, 0, 0]))
        self.started.set()
        assert self.release.wait(WAIT)
        self.release.clear()
        return [], np.full((2, 2), float(frame[0, 0, 0]), dtype=np.float32)


def frame_of(value):
    return np.full((4, 4, 3), value, dtype=np.uint8)


def wait_for(cond, timeout=WAIT):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_worker_drops_stale_frames_and_runs_the_newest():
    det = GatedDetector()
    w = InferenceWorker(det)
    try:
        w.submit(frame_of(1))
        assert det.started.wait(WAIT)  # busy on frame 1
        for v in (2, 3, 4):            # arrive while busy; only the newest should run
            w.submit(frame_of(v))
        det.release.set()
        assert wait_for(lambda: len(det.seen) == 2)
        det.release.set()
        assert wait_for(lambda: w.poll() is not None and w.poll().seq == 2)
        assert det.seen == [1, 4]
        assert w.dropped == 2
        assert w.poll().density[0, 0] == 4
    finally:
        det.release.set()
        w.close()


def test_worker_gets_a_copy_so_drawing_on_the_frame_cannot_change_it():
    det = GatedDetector()
    w = InferenceWorker(det)
    try:
        w.submit(frame_of(1))
        assert det.started.wait(WAIT)
        frame = frame_of(7)
        w.submit(frame)
        frame[:] = 200  # the main loop draws on its frame straight after submitting
        det.release.set()
        assert wait_for(lambda: len(det.seen) == 2)
        det.release.set()
        assert det.seen == [1, 7]
    finally:
        det.release.set()
        w.close()


def test_result_keeps_the_capture_time_of_its_frame():
    det = GatedDetector()
    det.release.set()
    w = InferenceWorker(det)
    try:
        w.submit(frame_of(1), captured_at=123.5)
        assert wait_for(lambda: w.poll() is not None)
        assert w.poll().captured_at == 123.5
    finally:
        w.close()


class FailingDetector:
    def analyse(self, frame):
        raise ValueError("model exploded")


def test_worker_error_is_raised_in_the_caller():
    w = InferenceWorker(FailingDetector())
    w.submit(frame_of(1))
    with pytest.raises(ValueError, match="model exploded"):
        assert wait_for(lambda: w.poll() and False)  # poll() raises once the worker fails
    assert w.close()


def test_idle_worker_stops_cleanly():
    w = InferenceWorker(GatedDetector())
    assert w.close(timeout=WAIT)
    assert not w._thread.is_alive()


def test_close_waits_for_an_inference_in_progress():
    det = GatedDetector()
    w = InferenceWorker(det)
    w.submit(frame_of(1))
    assert det.started.wait(WAIT)
    threading.Timer(0.2, det.release.set).start()
    assert w.close(timeout=WAIT)
    assert not w._thread.is_alive()


def test_threads_default_on_for_cameras_and_off_for_files():
    assert use_threads("auto", "0") and use_threads("auto", "2")
    assert not use_threads("auto", "crowd.mp4")
    assert use_threads("on", "crowd.mp4")
    assert not use_threads("off", "0")


class FakeCap:
    def __init__(self, n):
        self.n, self.pos = n, 0

    def grab(self):
        if self.pos >= self.n:
            return False
        self.pos += 1
        return True

    def read(self):
        if self.pos >= self.n:
            return False, None
        self.pos += 1
        return True, self.pos - 1


def test_realtime_reader_skips_frames_when_behind_and_waits_when_ahead():
    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    r = RealtimeReader(FakeCap(100), fps=10, clock=lambda: t[0], sleep=sleep)
    assert r.read() == (True, 0)
    t[0] += 0.02                  # fast: waits until frame 1 is due at 0.1 s
    assert r.read() == (True, 1)
    assert slept[-1] == pytest.approx(0.08)
    t[0] = 1.05                   # slow: frames 2-9 have passed, 10 is the newest due
    assert r.read() == (True, 10)
    assert r.skipped == 8
    t[0] = 100                    # past the end of the file
    assert r.read() == (False, None)


def test_gap_summary_reports_median_p99_max():
    s = gap_summary([0.04] * 99 + [1.0])
    assert "median 40 ms" in s and "max 1000 ms" in s


# ---------------- main loop with a stand-in detector ----------------

def write_video(path, n=8, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
    for _ in range(n):
        w.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    w.release()


class RecordingArduino:
    sent = []

    def __init__(self, port):
        RecordingArduino.sent = []

    def send(self, code):
        RecordingArduino.sent.append(code)

    def close(self):
        pass


def run_main(monkeypatch, tmp_path, detector, *extra):
    video = tmp_path / "in.mp4"
    write_video(video)
    monkeypatch.setattr(crowd_monitor, "build_detector", lambda args: detector)
    monkeypatch.setattr(crowd_monitor, "ArduinoSignal", RecordingArduino)
    out = tmp_path / "out.mp4"
    crowd_monitor.main(["--source", str(video), "--headless", "--heatmap-out", "",
                        "--output", str(out), *extra])
    return out


class SlowDetector:
    """No result for the whole (short) run."""

    def analyse(self, frame):
        time.sleep(1.0)
        return [], np.zeros(frame.shape[:2], dtype=np.float32)


def test_main_sends_nothing_before_the_first_result(monkeypatch, tmp_path, capsys):
    out = run_main(monkeypatch, tmp_path, SlowDetector(), "--threaded", "on")
    assert RecordingArduino.sent == []
    assert "waiting for first result" in capsys.readouterr().out
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8  # frames still shown, without stats
    cap.release()


def test_main_raises_a_worker_error(monkeypatch, tmp_path):
    class Failing:
        def analyse(self, frame):
            raise RuntimeError("worker broke")

    with pytest.raises(RuntimeError, match="worker broke"):
        # --realtime paces the 8 frames over ~0.3 s, so the error arrives mid-run
        run_main(monkeypatch, tmp_path, Failing(), "--threaded", "on", "--realtime")


def test_main_unthreaded_file_keeps_every_frame_and_signals(monkeypatch, tmp_path):
    class Instant:
        def analyse(self, frame):
            return [], np.zeros(frame.shape[:2], dtype=np.float32)

    out = run_main(monkeypatch, tmp_path, Instant())
    assert RecordingArduino.sent == ["G"] * 8
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    cap.release()
