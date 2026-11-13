"""
Crowd monitor: frame -> density map -> heatmap, density level, redirection, Arduino signal.

Every detector produces a density map (people per pixel, summing to the head count):
  yolo / roboflow  - detected people spread as small blobs (good for sparse scenes)
  csrnet / dmcount - pretrained crowd-counting networks (needed for dense crowds,
                     where individual people can no longer be detected)

With a ground calibration the monitor reports real crowd density in people/m^2 and
classifies on that; without one it falls back to raw head-count thresholds.

Examples:
    python crowd_monitor.py --source 0                          # webcam, YOLOv8
    python crowd_monitor.py --source crowd.mp4 --detector dmcount --output out.mp4 --headless
    python crowd_monitor.py --source 0 --calibrate ghat.json --calib-size 10x6   # click 4 head-height points
    python crowd_monitor.py --source 0 --detector csrnet --calib ghat.json
    python crowd_monitor.py --source 0 --serial COM7            # drive Arduino LEDs
    python crowd_monitor.py --source crowd.mp4 --detector dmcount --realtime --threaded on
    python crowd_monitor.py --source 0 --detector dmcount --calib ghat.json --zones zones.json
    python crowd_monitor.py --source 0 --detector dmcount --zones zones.json --log run.db   # SQLite log
    python crowd_monitor.py --source walk.mp4 --flow --zones zones_cctv_walk.json --log run.db  # crossings, stalls
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# ---------------- DEFAULT SETTINGS ----------------
CONF_TH = 0.40

HEAT_DECAY = 0.90      # per-frame EMA weight of the old heatmap (higher = slower fade)
BLOB_SIGMA = 12        # px; spread of one detected person in the density map
HEAD_OFFSET = 0.4      # head position above the box centre, as a fraction of box height

LOW_TH = 5             # uncalibrated: <= 5 people -> GREEN
MID_TH = 12            # uncalibrated: 6-12 -> YELLOW, > 12 -> RED
WARN_DENSITY = 2.0     # calibrated: people/m^2 at which YELLOW starts
CRIT_DENSITY = 4.0     # calibrated: people/m^2 at which RED starts (crowd-safety guidance
                       # treats ~4-5/m^2 as the onset of dangerous crowding)

REDIRECT_MARGIN = 2    # people difference needed to redirect (uncalibrated)
REDIRECT_FRACTION = 0.15  # ...or this share of the crowd, if larger (2 people is noise at 300)
DENSITY_MARGIN = 0.5   # people/m^2 difference needed to redirect (calibrated)
COOLDOWN = 2.0         # seconds a decision is held before it may change
HYSTERESIS = 0.10      # a level falls only once the value is this fraction below its threshold...
MIN_HOLD = 5.0         # ...and the level has been held this many seconds
SMOOTH_FRAMES = 10     # frames used to average the people count

UNCALIBRATED_BLOCK = 100  # px; uncalibrated local density is shown as people per 100x100 px block
UNCALIBRATED_FULL_SCALE = 60  # people per block drawn as full red on the uncalibrated heatmap

ZONE_MIN_INSIDE = 0.99    # warn when less of a zone than this lies inside the calibrated area

WORKER_STOP_TIMEOUT = 10  # s to wait for an inference in progress when shutting down
GAP_HISTORY = 200_000     # frame gaps kept for the end-of-run summary (~2 h at 25 fps)

ROBOFLOW_MODEL_ID = "crowd-detection-n30gw-isumb/2"
DENSITY_MODELS = {"csrnet": "CSRNet", "dmcount": "DM-Count"}
MODEL_DIR = "exported_models"  # ONNX exports of the density networks, for --backend onnx/openvino

# flow (--flow): YOLO + ByteTrack, sparse to moderate scenes only
TRACK_CONF = 0.10         # detections passed to ByteTrack (its track_low_thresh; it starts tracks at 0.25)
SPEED_WINDOW = 1.0        # s; a track's speed is its displacement over this window
TRACK_FORGET = 5.0        # s; forget a track's last position after this long unseen
LINE_BAND = 15.0          # px; a crossing must go from beyond this distance on one side to the other
STALL_SPEED = 0.2         # m/s, calibrated: median track speed below this is "not moving"
STALL_SPEED_PX = 15.0     # px/s, uncalibrated PLACEHOLDER: 0.2 m/s if a 1.7 m person is ~130 px tall
STALL_SECONDS = 10.0      # how long a dense zone must stay slow before a stall alert
STALL_MIN_TRACKS = 3      # tracks needed in a zone before its median speed is trusted
# ---------------------------------------------------

HOLD = "HOLD / BALANCED"
TO_LEFT = "REDIRECT -> LEFT"
TO_RIGHT = "REDIRECT -> RIGHT"

# density level -> (signal name, Arduino byte, BGR colour)
LEVELS = {
    "LOW": ("GREEN", "G", (0, 255, 0)),
    "MEDIUM": ("YELLOW", "Y", (0, 255, 255)),
    "HIGH": ("RED", "R", (0, 0, 255)),
}


@dataclass
class Detection:
    x: int  # box centre
    y: int
    w: int
    h: int
    conf: float
    track_id: int = None  # set by the tracker (--flow)


def head_point(d):
    """Where a detection's head is: the plane that calibration, zones and lines are drawn on."""
    return float(d.x), float(max(d.y - int(d.h * HEAD_OFFSET), 0))


# ---------------- Pure logic (unit tested) ----------------

def classify(value, low, mid):
    """LOW up to `low`, MEDIUM up to `mid`, HIGH above."""
    if value <= low:
        return "LOW"
    if value <= mid:
        return "MEDIUM"
    return "HIGH"


LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


class LevelTracker:
    """
    classify() with hysteresis, so a value hovering at a threshold doesn't flicker.
    Rises as soon as classify() says so. Falls only once the value is at or below the
    threshold minus `margin` x threshold AND the current level has been held `min_hold` s.
    margin=0, min_hold=0 is plain classify().
    """

    def __init__(self, low, mid, margin=HYSTERESIS, min_hold=MIN_HOLD, clock=time.monotonic):
        self.low, self.mid = low, mid
        self.margin = margin
        self.min_hold = min_hold
        self.clock = clock
        self.level = None
        self.since = None  # when the current level was entered

    def update(self, value):
        now = self.clock()
        raw = classify(value, self.low, self.mid)
        if self.level is None or LEVEL_ORDER[raw] > LEVEL_ORDER[self.level]:
            self.level, self.since = raw, now
        elif LEVEL_ORDER[raw] < LEVEL_ORDER[self.level] and now - self.since >= self.min_hold:
            keep = 1 - self.margin
            lower = classify(value, self.low * keep, self.mid * keep)
            if LEVEL_ORDER[lower] < LEVEL_ORDER[self.level]:
                self.level, self.since = lower, now
        return self.level


def redirect_decision(left, right, margin):
    # send people away from the busier side
    if left > right + margin:
        return TO_RIGHT
    if right > left + margin:
        return TO_LEFT
    return HOLD


class DecisionLock:
    """Holds a decision for `cooldown` seconds so the signage doesn't flicker."""

    def __init__(self, cooldown=COOLDOWN, clock=time.monotonic):
        self.cooldown = cooldown
        self.clock = clock
        self.decision = HOLD
        self.changed_at = clock()

    def update(self, new_decision):
        now = self.clock()
        if new_decision != self.decision and now - self.changed_at >= self.cooldown:
            self.decision = new_decision
            self.changed_at = now
        return self.decision


def density_from_detections(dets, shape, sigma=BLOB_SIGMA):
    """
    One unit of mass per person, Gaussian-spread, so the map sums to the count.
    Mass goes on the head, near the top of the box, to match the density networks
    and the head-height calibration plane.
    """
    h, w = shape[:2]
    density = np.zeros((h, w), dtype=np.float32)
    for d in dets:
        _, head_y = head_point(d)
        if 0 <= d.x < w and head_y < h:
            density[int(head_y), d.x] += 1.0
    if dets:
        density = cv2.GaussianBlur(density, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    return density


def resize_density(density, size):
    """Resize a density map to (w, h) while preserving its total count."""
    total = float(density.sum())
    out = cv2.resize(density, size, interpolation=cv2.INTER_LINEAR)
    s = float(out.sum())
    if s > 0:
        out *= total / s
    return out


# ---------------- Ground calibration ----------------

class Ground:
    """
    Maps pixels to real ground area.

    Calibrated: four image points outlining a W x L metre rectangle at HEAD height, i.e.
    where a standing person's head appears at each corner (in order: near-left, near-right,
    far-right, far-left). Density maps put people on their heads, so points clicked on the
    ground would leave people near the far edge outside the region. The head plane is
    parallel to the ground, so its real size is the same W x L. A homography gives the
    area of every pixel inside that region, which is also the monitored ROI.

    Uncalibrated: every pixel counts as 1 / UNCALIBRATED_BLOCK^2 "blocks", and the whole
    frame is the ROI.
    """

    def __init__(self, shape, points=None, width_m=None, length_m=None):
        h, w = shape[:2]
        self.calibrated = points is not None
        self._blurred_area = {}
        self.H = None  # head-plane pixels -> metres, calibrated only
        if not self.calibrated:
            self.unit = f"/{UNCALIBRATED_BLOCK}px blk"
            self.roi = np.ones((h, w), dtype=np.float32)
            self.pixel_area = np.full((h, w), 1.0 / UNCALIBRATED_BLOCK ** 2, dtype=np.float32)
            self.polygon = None
            self.split_x = np.full(h, w / 2, dtype=np.float32)
            return

        self.unit = "/m2"
        src = np.asarray(points, dtype=np.float32)
        dst = np.float32([[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]])
        H = self.H = cv2.getPerspectiveTransform(src, dst)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
        denom = H[2, 0] * xs + H[2, 1] * ys + H[2, 2]
        # area scale of a projective map is |det H| / denom^3 (Jacobian determinant)
        with np.errstate(divide="ignore", invalid="ignore"):
            area = np.abs(np.linalg.det(H)) / np.abs(denom) ** 3
        self.polygon = src.astype(np.int32)
        self.roi = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(self.roi, [self.polygon], 1.0)
        self.pixel_area = np.nan_to_num(area * self.roi).astype(np.float32)

        # left/right zones split along the ground centre line (near-mid to far-mid),
        # so each zone is the same real width regardless of perspective
        near_mid, far_mid = (src[0] + src[1]) / 2, (src[3] + src[2]) / 2
        t = (np.arange(h) - near_mid[1]) / ((far_mid[1] - near_mid[1]) or 1)
        self.split_x = (near_mid[0] + t * (far_mid[0] - near_mid[0])).astype(np.float32)

    @classmethod
    def load(cls, path, shape):
        with open(path) as f:
            cfg = json.load(f)
        pts = np.asarray(cfg["image_points"], dtype=np.float32)
        fw, fh = cfg.get("frame_size", [shape[1], shape[0]])
        pts *= [shape[1] / fw, shape[0] / fh]  # calibration may come from a different resolution
        return cls(shape, pts, cfg["width_m"], cfg["length_m"])

    @property
    def total_area(self):
        return float(self.pixel_area.sum())

    def to_plane(self, points):
        """Head-height image points -> metres on the calibrated plane; pixels if uncalibrated."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self.H is None or not len(pts):
            return pts
        return cv2.perspectiveTransform(pts[None], self.H)[0]

    def zone_masks(self):
        h, w = self.roi.shape
        left = (np.arange(w)[None, :] < self.split_x[:, None]).astype(np.float32) * self.roi
        return left, self.roi - left

    def local_density(self, density, sigma):
        """Normalised convolution: people / area over a Gaussian neighbourhood."""
        # a wide blur loses nothing at quarter size, and this runs on every frame
        h, w = density.shape
        s = sigma / 4
        small = cv2.resize(density * self.roi, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        num = cv2.GaussianBlur(small, (0, 0), s)
        if sigma not in self._blurred_area:  # the area map never changes; blur it once
            area = cv2.resize(self.pixel_area, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            self._blurred_area[sigma] = cv2.GaussianBlur(area, (0, 0), s)
        den = self._blurred_area[sigma]
        out = np.zeros_like(num)
        # ignore the ROI fringe, where the neighbourhood is almost entirely outside it
        np.divide(num, den, out=out, where=den > den.max() * 1e-3)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR) * self.roi


# ---------------- Named zones and exits ----------------

@dataclass
class Zone:
    name: str
    polygon: np.ndarray  # N x 2 int32, frame pixels at head height
    mask: np.ndarray     # float32, 1 inside the zone (and inside the calibrated area)
    area: float          # m^2 if calibrated, else UNCALIBRATED_BLOCK^2 px blocks


class Zones:
    """
    Named polygons, drawn at head height like the calibration points, and the exits they
    lead to. Each zone's density is people / area in the ground's unit: people/m^2 with a
    calibration, else people per 100x100 px block.
    """

    def __init__(self, ground, zones, exits=(), lines=()):
        """zones: [(name, polygon in frame pixels)]; exits: [(exit name, zone name)];
        lines: [CountLine] for --flow."""
        self.ground = ground
        self.zones = []
        where = "calibrated area" if ground.calibrated else "frame"
        for name, poly in zones:
            poly = np.round(np.asarray(poly, dtype=np.float64)).astype(np.int32)
            full = np.zeros(ground.roi.shape, dtype=np.float32)
            cv2.fillPoly(full, [poly], 1.0)
            mask = full * ground.roi
            area = float((ground.pixel_area * mask).sum())
            if area <= 0:
                sys.exit(f"Zone {name!r} has no area inside the {where}")
            inside = float(mask.sum() / max(full.sum(), 1))
            if inside < ZONE_MIN_INSIDE:
                print(f"Warning: only {inside:.0%} of zone {name!r} is inside the {where}; "
                      f"only that part is measured")
            self.zones.append(Zone(name, poly, mask, area))
        self.exits = list(exits)
        self.lines = list(lines)

    @classmethod
    def load(cls, path, ground):
        def fail(problem):
            sys.exit(f"{path}: {problem}")

        with open(path) as f:
            cfg = json.load(f)
        h, w = ground.roi.shape
        fw, fh = cfg.get("frame_size", [w, h])
        scale = np.float64([w / fw, h / fh])  # zones may come from a different resolution
        zones, names = [], set()
        for i, z in enumerate(cfg.get("zones", []), 1):
            name = z.get("name")
            poly = np.asarray(z.get("polygon", []), dtype=np.float64)
            if not name:
                fail(f"zone {i} has no name")
            if name in names:
                fail(f"zone name {name!r} is used twice")
            if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
                fail(f"zone {name!r} needs a polygon of at least 3 [x, y] points")
            names.add(name)
            zones.append((name, poly * scale))
        if not zones:
            fail("no zones defined")
        exits = []
        for e in cfg.get("exits", []):
            if not e.get("name"):
                fail("an exit has no name")
            if e.get("zone") not in names:
                fail(f"exit {e['name']!r} refers to unknown zone {e.get('zone')!r}")
            if e["name"] in (x for x, _ in exits):
                fail(f"exit name {e['name']!r} is used twice")
            exits.append((e["name"], e["zone"]))
        lines = []
        for i, ln in enumerate(cfg.get("lines", []), 1):
            name = ln.get("name")
            pts = np.asarray(ln.get("points", []), dtype=np.float64)
            dirs = tuple(ln.get("directions", CountLine.DEFAULT_DIRECTIONS))
            if not name:
                fail(f"line {i} has no name")
            if name in (x.name for x in lines):
                fail(f"line name {name!r} is used twice")
            if pts.shape != (2, 2) or np.allclose(pts[0], pts[1]):
                fail(f"line {name!r} needs 2 different [x, y] points")
            if len(dirs) != 2 or len(set(dirs)) != 2 or not all(isinstance(d, str) and d for d in dirs):
                fail(f"line {name!r} needs 2 different direction names")
            pts = pts * scale
            lines.append(CountLine(name, pts[0], pts[1], dirs))
        return cls(ground, zones, exits, lines)

    def describe(self):
        unit = "m2" if self.ground.calibrated else f"{UNCALIBRATED_BLOCK}x{UNCALIBRATED_BLOCK} px blocks"
        zones = ", ".join(f"{z.name} ({z.area:.1f} {unit})" for z in self.zones)
        exits = ", ".join(f"{e} -> {z}" for e, z in self.exits) or "none"
        lines = ", ".join(f"{ln.name} ({ln.directions[0]} / {ln.directions[1]})" for ln in self.lines)
        return f"Zones: {zones}; exits: {exits}" + (f"; lines: {lines}" if lines else "")

    def measure(self, heat, local):
        """Per-zone people, area, density and peak local density, in file order."""
        out = []
        for z in self.zones:
            people = float((heat * z.mask).sum())
            out.append(dict(name=z.name, people=people, area=z.area, density=people / z.area,
                            peak=float((local * z.mask).max())))
        return out


def exit_margin(a, b, calibrated):
    """
    How much lower zone b's density must be than zone a's for b to be clearly better.
    Calibrated: DENSITY_MARGIN people/m^2. Uncalibrated: the left/right people margin,
    divided by the zones' mean area, which is exactly the left/right rule for equal halves.
    """
    if calibrated:
        return DENSITY_MARGIN
    people = max(REDIRECT_MARGIN, REDIRECT_FRACTION * (a["people"] + b["people"]))
    return people / ((a["area"] + b["area"]) / 2)


class ExitChooser:
    """
    Recommends the exit whose zone is least dense. Starts on the best exit, then switches
    only to an exit that is clearly better (exit_margin) once `cooldown` has passed.
    """

    def __init__(self, exits, calibrated, cooldown=COOLDOWN, clock=time.monotonic):
        self.exits = list(exits)
        self.calibrated = calibrated
        self.cooldown = cooldown
        self.clock = clock
        self.current = None  # (exit name, zone name)
        self.changed_at = None

    def update(self, zone_stats):
        by_zone = {z["name"]: z for z in zone_stats}
        now = self.clock()
        if self.current is None:
            # ties go to the exit listed first
            self.current = min(self.exits, key=lambda e: by_zone[e[1]]["density"])
            self.changed_at = now
        elif now - self.changed_at >= self.cooldown:
            here = by_zone[self.current[1]]
            better = [e for e in self.exits
                      if here["density"] - by_zone[e[1]]["density"]
                      > exit_margin(here, by_zone[e[1]], self.calibrated)]
            if better:
                self.current = min(better, key=lambda e: by_zone[e[1]]["density"])
                self.changed_at = now
        return self.current[0]


# ---------------- Flow: line crossings and stalls (--flow) ----------------

@dataclass
class CountLine:
    """
    A named counting line, drawn at head height like the zones. directions[0] ("forward") is
    a crossing from the left of the line to its right, looking from point a to point b. For a
    line drawn left to right across the image, forward means moving down the image.
    """
    name: str
    a: np.ndarray
    b: np.ndarray
    directions: tuple = ("forward", "backward")
    DEFAULT_DIRECTIONS = ("forward", "backward")

    def side(self, p, band=0.0):
        """+1 right of the line, -1 left, 0 within `band` px of it (image coordinates, y down)."""
        d = self.b - self.a
        dist = (d[0] * (p[1] - self.a[1]) - d[1] * (p[0] - self.a[0])) / np.hypot(*d)
        return 0 if abs(dist) <= band else int(np.sign(dist))

    def segment_crosses(self, p, q):
        """True if the move p -> q passes between a and b (p and q already on opposite sides)."""
        v = np.subtract(q, p)
        s1 = v[0] * (self.a[1] - p[1]) - v[1] * (self.a[0] - p[0])
        s2 = v[0] * (self.b[1] - p[1]) - v[1] * (self.b[0] - p[0])
        return s1 * s2 <= 0


class LineCounter:
    """
    Counts tracks crossing named lines, each direction separately, and each track at most
    once per direction per line. A crossing needs the track to go from more than `band` px
    on one side to more than `band` px on the other: a track inside the band keeps its last
    side, so a head point jittering on the line, or stepping onto it and back, is not a
    crossing. A track that disappears for a few frames and comes back still counts, from its
    last position; after `forget` s unseen it is dropped.
    """

    def __init__(self, lines, forget=TRACK_FORGET, band=LINE_BAND):
        self.lines = list(lines)
        self.forget = forget
        self.band = band
        self.counts = {ln.name: {d: 0 for d in ln.directions} for ln in self.lines}
        self.state = {}     # (track, line name) -> (side, point) at its last off-line position
        self.counted = set()  # (track, line name, direction)
        self.seen = {}      # track -> last time seen

    def update(self, points, t):
        """points: {track id: (x, y)} at time t. Returns [(line, direction, track id)] new crossings."""
        events = []
        for tid, p in points.items():
            self.seen[tid] = t
            for ln in self.lines:
                s = ln.side(p, self.band)
                if s == 0:
                    continue
                key = (tid, ln.name)
                old = self.state.get(key)
                self.state[key] = (s, p)
                if old is None or old[0] == s or not ln.segment_crosses(old[1], p):
                    continue
                direction = ln.directions[0] if s > 0 else ln.directions[1]
                if (tid, ln.name, direction) in self.counted:
                    continue
                self.counted.add((tid, ln.name, direction))
                self.counts[ln.name][direction] += 1
                events.append((ln.name, direction, tid))
        for tid in [k for k, seen in self.seen.items() if t - seen > self.forget]:
            del self.seen[tid]
            for ln in self.lines:
                self.state.pop((tid, ln.name), None)
                for d in ln.directions:
                    self.counted.discard((tid, ln.name, d))
        return events


class TrackSpeeds:
    """Speed of each track: displacement over the last `window` s on the ground plane
    (m/s if calibrated, else px/s). Displacement, not path length, so a box jittering in
    place reads as standing still. None until a track has been seen for half a window."""

    def __init__(self, ground, window=SPEED_WINDOW, forget=TRACK_FORGET):
        self.ground = ground
        self.window = window
        self.forget = forget
        self.history = {}  # track -> deque of (t, plane x, plane y)

    def update(self, points, t):
        if points:
            plane = self.ground.to_plane(list(points.values()))
            for tid, (x, y) in zip(points, plane):
                hist = self.history.setdefault(tid, deque())
                hist.append((t, x, y))
                while len(hist) > 2 and t - hist[1][0] >= self.window:
                    hist.popleft()  # keep the newest sample at least `window` old
        for tid in [k for k, h in self.history.items() if t - h[-1][0] > self.forget]:
            del self.history[tid]
        speeds = {}
        for tid in points:
            (t0, x0, y0), (t1, x1, y1) = self.history[tid][0], self.history[tid][-1]
            if t1 - t0 >= self.window / 2:
                speeds[tid] = float(np.hypot(x1 - x0, y1 - y0) / (t1 - t0))
        return speeds


class StallDetector:
    """
    Per zone: a stall starts once the zone has been dense (at or above the warning level)
    with a median track speed below `threshold` for `seconds`, and ends as soon as that stops
    being true. A zone with fewer than `min_tracks` speeds is not judged (not a stall).
    """

    def __init__(self, threshold, seconds=STALL_SECONDS, min_tracks=STALL_MIN_TRACKS):
        self.threshold = threshold
        self.seconds = seconds
        self.min_tracks = min_tracks
        self.since = {}    # zone -> time the slow, dense condition began
        self.stalled = set()

    def update(self, zone, dense, speeds, t):
        """Returns ("stall", old, new, detail) when the zone starts or stops stalling, else None."""
        median = float(np.median(speeds)) if len(speeds) >= self.min_tracks else None
        slow = dense and median is not None and median < self.threshold
        detail = f"{zone}: median speed " + ("n/a" if median is None else f"{median:.2f}") + \
                 f" over {len(speeds)} tracks"
        if slow:
            self.since.setdefault(zone, t)
            if zone not in self.stalled and t - self.since[zone] >= self.seconds:
                self.stalled.add(zone)
                return ("stall", "flowing", "stalled", detail)
            return None
        self.since.pop(zone, None)
        if zone in self.stalled:
            self.stalled.discard(zone)
            return ("stall", "stalled", "flowing", detail)
        return None


class FlowMonitor:
    """Tracks -> line crossings, per-zone median speed and stall alerts."""

    def __init__(self, ground, zones, args):
        self.ground = ground
        self.zones = zones
        self.lines = LineCounter(zones.lines, band=args.line_band)
        self.speeds = TrackSpeeds(ground)
        calibrated = ground.calibrated
        self.low = args.warn_density if calibrated else args.low
        self.mid = args.crit_density if calibrated else args.mid
        threshold = args.stall_speed if args.stall_speed is not None else (
            STALL_SPEED if calibrated else STALL_SPEED_PX)
        self.speed_unit = "m/s" if calibrated else "px/s"
        self.stalls = StallDetector(threshold, args.stall_seconds, args.stall_min_tracks)

    def dense(self, zone_stats):
        """At or above the warning level: people/m^2 if calibrated, else the zone's people
        count against the uncalibrated people thresholds."""
        value = zone_stats["density"] if self.ground.calibrated else zone_stats["people"]
        return classify(value, self.low, self.mid) != "LOW"

    def update(self, dets, t, zone_stats):
        """Adds speed and tracks to each zone's stats; returns (crossings, alerts)."""
        points = {d.track_id: head_point(d) for d in dets if d.track_id is not None}
        crossings = self.lines.update(points, t)
        speeds = self.speeds.update(points, t)
        alerts = []
        for z, s in zip(self.zones.zones, zone_stats):
            inside = [v for tid, v in speeds.items()
                      if cv2.pointPolygonTest(z.polygon.reshape(-1, 1, 2).astype(np.float32),
                                              points[tid], False) >= 0]
            s["tracks"] = len(inside)
            s["speed"] = float(np.median(inside)) if inside else None
            alert = self.stalls.update(z.name, self.dense(s), inside, t)
            if alert:
                alerts.append(alert)
        return crossings, alerts


def run_calibration(cap, path, size):
    """Click 4 head-height points on the first frame, save them with the real size."""
    width_m, length_m = (float(v) for v in size.lower().split("x"))
    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read a frame to calibrate on")
    pts = []
    labels = ["near-left", "near-right", "far-right", "far-left"]
    win = "Calibrate: click HEAD height at near-left, near-right, far-right, far-left (r = reset, Enter = save)"

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x, y])

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    while True:
        view = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, tuple(p), 6, (0, 0, 255), -1)
            put_text(view, labels[i], (p[0] + 8, p[1] - 8), 0.6, (255, 255, 255))
        if len(pts) > 1:
            cv2.polylines(view, [np.int32(pts)], len(pts) == 4, (0, 255, 255), 2)
        if len(pts) < 4:
            put_text(view, f"Click {labels[len(pts)]} at head height", (20, 40), 0.9, (0, 255, 255))
        else:
            put_text(view, f"{width_m} x {length_m} m - Enter to save", (20, 40), 0.9, (0, 255, 0))
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            pts.clear()
        elif key in (13, 10) and len(pts) == 4:
            break
        elif key == 27:
            sys.exit("Calibration cancelled")
    cv2.destroyWindow(win)
    h, w = frame.shape[:2]
    with open(path, "w") as f:
        json.dump({"image_points": pts, "width_m": width_m, "length_m": length_m,
                   "frame_size": [w, h]}, f, indent=2)
    print(f"Saved calibration to {path} ({width_m} x {length_m} m)")


# ---------------- Detectors: frame -> (detections, density map) ----------------

class BoxDetector:
    """Base for detectors that find individual people; density is built from the boxes."""

    def analyse(self, frame):
        dets = self.detect(frame)
        return dets, density_from_detections(dets, frame.shape)


class YoloDetector(BoxDetector):
    """Local Ultralytics YOLO; class 0 is 'person' in COCO."""

    def __init__(self, weights="yolov8n.pt", conf=CONF_TH):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame):
        result = self.model(frame, conf=self.conf, classes=[0], verbose=False)[0]
        dets = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append(Detection((x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1,
                                  float(box.conf[0])))
        return dets


def lapjv_standin(cost, extend_cost=True, cost_limit=np.inf):
    """lap.lapjv's interface on Ultralytics' own solver (SciPy if present, else NumPy):
    an optimal assignment with pairs over cost_limit dropped. This is what Ultralytics
    itself does with use_lap=False."""
    from ultralytics.utils.ops import linear_sum_assignment
    x = np.full(cost.shape[0], -1, dtype=int)
    y = np.full(cost.shape[1], -1, dtype=int)
    if cost.size == 0:
        return 0.0, x, y
    rows, cols = (np.asarray(v, dtype=int) for v in linear_sum_assignment(cost))
    keep = cost[rows, cols] <= cost_limit
    x[rows[keep]] = cols[keep]
    y[cols[keep]] = rows[keep]
    return float(cost[rows[keep], cols[keep]].sum()), x, y


def ensure_assignment_solver():
    """
    Ultralytics' ByteTrack imports `lap`, and if that fails it tries to pip-install it. lap is
    optional here: when it is missing, or can't load (Windows Smart App Control blocks its
    unsigned DLL on the test laptop), a stand-in module with lapjv_standin is registered
    instead, and Ultralytics' auto-install is switched off. Returns which solver is used.
    """
    import importlib
    import types
    os.environ["YOLO_AUTOINSTALL"] = "false"  # read when ultralytics is first imported...
    from ultralytics.utils import checks
    checks.AUTOINSTALL = False                # ...and patched in case it already was
    try:
        lap = importlib.import_module("lap")
        if lap.__version__:
            return f"lap {lap.__version__}"
    except (ImportError, AttributeError):
        pass
    shim = types.ModuleType("lap")
    shim.__version__ = "stand-in"
    shim.lapjv = lapjv_standin
    sys.modules["lap"] = shim
    return "built-in (lap not available)"


class TrackingDetector(YoloDetector):
    """YOLO + ByteTrack through Ultralytics; each detection carries a track id."""

    def __init__(self, weights="yolov8n.pt", conf=TRACK_CONF):
        print(f"Assignment solver: {ensure_assignment_solver()}")
        super().__init__(weights, conf)

    def detect(self, frame):
        result = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.conf,
                                  classes=[0], verbose=False)[0]
        boxes = result.boxes
        if boxes.id is None:  # no confirmed tracks yet
            return []
        dets = []
        for (x1, y1, x2, y2), c, tid in zip(boxes.xyxy.int().tolist(), boxes.conf.tolist(),
                                            boxes.id.int().tolist()):
            dets.append(Detection((x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1, float(c), tid))
        return dets


class RoboflowDetector(BoxDetector):
    """Hosted Roboflow crowd model. API key comes from ROBOFLOW_API_KEY, never source."""

    def __init__(self, model_id=ROBOFLOW_MODEL_ID, conf=CONF_TH):
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            sys.exit("Set the ROBOFLOW_API_KEY environment variable to use --detector roboflow")
        from inference import get_model
        self.model = get_model(model_id=model_id, api_key=api_key)
        self.conf = conf

    def detect(self, frame):
        result = self.model.infer(frame)
        if isinstance(result, list):
            result = result[0]
        preds = result.dict().get("predictions", []) if hasattr(result, "dict") else []
        return [Detection(int(p["x"]), int(p["y"]), int(p["width"]), int(p["height"]),
                          p.get("confidence", 0))
                for p in preds if p.get("confidence", 0) >= self.conf]


def exported_model_path(name, weights, folder=MODEL_DIR):
    return os.path.join(folder, f"{DENSITY_MODELS[name]}_{weights}.onnx")


def onnx_runner(path, threads):
    """ONNX Runtime on the CPU: fn(NCHW float32 array) -> raw output array."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    return lambda x: sess.run(None, {sess.get_inputs()[0].name: x})[0]


def openvino_runner(path, threads, precision="f32", static_hw=None):
    """
    OpenVINO on the CPU. Precision is pinned: on CPUs with bf16 support OpenVINO
    silently defaults to bf16, which changes the counts. static_hw fixes the input size.
    """
    import openvino as ov
    core = ov.Core()
    model = core.read_model(path)
    if static_hw:
        model.reshape([1, 3, *static_hw])
    req = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": threads,
                                            "PERFORMANCE_HINT": "LATENCY",
                                            "INFERENCE_PRECISION_HINT": precision}).create_infer_request()

    def run(x):
        req.infer({0: x})
        return req.get_output_tensor(0).data.copy()
    return run


class DensityDetector:
    """
    Pretrained crowd-counting network (CSRNet or DM-Count via lwcc) that regresses a
    density map directly - works under heavy occlusion where box detectors fail.
    Weights download on first use to C:\\.lwcc\\weights (or /.lwcc/weights).

    backend "onnx" or "openvino" runs the same network exported to ONNX
    (python benchmark_backends.py export) instead of PyTorch.
    """

    MEAN = np.float32([0.485, 0.456, 0.406])
    STD = np.float32([0.229, 0.224, 0.225])

    def __init__(self, name="csrnet", weights="SHA", max_side=1000, model=None, backend="torch",
                 model_dir=MODEL_DIR, precision="f32"):
        import torch
        self.torch = torch
        self.max_side = max_side
        self.backend = backend
        if backend != "torch":
            path = exported_model_path(name, weights, model_dir)
            if not os.path.isfile(path):
                sys.exit(f"No exported model at {path}; create it with: "
                         f"python benchmark_backends.py export --models {name} --density-weights {weights}")
            threads = torch.get_num_threads()  # same thread count as the PyTorch path
            self.run = (onnx_runner(path, threads) if backend == "onnx"
                        else openvino_runner(path, threads, precision))
            return
        if model is None:
            from lwcc import LWCC
            model = LWCC.load_model(DENSITY_MODELS[name], weights)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()

    def preprocess(self, frame):
        if self.max_side:  # max_side=0 feeds the frame untouched, as the published benchmarks do
            h, w = frame.shape[:2]
            scale = min(1.0, self.max_side / max(h, w))
            # networks downsample by 8-16x; keep sides divisible by 16
            nw, nh = max(16, round(w * scale / 16) * 16), max(16, round(h * scale / 16) * 16)
            frame = cv2.resize(frame, (nw, nh))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255 - self.MEAN) / self.STD
        return self.torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0)

    def predict(self, frame):
        """Raw network output. CSRNet's last layer is linear, so values can be negative."""
        x = self.preprocess(frame)
        if self.backend != "torch":
            return self.run(x.numpy())[0, 0].astype(np.float32)
        with self.torch.no_grad():
            out = self.model(x.to(self.device))
        return out[0, 0].cpu().numpy().astype(np.float32)

    def analyse(self, frame):
        density = np.clip(self.predict(frame), 0, None)
        return [], resize_density(density, (frame.shape[1], frame.shape[0]))


@dataclass
class InferenceResult:
    detections: list
    density: np.ndarray
    captured_at: float  # clock time at which the frame it came from was read
    seq: int            # 1 for the first result, then counting up
    media_t: float = None  # that frame's time in the video (files) or capture time (cameras)


class InferenceWorker:
    """
    Runs detector.analyse on a background thread, always on the newest frame.

    submit() replaces a frame that is still waiting, so a slow model skips frames instead
    of falling behind the camera. It stores a copy, because the caller draws on its frame.
    poll() returns the newest result and raises a worker error in the caller's thread.
    """

    def __init__(self, detector, clock=time.perf_counter):
        self.detector = detector
        self.clock = clock
        self.dropped = 0  # frames replaced before the worker got to them
        self._cond = threading.Condition()
        self._pending = None
        self._result = None
        self._error = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self._thread.start()

    def submit(self, frame, captured_at=None, media_t=None):
        frame = frame.copy()
        with self._cond:
            if self._pending is not None:
                self.dropped += 1
            self._pending = (frame, self.clock() if captured_at is None else captured_at, media_t)
            self._cond.notify()

    def poll(self):
        """The newest result, or None before the first one."""
        with self._cond:
            if self._error is not None:
                raise self._error
            return self._result

    def close(self, timeout=WORKER_STOP_TIMEOUT):
        """Stop the thread, letting an inference in progress finish. True if it stopped."""
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self):
        seq = 0
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                frame, captured_at, media_t = self._pending
                self._pending = None
            try:
                dets, density = self.detector.analyse(frame)
            except BaseException as e:  # handed to the main loop by poll()
                with self._cond:
                    self._error = e
                return
            seq += 1
            with self._cond:
                self._result = InferenceResult(dets, density, captured_at, seq, media_t)


def build_detector(args):
    if args.flow:
        if args.detector != "yolo":
            sys.exit("--flow needs --detector yolo: density networks don't find individual people to track")
        return TrackingDetector(args.weights)
    if args.detector == "yolo":
        return YoloDetector(args.weights, args.conf)
    if args.detector == "roboflow":
        return RoboflowDetector(conf=args.conf)
    print(f"Backend: {args.backend}" + (f" ({args.precision})" if args.backend == "openvino" else ""))
    return DensityDetector(args.detector, args.density_weights, args.max_side, backend=args.backend,
                           precision=args.precision)


# ---------------- Analysis ----------------

class CrowdAnalyser:
    """Turns per-frame density maps into smoothed counts, zone stats and a level."""

    def __init__(self, ground, args, clock=time.monotonic, zones=None):
        self.ground = ground
        self.args = args
        self.zones = zones
        self.chooser = (ExitChooser(zones.exits, ground.calibrated, clock=clock)
                        if zones is not None and zones.exits else None)
        self.heat = None  # EMA of the density map
        self.history = deque(maxlen=SMOOTH_FRAMES)
        self.lock = DecisionLock(clock=clock)
        low, mid = ((args.warn_density, args.crit_density) if ground.calibrated
                    else (args.low, args.mid))
        self.levels = LevelTracker(low, mid, args.hysteresis, args.min_hold, clock=clock)
        self.left_mask, self.right_mask = ground.zone_masks()
        self.left_area = float((ground.pixel_area * self.left_mask).sum())
        self.right_area = float((ground.pixel_area * self.right_mask).sum())
        h, w = ground.roi.shape
        self.sigma = max(h, w) / 40  # neighbourhood for local density

    def update(self, density):
        if self.heat is None:
            self.heat = density.copy()
        else:
            self.heat = HEAT_DECAY * self.heat + (1 - HEAT_DECAY) * density

        g, a = self.ground, self.args
        count = float((density * g.roi).sum())
        self.history.append(count)
        people = sum(self.history) / len(self.history)
        left = float((self.heat * self.left_mask).sum())
        right = float((self.heat * self.right_mask).sum())
        self.local = g.local_density(self.heat, self.sigma)
        peak = float(self.local.max())

        if g.calibrated:
            left_d = left / max(self.left_area, 1e-6)
            right_d = right / max(self.right_area, 1e-6)
            level = self.levels.update(peak)
            decision = redirect_decision(left_d, right_d, DENSITY_MARGIN)
        else:
            left_d = right_d = None
            level = self.levels.update(people)
            margin = max(REDIRECT_MARGIN, REDIRECT_FRACTION * (left + right))
            decision = redirect_decision(left, right, margin)

        stats = dict(people=people, left=left, right=right, left_d=left_d, right_d=right_d,
                     peak=peak, unit=g.unit, density=level,
                     avg_density=people / g.total_area if g.calibrated else None,
                     decision=self.lock.update(decision))
        if self.zones is not None:
            stats["zones"] = self.zones.measure(self.heat, self.local)
            if self.chooser:  # named exits replace the left/right decision
                stats["exit"] = self.chooser.update(stats["zones"])
                stats["decision"] = f"USE {stats['exit']}"
        return stats

    def heat_colour(self):
        # fixed scale, so colours mean the same thing every frame. Calibrated: full red just
        # past the critical density. Uncalibrated: square root, so one person is visible
        # and a packed crowd (tens of people per block) doesn't flatten into solid red.
        if self.ground.calibrated:
            level = self.local / (self.args.crit_density * 1.25)
        else:
            level = np.sqrt(self.local / UNCALIBRATED_FULL_SCALE)
        return cv2.applyColorMap((np.clip(level, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)


# ---------------- Arduino ----------------

class ArduinoSignal:
    """Sends G/Y/R over serial, only when the signal changes."""

    def __init__(self, port, baud=9600):
        self.port = None
        self.last = None
        if not port:
            return
        try:
            import serial
            self.port = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # board resets on connect
            print(f"Arduino connected on {port}")
        except Exception as e:
            print(f"Arduino not connected ({e}); skipping LED control")

    def send(self, code):
        if self.port and code != self.last:
            self.port.write(code.encode())
            self.last = code

    def close(self):
        if self.port:
            self.port.close()


# ---------------- Drawing ----------------

def put_text(img, text, org, scale, color, thickness=2):
    # dark outline keeps text readable over the heatmap
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def unit_label(unit):
    return "/m2" if unit == "/m2" else f"/{UNCALIBRATED_BLOCK}px"


def draw_zones(out, zones, stats):
    """Zone outlines with name, people and density; the recommended exit's zone in green."""
    unit = unit_label(stats["unit"])
    exits_by_zone = {}
    for name, zone in zones.exits:
        exits_by_zone.setdefault(zone, []).append(name)
    for z, s in zip(zones.zones, stats["zones"]):
        names = exits_by_zone.get(z.name, [])
        chosen = stats.get("exit") in names
        col = (0, 255, 0) if chosen else (255, 255, 255)
        cv2.polylines(out, [z.polygon], True, col, 4 if chosen else 2)
        # label the bottom-left corner: the stats panel covers the top left of the frame
        lines = [(z.name, 0.8), (f"{s['people']:.0f} people  {s['density']:.2f}{unit}", 0.6)]
        if "tracks" in s:  # --flow
            speed_unit = "m/s" if stats["unit"] == "/m2" else "px/s"
            speed = "-" if s["speed"] is None else f"{s['speed']:.2f}"
            lines.append((f"speed {speed} {speed_unit} ({s['tracks']} tracks)", 0.6))
        lines += [(("USE " if n == stats.get("exit") else "exit ") + n,
                   0.7 if n == stats.get("exit") else 0.6) for n in names]
        x = int(z.polygon[:, 0].min()) + 8
        y = int(z.polygon[:, 1].max()) - 12 - 28 * (len(lines) - 1)
        for i, (text, scale) in enumerate(lines):
            put_text(out, text, (x, y + 28 * i), scale, col)


def draw_overlay(frame, heat_color, detections, ground, stats, fps, age=None, zones=None):
    h, w = frame.shape[:2]
    for d in detections:
        cv2.rectangle(frame, (d.x - d.w // 2, d.y - d.h // 2),
                      (d.x + d.w // 2, d.y + d.h // 2), (0, 255, 0), 2)
    out = cv2.addWeighted(frame, 0.55, heat_color, 0.45, 0)
    if ground.polygon is not None:
        cv2.polylines(out, [ground.polygon], True, (255, 255, 255), 2)

    if zones is not None:
        draw_zones(out, zones, stats)
    if "exit" not in stats:  # left/right redirection, unless named exits replace it
        # zone divider follows the ground centre line
        rows = np.nonzero(ground.roi.any(axis=1))[0]
        y0, y1 = (int(rows[0]), int(rows[-1])) if len(rows) else (0, h - 1)
        cv2.line(out, (int(ground.split_x[y0]), y0), (int(ground.split_x[y1]), y1), (255, 255, 255), 2)

        margin = max(w // 8, 40)
        if stats["decision"] == TO_LEFT:
            cv2.arrowedLine(out, (w - margin, h // 2), (margin, h // 2), (0, 0, 255), 8, tipLength=0.3)
        elif stats["decision"] == TO_RIGHT:
            cv2.arrowedLine(out, (margin, h // 2), (w - margin, h // 2), (0, 0, 255), 8, tipLength=0.3)
        else:
            put_text(out, "HOLD", (w // 2 - 70, h // 2), 2, (0, 0, 255), 4)

    signal, _, color = LEVELS[stats["density"]]
    lines = [(f"People: {stats['people']:.0f}", 1, (255, 255, 255))]
    if stats["left_d"] is not None:
        lines += [(f"Avg: {stats['avg_density']:.2f}/m2  Peak: {stats['peak']:.2f}/m2", 0.8, (255, 255, 255))]
        if "exit" not in stats:
            lines += [(f"Left {stats['left_d']:.2f}/m2  Right {stats['right_d']:.2f}/m2", 0.8, (255, 255, 255))]
    elif "exit" not in stats:
        lines += [(f"Left: {stats['left']:.0f}  Right: {stats['right']:.0f}", 0.9, (255, 255, 255))]
    lines += [(f"Decision: {stats['decision']}", 1, (0, 255, 255)),
              (f"Density: {stats['density']}  Signal: {signal}", 1, color),
              (f"FPS: {fps:.1f}" + ("" if age is None else f"  Result age: {age:.1f}s"),
               0.8, (255, 255, 255))]
    for i, (text, scale, col) in enumerate(lines):
        put_text(out, text, (20, 40 + i * 40), scale, col)
    return out


def draw_flow(out, flow, detections):
    """Counting lines with their per-direction totals, and track ids at head points."""
    for ln in flow.lines.lines:
        a, b = tuple(np.int32(ln.a)), tuple(np.int32(ln.b))
        cv2.line(out, a, b, (255, 0, 255), 3)
        counts = flow.lines.counts[ln.name]
        text = f"{ln.name}: " + "  ".join(f"{d} {counts[d]}" for d in ln.directions)
        put_text(out, text, (min(a[0], b[0]), min(a[1], b[1]) - 10), 0.6, (255, 0, 255))
    for d in detections:
        if d.track_id is not None:
            x, y = head_point(d)
            put_text(out, f"#{d.track_id}", (int(x) - 10, int(y) - 6), 0.5, (255, 255, 0), 1)


def draw_waiting(frame, fps):
    """Before the first result: the frame with no stats, so nothing looks measured."""
    put_text(frame, "Waiting for first model result...", (20, 40), 0.9, (255, 255, 255))
    put_text(frame, f"FPS: {fps:.1f}", (20, 80), 0.8, (255, 255, 255))
    return frame


# ---------------- Frame sources ----------------

class RealtimeReader:
    """
    Plays a video file at its own frame rate, the way a camera delivers frames: when
    processing falls behind, frames whose time has passed are skipped, not queued.
    """

    def __init__(self, cap, fps, clock=time.perf_counter, sleep=time.sleep):
        self.cap = cap
        self.period = 1.0 / fps
        self.clock = clock
        self.sleep = sleep
        self.start = None
        self.index = -1    # file index of the last frame returned
        self.skipped = 0

    def read(self):
        now = self.clock()
        if self.start is None:
            self.start = now
        due = int((now - self.start) / self.period)  # newest frame whose time has come
        target = max(due, self.index + 1)
        while self.index + 1 < target:
            if not self.cap.grab():
                return False, None
            self.index += 1
            self.skipped += 1
        wait = self.start + target * self.period - self.clock()
        if wait > 0:
            self.sleep(wait)
        ok, frame = self.cap.read()
        if ok:
            self.index += 1
        return ok, frame


def change_alerts(prev, stats, calibrated):
    """(kind, old, new, detail) for each level or decision change since `prev` stats.
    The first stats of a run (prev None) record the starting level."""
    out = []
    old_level = None if prev is None else prev["density"]
    if stats["density"] != old_level:
        value = (f"peak {stats['peak']:.2f}/m2" if calibrated else f"people {stats['people']:.1f}")
        out.append(("level", old_level, stats["density"], value))
    if prev is not None and stats["decision"] != prev["decision"]:
        zones = ", ".join(f"{z['name']} {z['density']:.2f}" for z in stats.get("zones", []))
        out.append(("decision", prev["decision"], stats["decision"], zones))
    return out


def use_threads(mode, source):
    """--threaded auto: on for cameras, off for files so --output keeps every frame."""
    return mode == "on" or (mode == "auto" and source.isdigit())


def gap_summary(gaps):
    """Median, p99 and max of the gaps between shown frames, in ms."""
    if not gaps:
        return "no frame gaps (fewer than 2 frames)"
    g = np.asarray(gaps) * 1000
    return (f"frame gaps median {np.median(g):.0f} ms, p99 {np.percentile(g, 99):.0f} ms, "
            f"max {g.max():.0f} ms over {len(g)} gaps")


# ---------------- Main loop ----------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="camera index (0/1/2) or video file path")
    p.add_argument("--detector", choices=["yolo", "roboflow", "csrnet", "dmcount"], default="yolo")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLO weights file")
    p.add_argument("--density-weights", default="SHA",
                   help="csrnet: SHA/SHB, dmcount: SHA/SHB/QNRF (SHA/QNRF = dense crowds)")
    p.add_argument("--max-side", type=int, default=1000,
                   help="longest side fed to density networks (higher = slower, better for tiny heads)")
    p.add_argument("--backend", choices=["torch", "onnx", "openvino"], default="torch",
                   help="csrnet/dmcount: PyTorch, or the ONNX export (python benchmark_backends.py "
                        "export) in ONNX Runtime or OpenVINO. Same counts; speed-up varied with power "
                        "state on the test laptop (see README)")
    p.add_argument("--precision", choices=["f32", "bf16"], default="f32",
                   help="--backend openvino: bf16 is ~2x faster again but shifts counts (see README)")
    p.add_argument("--infer-every", type=int, default=1,
                   help="run the model every N frames and reuse the result in between")
    p.add_argument("--conf", type=float, default=CONF_TH)
    p.add_argument("--low", type=float, default=LOW_TH, help="uncalibrated: max people for GREEN")
    p.add_argument("--mid", type=float, default=MID_TH, help="uncalibrated: max people for YELLOW")
    p.add_argument("--calib", help="ground calibration JSON (enables people/m^2)")
    p.add_argument("--calibrate", metavar="JSON", help="click 4 head-height points on the first frame and save")
    p.add_argument("--calib-size", default="10x10", help="real size of the clicked rectangle, WxL metres")
    p.add_argument("--zones", metavar="JSON",
                   help="named zones and exits (head-height polygons); recommends the least dense exit")
    p.add_argument("--warn-density", type=float, default=WARN_DENSITY, help="calibrated: people/m^2 for YELLOW")
    p.add_argument("--crit-density", type=float, default=CRIT_DENSITY, help="calibrated: people/m^2 for RED")
    p.add_argument("--hysteresis", type=float, default=HYSTERESIS,
                   help="a level falls only once the value is this fraction below its threshold")
    p.add_argument("--min-hold", type=float, default=MIN_HOLD,
                   help="seconds a level is held before it may fall (rises are immediate)")
    p.add_argument("--log", metavar="DB", help="log readings, zones and alerts to this SQLite file")
    p.add_argument("--flow", action="store_true",
                   help="track people (YOLO + ByteTrack) to count crossings of the zones file's lines "
                        "and alert on stalled zones. Sparse to moderate scenes only (see README)")
    p.add_argument("--line-band", type=float, default=LINE_BAND,
                   help="px; a crossing must go from beyond this distance on one side of a line to "
                        "beyond it on the other, so a head point jittering on the line isn't counted")
    p.add_argument("--stall-speed", type=float, default=None,
                   help=f"median track speed below which a dense zone counts as stalled; default "
                        f"{STALL_SPEED} m/s calibrated, {STALL_SPEED_PX:g} px/s uncalibrated (a placeholder)")
    p.add_argument("--stall-seconds", type=float, default=STALL_SECONDS,
                   help="how long a dense zone must stay slow before a stall alert")
    p.add_argument("--stall-min-tracks", type=int, default=STALL_MIN_TRACKS,
                   help="tracks needed in a zone before its median speed is used")
    p.add_argument("--serial", help="Arduino serial port, e.g. COM7")
    p.add_argument("--output", help="write the annotated video here")
    p.add_argument("--heatmap-out", default="crowd_density_heatmap.png",
                   help="save the final heatmap image here")
    p.add_argument("--headless", action="store_true", help="don't open a window")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    p.add_argument("--max-seconds", type=float, default=0, help="stop after N seconds (0 = no limit)")
    p.add_argument("--threaded", choices=["auto", "on", "off"], default="auto",
                   help="run the model on a background thread on the newest frame, dropping older "
                        "ones; auto = on for cameras, off for files (so --output keeps every frame)")
    p.add_argument("--realtime", action="store_true",
                   help="play a video file at its own FPS, skipping frames when behind, like a camera")
    return p.parse_args(argv)


def open_source(source):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        hint = "try --source 0, 1 or 2" if source.isdigit() else "check the file path"
        sys.exit(f"Could not open source {source!r}: {hint}")
    return cap


def main(argv=None):
    args = parse_args(argv)
    if args.flow and not args.zones:
        sys.exit("--flow needs --zones: zones for stall alerts, and the file's lines for crossing counts")
    cap = open_source(args.source)
    if args.calibrate:
        run_calibration(cap, args.calibrate, args.calib_size)
        cap.release()
        return

    camera = args.source.isdigit()
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    reader = cap
    if args.realtime:
        if camera:
            print("--realtime only applies to video files; a camera already runs in real time")
        else:
            reader = RealtimeReader(cap, src_fps)
    threaded = use_threads(args.threaded, args.source)

    print(f"Loading {args.detector} detector...")
    detector = build_detector(args)
    print("Inference: " + ("background thread, newest frame only" if threaded
                           else "in the main loop"), flush=True)
    worker = InferenceWorker(detector) if threaded else None
    arduino = ArduinoSignal(args.serial)
    log = None
    if args.log:
        from crowd_log import CrowdLog
        label = args.detector + (f"/{args.backend}" if args.detector in DENSITY_MODELS else "")
        log = CrowdLog(args.log, label)
        log.alert("start", None, None, f"source={args.source} detector={label} "
                                       f"calib={args.calib or '-'} zones={args.zones or '-'}")
        print(f"Logging to {args.log}")
    prev_stats = None
    analyser = None
    flow = None
    flow_t = None         # media time of the frame the current result came from
    last_flow = None      # per-zone (speed, tracks) from the latest result
    writer = None
    dets, density = [], None
    stats = None          # None until the first result arrives
    result_at = None      # read time of the frame the current result came from
    last_seq = results = frames = 0
    peak_people = peak_density = 0.0
    gaps = deque(maxlen=GAP_HISTORY)
    started = prev = time.perf_counter()
    shown_at = None

    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            captured = time.perf_counter()
            frames += 1
            # flow speeds use the video's own clock for files (processing may be slower than
            # real time), and the capture clock for cameras
            index = reader.index if reader is not cap else frames - 1
            media_t = captured if camera else index / src_fps

            if analyser is None:
                ground = (Ground.load(args.calib, frame.shape) if args.calib
                          else Ground(frame.shape))
                zones = Zones.load(args.zones, ground) if args.zones else None
                if zones is not None:
                    print(zones.describe())
                analyser = CrowdAnalyser(ground, args, zones=zones)
                if args.flow:
                    flow = FlowMonitor(ground, zones, args)
            if worker:
                worker.submit(frame, captured, media_t)
                res = worker.poll()
                fresh = res is not None and res.seq != last_seq
                if fresh:
                    # one analyser update per model result, so smoothing averages real results
                    last_seq, results = res.seq, results + 1
                    dets, density, result_at = res.detections, res.density, res.captured_at
                    flow_t = res.media_t
                    stats = analyser.update(density)
            else:
                fresh = density is None or (frames - 1) % args.infer_every == 0
                if fresh:
                    dets, density = detector.analyse(frame)
                    result_at, results, flow_t = captured, results + 1, media_t
                stats = analyser.update(density)

            flow_alerts = []
            if flow is not None and stats is not None:
                if fresh:
                    crossings, flow_alerts = flow.update(dets, flow_t, stats["zones"])
                    last_flow = [(s["speed"], s["tracks"]) for s in stats["zones"]]
                    for line, direction, tid in crossings:
                        when = "" if camera else f" at {flow_t:.2f}s"
                        print(f"CROSS {line}: {direction} (track {tid}){when}", flush=True)
                        if log:
                            log.crossing(line, direction, tid)
                elif last_flow is not None:  # stats rebuilt from the same result: keep its speeds
                    for s, (speed, tracks) in zip(stats["zones"], last_flow):
                        s["speed"], s["tracks"] = speed, tracks

            now = time.perf_counter()
            fps = 1 / max(now - prev, 1e-6)
            prev = now
            if stats is None:
                age = None
                out = draw_waiting(frame, fps)
            else:
                age = now - result_at
                peak_people = max(peak_people, stats["people"])
                peak_density = max(peak_density, stats["peak"])
                arduino.send(LEVELS[stats["density"]][1])
                out = draw_overlay(frame, analyser.heat_colour(), dets, analyser.ground, stats,
                                   fps, age, analyser.zones)
                if flow is not None:
                    draw_flow(out, flow, dets)
                alerts = change_alerts(prev_stats, stats, analyser.ground.calibrated) + flow_alerts
                for kind, old, new, detail in alerts:
                    print(f"ALERT {kind}: {old} -> {new} ({detail})", flush=True)
                    if log:
                        log.alert(kind, old, new, detail)
                prev_stats = stats
                if log and fresh:
                    log.reading(stats, age)

            if args.output:
                if writer is None:
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps, (out.shape[1], out.shape[0]))
                writer.write(out)

            if frames % 25 == 0 or frames == 1:
                if stats is None:
                    print(f"frame {frames}: waiting for first result fps={fps:.1f}", flush=True)
                else:
                    if "zones" in stats:
                        extra = " zones[" + ", ".join(
                            f"{z['name']}={z['people']:.0f}p {z['density']:.2f}" for z in stats["zones"]) + f"]{stats['unit']}"
                    elif stats["left_d"] is not None:
                        extra = f" L={stats['left_d']:.2f} R={stats['right_d']:.2f}{stats['unit']}"
                    else:
                        extra = f" L={stats['left']:.0f} R={stats['right']:.0f}"
                    print(f"frame {frames}: people={stats['people']:.1f}{extra} "
                          f"peak={stats['peak']:.2f}{stats['unit']} level={stats['density']} "
                          f"decision={stats['decision']} fps={fps:.1f} age={age:.2f}s", flush=True)

            if not args.headless:
                cv2.imshow("Crowd Monitor", out)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
            t = time.perf_counter()
            if shown_at is not None:
                gaps.append(t - shown_at)
            shown_at = t
            if args.max_frames and frames >= args.max_frames:
                break
            if args.max_seconds and t - started >= args.max_seconds:
                break
    finally:
        if worker and not worker.close():
            print(f"Inference thread still busy after {WORKER_STOP_TIMEOUT}s; leaving it to exit")
        cap.release()
        if writer:
            writer.release()
        arduino.close()
        if log:
            log.close()
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    if stats is not None and args.heatmap_out:
        cv2.imwrite(args.heatmap_out, analyser.heat_colour())
        print(f"Saved heatmap: {args.heatmap_out}")
    unit = analyser.ground.unit if analyser else ""
    print(f"Processed {frames} frames, peak count {peak_people:.1f}, "
          f"peak local density {peak_density:.2f}{unit}")
    if flow is not None:
        for name, counts in flow.lines.counts.items():
            print(f"Crossings {name}: " + ", ".join(f"{d} {n}" for d, n in counts.items()))
    dropped = f", {worker.dropped} frames dropped by the worker" if worker else ""
    skipped = f", {reader.skipped} frames skipped by --realtime" if reader is not cap else ""
    print(f"Timing: {elapsed:.1f} s, {results} model results ({results / max(elapsed, 1e-6):.2f}/s), "
          f"{gap_summary(gaps)}{dropped}{skipped}", flush=True)


if __name__ == "__main__":
    main()
