# Kumbh Mela Crowd Management

Crowd monitoring from a camera or video. It counts people, maps crowd density, and classifies each zone as LOW, MEDIUM or HIGH. It suggests which exit to use, logs readings and alerts to SQLite, and shows them on a dashboard. On sparse scenes it can also track people across counting lines. Measured accuracy and known limits are in the sections below. Nothing here has been measured on labelled Kumbh footage.

## Quick start

```bash
pip install -r requirements.txt

python crowd_monitor.py --source 0                         # webcam, YOLOv8 (try 1 or 2 if 0 fails)
python crowd_monitor.py --source crowd.mp4 --detector dmcount --output out.mp4 --headless
python crowd_monitor.py --source 0 --serial COM7           # also drive Arduino G/Y/R LEDs
python crowd_monitor.py --detector roboflow --source 0     # needs ROBOFLOW_API_KEY env var
python evaluate.py --data C:\datasets\ShanghaiTech         # accuracy on labelled data (see below)
python benchmark_backends.py export                        # ONNX export, for --backend onnx / openvino
python crowd_monitor.py --source crowd.mp4 --detector dmcount --realtime --threaded on   # play a file like a camera
python crowd_monitor.py --source walk.mp4 --flow --zones zones_cctv_walk.json --log run.db   # line crossings, stalls
streamlit run dashboard.py -- --db run.db                  # dashboard over the log (reads it only)
python -m pytest tests                                     # unit tests
```

`crowd_monitor.py` turns each frame into a density map (people per pixel, summing to the head count),
then derives the heatmap, density level, left/right redirection arrow and Arduino signal from that map.
Press ESC to quit. The final heatmap is saved to `crowd_density_heatmap.png`.

### Choosing a detector

| `--detector` | Method | Use for |
|---|---|---|
| `yolo` (default) | Detects individuals (YOLOv8n) | Sparse scenes, webcam demos |
| `dmcount` | Density regression (DM-Count) | Dense crowds; recommended over CSRNet (below) |
| `csrnet` | Density regression (CSRNet) | Dense crowds only; over-predicts on sparse scenes |
| `roboflow` | Hosted detection model | Needs an API key |

`csrnet` and `dmcount` download weights on first use to `C:\.lwcc\weights` (65 MB and 86 MB). Choose the
training set with `--density-weights`: `SHA` (default) or `QNRF` for dense crowds, `SHB` for sparser ones.
`QNRF` is DM-Count only.

These networks are not real time on CPU. On this laptop, DM-Count's network call took 1.7-1.9 s per 720p
frame in PyTorch (see "Running live on a CPU" below). In PyTorch they use a GPU automatically if one is
available. Otherwise use `--threaded` to keep the video smooth, `--infer-every N` to reuse a result for N
frames, or lower `--max-side`, which runs faster but counted up to 2x lower on dense photos.

### Measuring density in people/m²

Head counts alone don't tell you how packed a place is. Calibrate once per camera position:

```bash
python crowd_monitor.py --source 0 --calibrate ghat.json --calib-size 10x6
python crowd_monitor.py --source 0 --detector dmcount --calib ghat.json
```

Mark out a rectangle of known size (here 10 m by 6 m), then click its corners in the order near-left,
near-right, far-right, far-left. **Click where a standing person's head appears at each corner, not on the
ground.** Density maps place people on their heads, so a region clicked on the ground misses people
standing near its far edge.

The rectangle becomes the monitored area. With a calibration, the level comes from the peak local density:
YELLOW from `--warn-density` (2/m²) and RED from `--crit-density` (4/m²). Redirection compares the average
density of the two halves and needs a 0.5/m² difference. Without a calibration, the level comes from the
head count (`--low`, `--mid`).

The default thresholds are starting points based on general crowd-safety guidance. Set them with your
site's safety planners before relying on them.

### Named zones and exits

Instead of a left/right split, you can name the areas a camera sees and the exits each one leads to:

```bash
python crowd_monitor.py --source 0 --detector dmcount --zones zones.json              # densities per 100x100 px
python crowd_monitor.py --source 0 --detector dmcount --calib ghat.json --zones zones.json   # people/m2
```

```json
{"frame_size": [1280, 720],
 "zones": [{"name": "Ghat A", "polygon": [[20, 120], [420, 120], [400, 700], [20, 700]]},
           {"name": "Ghat B", "polygon": [[440, 120], [840, 120], [860, 700], [420, 700]]}],
 "exits": [{"name": "Gate 1", "zone": "Ghat A"}, {"name": "Gate 2", "zone": "Ghat B"}]}
```

`zones_example.json` is a complete example for a 1280x720 frame.

**Drawing the zones:**
- Draw each polygon where standing people's heads appear, as with the calibration points. Density maps place
  people on their heads.
- Points are rescaled from `frame_size` if the video has a different resolution.
- Each exit names the zone a crowd must cross to reach it. Several exits may share a zone.

**What is measured:**
- Each zone gets a people count and a density: people/m² with a calibration, otherwise people per 100x100 px
  block. Pixels far from the camera cover more ground, so uncalibrated densities are only comparable
  between zones at a similar distance.
- With a calibration, only the part of a zone inside the calibrated rectangle is measured, and the
  monitor warns when part of a zone falls outside it.
- Zones may overlap. People in an overlap count in both.

**How the exit is chosen:**
- The monitor recommends the exit whose zone has the lowest density (`Decision: USE Gate 1`).
- It switches only when another exit is clearly better and at least 2 s have passed since the last switch.
- "Clearly better" uses the left/right margin: 0.5/m² with a calibration. Without one, it is max(2 people,
  15% of both zones' people), divided by the two zones' mean area. For two equal halves, that is exactly
  the left/right rule.
- On start-up the best exit is chosen straight away.
- A zones file without exits only adds per-zone stats, and the left/right decision stays.

**On screen and in the console:** zones are outlined, and the recommended exit's zone is green. Each zone
is labelled with its name, people count and density, and the console log lists them every 25 frames.
Without `--zones`, nothing changes.

**This has not been validated.** No site layout or labelled footage has been used. The example file is
three strips drawn on the test video, not a real ghat, and the 2 s cooldown and margins have not been
tuned on real crowd movement.

### Alerts and logging

**Levels don't flicker.** A level rises as soon as its threshold is passed, with the same boundaries as
before. It falls only when both of these hold:
- the value is at or below the threshold minus `--hysteresis`, 10% of the threshold by default;
- the level has been held for `--min-hold`, 5 s by default.

For example, with the 2/m² YELLOW threshold, YELLOW is entered above 2.0/m² and only falls back to GREEN at
or below 1.8/m², and no sooner than 5 s after it was entered. `--hysteresis 0 --min-hold 0` gives the old
behaviour. Every level or decision change is printed as an `ALERT` line.

**`--log run.db` writes a SQLite file** (Python's built-in `sqlite3`; a sample query follows the list):

- `readings(ts, people, peak, level, decision, result_age_s, detector)`: at most one per second, written
  only when a new model result has arrived.
- `zone_readings(ts, zone, people, density)`: one row per zone for each reading, with `--zones`.
- `alerts(ts, kind, old_level, new_level, detail)`:
  - `start`, once per run, with the source, detector, calibration and zones in `detail`.
  - `level`, for each level change, including the starting level, with the peak density or people count in
    `detail`.
  - `decision`, for each change of redirect or exit, with the zone densities in `detail`.

  Alerts are written when they happen. Hysteresis and the 2 s decision cooldown already limit how often
  that can be.

`ts` is Unix time in seconds. Runs append to the same file, and each starts with a `start` row. The file
uses SQLite's WAL mode, so a dashboard can read it while the monitor writes.

```bash
python crowd_monitor.py --source 0 --detector dmcount --zones zones.json --log run.db
sqlite3 run.db "SELECT datetime(ts, 'unixepoch', 'localtime'), kind, old_level, new_level, detail FROM alerts"
```

**Tested on the test video, not on a real site.** Two runs went into one database: 60 s of YOLO with
zones (52 readings, 4 level changes, 2 exit switches), then 30 s of DM-Count (17 readings, HIGH
throughout). In the YOLO run, both falls came exactly 5 s after the rise before them, which shows the
minimum hold at work. The unit tests show the flicker suppression: a value hovering around 2.0/m² for 40 s
changes level once, against more than 30 times without hysteresis. The 10% margin and 5 s hold are
defaults, not values tuned on real crowds.

### Flow: line crossings and stalls (`--flow`)

**Sparse to moderate scenes only** (see "Where flow fails" below). `--flow` runs YOLO with ByteTrack
(through Ultralytics), so it needs `--detector yolo` and a `--zones` file.

```bash
python crowd_monitor.py --source walk.mp4 --flow --zones zones_cctv_walk.json --log run.db --output flow.mp4
```

**Crossings.** Add named lines to the zones file, drawn at head height like the zones:

```json
"lines": [{"name": "Pavement line", "points": [[480, 250], [1080, 250]],
           "directions": ["towards camera", "away from camera"]}]
```

- The first direction is a crossing from the left of the line to its right, looking from the first point
  to the second. For a line drawn left to right, that means moving down the image. Without
  `directions`, they're called `forward` and `backward`.
- Each direction is counted separately, and each track at most once per direction per line.
- A track's position is its head point, the same point the density map uses.
- A crossing must go from more than `--line-band` (15 px) on one side to more than 15 px on the other. A
  head point jittering on the line therefore isn't counted.
- Each crossing prints a `CROSS` line, and the totals are printed at the end.

**Stall alert.** A zone is stalled when two things have held for `--stall-seconds` (10 s):
- it is at or above the warning level;
- the median speed of the tracks in it is below `--stall-speed`.

The details:
- **Warning level:** 2/m² when calibrated. Uncalibrated, it's more than `--low` people in the zone.
- **Track speed:** its displacement over the last 1 s, so a box jittering in place reads as standing still.
- **Default `--stall-speed`:** 0.2 m/s when calibrated. **Uncalibrated, it's a placeholder: 15 px/s.** That
  equals 0.2 m/s only if a 1.7 m person is about 130 px tall, so set it for your camera.
- **Too few tracks:** a zone with fewer than `--stall-min-tracks` (3) tracks with a speed isn't judged.
- **Alerts:** a stall is logged as an alert of kind `stall`, from `flowing` to `stalled`. Its end is logged
  as `stalled` to `flowing`.
- **Clock:** for video files, speeds and timers use the video's own clock, so slow processing doesn't make
  people look slow.

**`--log` adds a `crossings(ts, line, direction, track_id)` table.**

**`lap` is optional.** Ultralytics' ByteTrack imports `lap`, and if that import fails it tries to
pip-install it. On the test laptop, `pip install lap` worked, but Windows Smart App Control blocks lap's
unsigned DLL, so it can never load there. When `lap` is missing or blocked, `--flow` does two things:
- it registers a stand-in that uses Ultralytics' own assignment solver;
- it switches Ultralytics' auto-install off.

The run prints which solver it used. On the first 400 frames of the test clip, the stand-in and an exact
`lapjv` equivalent produced 58 and 59 track IDs.

**Tested on one real clip, against a hand count.** The clip is street CCTV of a rainy pavement with about
10 people in view: "The CCTV People Demo 2" (YouTube `GJNjaRJWVP8`). I used its 63 s fixed-camera shot
(frames 104–1675 of the 720p stream). It isn't in the repo. I counted heads crossing the line by eye in the
first 20 s, from crops taken every 0.4 s:

| Version (same clip, first 20 s) | Towards camera | Away from camera |
|---|---:|---:|
| Hand count | 6 (2 uncertain) | 3 |
| Line at y=360 (only the 63 s totals were logged) | 8 in 63 s | 1 in 63 s |
| Line at y=250, no band | 10 | 5 |
| Line at y=250, band 15 px, line stopped at x=1000 | 2 | 3 |
| **Line at y=250, band 15 px (default)** | **4** | **3** |

What the table shows:
- **The line has to be where heads actually pass.** At y=360, people only reached it as they left the frame.
- **Without the band, crossings were double-counted.** Several people standing at the shop front carried 5
  overlapping track IDs, and their heads sat on the line. Some tracks were also counted both ways within
  one frame.
- **With the band,** the "away" count matches the hand count, with times within about 1 s. Of the 6
  "towards" crossings, it found 4. It missed 2 people who were already at the line when the clip started,
  before tracking had begun.
- **This isn't an accuracy figure.** I chose the line and the band on these same 20 s, it's one clip, and
  the hand count is mine.

**The full 63 s logged 8 crossings towards the camera and 10 away**, plus 201 readings and 402 zone rows.
Only the first 20 s were hand-checked. In the rest, track 204 was counted towards at 20.6 s and away at
21.1 s, which I didn't check by eye.

**It isn't real time on a CPU.** YOLOv8n with tracking ran at 7.05 frames/s on 720p, with every frame
tracked. With `--realtime` or a camera it would skip frames, and ByteTrack loses tracks more easily when it
does. That hasn't been measured.

**No stall happened in this clip, so the stall alert is tested only on synthetic tracks** (unit tests: a
dense crowd standing still alerts at 10 s; moving, sparse and too-few-track zones don't).

#### Where flow fails

- **Tracking needs people YOLO can see, and in dense crowds it can't.** On ShanghaiTech Part A, YOLOv8n
  found 0.7% of the labelled people, never more than 13 per image, and exactly 0 on 74 of 182 images (see
  "Measured accuracy").
- **It already breaks down at moderate density.** On the 10-person test clip, 400 frames (16 s) produced
  58 track IDs, of which only 25 lasted 1 s or more. Umbrellas, people overlapping and clothing racks split
  tracks and duplicated boxes.
- **In a dense crowd the stall alert goes silent, not red.**
  - With fewer than 3 speeds in a zone, it can't judge the zone at all.
  - Where some people are still detected, they are the visible ones at the edge, and their speed says
    little about the crowd inside.
  - **No stall alert is not evidence that a crowd is moving.** At Kumbh densities, use the density levels
    from `--detector dmcount`, not `--flow`.
- **Counts depend on the line's placement and on the camera.** People crossing at the start of a video, or
  out of view, aren't counted.

**Labelled data needed to measure accuracy:**
- **For crossings:** fixed-camera video from the deployment cameras, at the densities you care about.
  Either label every head with a persistent ID (MOT-style tracks, as in the CroHD head-tracking dataset), or
  at least hand-count crossings per line and direction with timestamps. Several hours, spanning sparse to
  dense, so error can be reported per density band.
- **For stalls:** the same kind of video, with calibrated cameras for m/s, and the start and end times of
  real stop-and-go episodes marked by observers.

### Dashboard (`dashboard.py`)

A Streamlit page that reads the `--log` database and nothing else. It opens the file read-only and never
opens a camera or video, so it can run beside the monitor or on another machine with a copy of the log.

```bash
python crowd_monitor.py --source 0 --zones zones.json --log run.db      # terminal 1: writes the log
streamlit run dashboard.py -- --db run.db                              # terminal 2: http://localhost:8501
```

It shows:
- **Latest reading:** level and decision, people in view, peak density, the result's age, and each zone's
  people and density. A "live" badge appears while readings are under 10 s old.
- **Trends:** level over time, people in view, and people and density per zone.
- **Line crossings** (`--flow` runs): totals per line and direction, and cumulative counts.
- **Alert history:** newest first.

**How it works:**
- It refreshes itself every 2 s (`--refresh`, or `CROWD_LOG_REFRESH`; 0 turns it off).
- Charts show the last 5 minutes before the latest reading, or the whole run. Because the window ends at
  the latest reading, a finished run still shows its data.
- A log can hold several runs: each run of the monitor writes a `start` alert, and the run picker splits
  the log there. "Latest run" follows new runs as they start.
- If the log doesn't exist yet, the page says so and retries.
- Times are in the dashboard machine's local time.
- Zone levels aren't logged, only each zone's people and density, so the level shown is the overall one.

#### Hosted demo

[`app/streamlit_app.py`](app/streamlit_app.py) runs the same dashboard in demo mode, for hosting.
Instead of a file path on the server, visitors pick one of the recorded runs in `eval_results/` or upload
a log of their own (`crowd_monitor.py --log run.db`).
- **Short install:** [`app/requirements.txt`](app/requirements.txt) lists only Streamlit, Altair and
  pandas, so the host doesn't install PyTorch or OpenCV.
- **Nothing runs a model:** no camera or video is opened, and the recorded runs are labelled as test
  videos, not live or Kumbh data.
- **Uploads:** an uploaded file is saved in the server's temp folder under a hash of its contents. It
  stays there until the app restarts, and anyone who uploads an identical file gets the same copy. Don't
  upload logs you wouldn't share.

**Deploy on Streamlit Community Cloud (free):**
1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Choose **Create app**, then deploy from GitHub with:
   - repository `parthverma83/Kumbh-Mela-Crowd-Management-Project`;
   - branch `main`;
   - main file `app/streamlit_app.py`.
3. Choose a subdomain, for example `kumbh-crowd-monitor`, which gives the address
   `https://kumbh-crowd-monitor.streamlit.app`.

**Try it locally:** `streamlit run app/streamlit_app.py`.

**Checked:**
- **Tests:** `tests/test_dashboard.py` (19 tests) covers the query functions on logs written by
  `CrowdLog`, plus the whole page through Streamlit's `AppTest`.
- **Real logs:** the page rendered without errors on both real logs, `eval_results/task4_run.db` (two runs)
  and `eval_results/task5_run.db`. It showed the same crossing totals as the monitor's own summary
  (8 towards, 10 away), and its time charts show times as HH:MM:SS.
- **Live:** `crowd_monitor.py --flow --realtime` wrote `eval_results/task6_live.db` for 60 s while a
  headless browser watched the open page. In 30 s it saw 10 refreshes, each with a new people count, and
  no reloads. At the end, the dashboard and the monitor's summary agreed: 3 towards, 6 away.
- **Opened before the log existed:** the page retried, and showed the first reading 10.4 s after the
  monitor started, without a reload; most of that was the monitor loading its model. When the monitor
  exited, its WAL file was empty, meaning all the data was in the database file. The dashboard opens the
  log for each refresh and closes it again: in the first live test it held one connection open, and the
  log was left in its `-wal` file.

### Measured counts

Six photos: five real Kumbh Mela crowds from Wikimedia Commons and one street scene with about 4-5 people.
Figures are at `--max-side 1000` with `SHA` weights.

| Photo | YOLOv8n | CSRNet | DM-Count |
|---|---:|---:|---:|
| Kumbh 1 | 5 | 289 | 168 |
| Kumbh 2 | 7 | 210 | 239 |
| Kumbh 3 | 11 | 204 | 314 |
| Kumbh 4 | 5 | 252 | 232 |
| Kumbh 5 | 11 | 75 | 57 |
| Street scene | 3 | 116 | 3 |

This shows YOLO missing most of a dense crowd, and CSRNet over-predicting on the sparse scene. **It does
not show accuracy.** The Kumbh photos have no ground-truth counts, and the two networks differ by up to
about 1.7x on them. Heads far from the camera are also very small, so these counts may be low. The next
section measures accuracy on labelled images, but not on Kumbh footage.

### Measured accuracy (ShanghaiTech Part A)

`evaluate.py` scores the detectors on the 182 labelled images of the Part A test set (78,970 heads, 66 to
2,256 per image), using `SHA` weights. Lower is better. Bias is the average signed error, so a negative
value means under-counting.

| Detector | Input size | MAE (95% interval) | RMSE | Bias | Median s/image (CPU) |
|---|---|---:|---:|---:|---:|
| DM-Count | native | **61.6** (50.4-73.8) | **98.8** | -4.6 | 1.40 |
| DM-Count | `--max-side 1000` | 67.3 (54.9-81.7) | 114.0 | -14.7 | 1.35 |
| CSRNet | native | 74.3 (62.4-87.3) | 111.8 | +6.1 | 1.42 |
| CSRNet | `--max-side 1000` | 76.5 (62.7-91.9) | 124.2 | -5.4 | 1.61 |
| YOLOv8n | conf 0.40 | 431.0 (380.9-487.9) | 558.1 | -431.0 | 0.04 |

`--max-side 1000` is what `crowd_monitor.py` runs by default. Through OpenVINO (`--backend openvino`), all
four density configs give the same MAE as PyTorch to 0.01 (see "Running live on a CPU"). The largest Part A images are 1024 px, so
"native" is only slightly larger; larger frames have not been measured.

**How these compare with the papers.** The DM-Count paper reports 59.7 / 95.7 (MAE / RMSE) for DM-Count and
68.2 / 115.0 for CSRNet on this test set. Both published MAEs are inside the intervals above, so these
weights appear to be working as intended. Published figures use the plain network sum, which gives CSRNet
73.9 / 112.2 here. `crowd_monitor.py` clips negative values, which adds about 4 people per image at native
size and barely changes the MAE.

**What the results show:**

- **DM-Count is more accurate than CSRNet:** it has a lower MAE by 12.7 at native size (paired 95% interval
  5.4 to 20.1) and by 9.2 at `--max-side 1000` (1.6 to 16.9).
- **Native input helps DM-Count:** MAE drops by 5.7 (0.6 to 11.5). CSRNet shows no clear difference.
- **YOLO can't count dense crowds:** it found 0.7% of the labelled people, never more than 13 in an image,
  and exactly 0 on 74 of 182 images. Lowering `--conf` doesn't fix this: at 0.05 it still found only 6.1%
  (MAE 407.4). Part A has no sparse scenes, so this says nothing about YOLO on sparse scenes.
- **The largest crowds are under-counted, which is the unsafe direction:** the near-zero overall bias hides
  this. On the 19 images with over 1,000 people, DM-Count averages -130.8 at `--max-side 1000` and -78.1 at
  native size, with an MAE of 234.1 and 189.4.
- **Small crowds can be badly over-counted:** on the sparsest image (66 people), CSRNet gave 315 at
  `--max-side 1000` and DM-Count gave 125, or 168 and 100 at native size.

**Limits:**

- **This is not Kumbh accuracy.** These weights were trained on the Part A training images, so this test
  is close to a best case. Kumbh footage may differ in camera angle, distance, lighting and clothing, and
  no Kumbh footage has been scored.
- These are head counts per image. People/m² and the 2/m² and 4/m² thresholds are still unvalidated.
- Speeds are from one CPU run on images of at most 1024 px.

To reproduce, get the dataset from the [authors' links](https://github.com/desenzhou/ShanghaiTechDataset),
unzip it, and run the command below. Per-image results are in `eval_results/shanghaitech_A_test.csv`, and
the script resumes from that file, so use a new `--out` to start again.

```bash
python evaluate.py --data C:\datasets\ShanghaiTech
```

### Running live on a CPU

Inference runs on a background thread for cameras, so the video keeps playing while the model works. This
doesn't make results arrive faster, and doesn't make counts more accurate. The density networks can also run
through ONNX Runtime or OpenVINO with identical counts. On this laptop that was about 2x faster on battery,
but only about 1.2x plugged in, so PyTorch stays the default.

```bash
python crowd_monitor.py --source 0 --detector dmcount                   # camera: threaded by default
python crowd_monitor.py --source crowd.mp4 --detector dmcount --realtime --threaded on
python benchmark_backends.py export        # once, for the options below: writes exported_models/*.onnx
python crowd_monitor.py --source 0 --detector dmcount --backend openvino
```

**Backends.** `--backend torch` is the default. `--backend onnx|openvino` runs the exported model and needs
the export step above. OpenVINO runs in FP32. On this CPU it would otherwise silently pick bf16, which
changes the counts. `--precision bf16` opts in.

The table below comes from one AMD Ryzen AI 7 350 laptop **running on battery** (Balanced power plan). On
battery, PyTorch was about 1.5x slower than in the accuracy run above (2.01 s against 1.35 s median on Part A
at `--max-side 1000`), so compare speed-ups, not seconds. The plugged-in results that follow it changed the
decision.

- Every backend used 8 threads.
- The inputs were 20 random Part A test images at `--max-side 1000` (14 different sizes, at most 1024 px),
  and the same images stretched to 1280x720, which the monitor feeds to the network at 992x560.
- Only the network call is timed, because preprocessing is identical for every backend.
- Each image got one discarded warm-up run per backend, then one timed run, with the backend order rotated
  per image. Everything was repeated in a second pass, giving 40 timed runs per cell.
- The speed-up is the median of the per-image ratios (PyTorch time / backend time). Its 95% interval comes
  from a bootstrap that resamples whole images.

| Model | Input | Backend | Median s | p95 s | Speed-up (95% interval) | Largest count difference from PyTorch |
|---|---|---|---:|---:|---:|---:|
| DM-Count | Part A | PyTorch | 2.01 | 2.27 | 1.00 | - |
| DM-Count | Part A | ONNX Runtime | 1.59 | 1.94 | 1.26 (1.23-1.27) | 0.0001 people |
| DM-Count | Part A | **OpenVINO** | **1.04** | 1.27 | **1.91 (1.84-1.94)** | 0.0001 people |
| DM-Count | Part A | OpenVINO bf16 | 0.54 | 0.63 | 3.70 (3.63-3.78) | 4.3 people (0.40%) |
| DM-Count | 720p (992x560) | PyTorch | 1.78 | 2.04 | 1.00 | - |
| DM-Count | 720p (992x560) | ONNX Runtime | 1.42 | 1.56 | 1.26 (1.25-1.28) | 0.0001 people |
| DM-Count | 720p (992x560) | **OpenVINO** | **0.90** | 0.98 | **1.96 (1.94-2.02)** | 0.0001 people |
| DM-Count | 720p (992x560) | OpenVINO, static shape | 0.89 | 0.98 | 1.98 (1.93-2.00) | 0.0001 people |
| DM-Count | 720p (992x560) | OpenVINO bf16 | 0.48 | 0.52 | 3.78 (3.70-3.86) | 3.5 people (0.37%) |
| CSRNet | Part A | PyTorch | 2.03 | 2.47 | 1.00 | - |
| CSRNet | Part A | ONNX Runtime | 1.64 | 1.97 | 1.23 (1.22-1.25) | 0.0002 people |
| CSRNet | Part A | **OpenVINO** | **1.03** | 1.17 | **1.97 (1.91-2.04)** | 0.0002 people |
| CSRNet | Part A | OpenVINO bf16 | 0.53 | 0.61 | 3.85 (3.79-3.92) | 2.7 people (1.31%) |
| CSRNet | 720p (992x560) | PyTorch | 1.75 | 1.93 | 1.00 | - |
| CSRNet | 720p (992x560) | ONNX Runtime | 1.42 | 1.58 | 1.22 (1.20-1.24) | 0.0002 people |
| CSRNet | 720p (992x560) | **OpenVINO** | **0.89** | 1.16 | **1.95 (1.91-1.97)** | 0.0002 people |
| CSRNet | 720p (992x560) | OpenVINO, static shape | 0.89 | 1.00 | 1.97 (1.94-2.01) | 0.0001 people |
| CSRNet | 720p (992x560) | OpenVINO bf16 | 0.46 | 0.52 | 3.77 (3.72-3.88) | 1.8 people (1.15%) |

The rule for making a backend the default was a median speed-up of at least 1.2x on both input sets, with
every count within 0.5% of PyTorch's.
- On battery, OpenVINO FP32 passed for both models, and ONNX Runtime passed more narrowly.
- A static input shape gave no measurable gain, and only works when every frame is the same size. Part A
  images vary in size, so it was not run on them.
- bf16 fails for CSRNet (1.3%).

**Plugged in, the speed-up mostly disappeared.** The benchmark was repeated on AC power, DM-Count only. Each
run was stopped part-way, so these figures cover the first pass over the 20 Part A images (20 runs per
backend):

| Condition | PyTorch median | ONNX Runtime | OpenVINO FP32 | OpenVINO bf16 |
|---|---:|---:|---:|---:|
| Battery (table above, 40 runs) | 2.01 s | 1.26x (1.23-1.27) | 1.91x (1.84-1.94) | 3.70x (3.63-3.78) |
| Plugged in, charging, after about an hour of load | 1.90 s | 1.23x (1.18-1.27) | 1.03x (0.98-1.10) | 2.12x (1.91-2.36) |
| Plugged in, fully charged, after 45 min idle | 1.84 s | 1.19x (1.18-1.22) | 1.18x (1.15-1.31) | 2.41x (2.32-2.53) |

- PyTorch speed barely changed. OpenVINO's did, and varied from 1.2 to 3.0 s on one image.
- Thread count (4, 8 or 16) and CPU pinning did not explain it, and neither did charging. The cause is
  unknown. Windows' power mode, which can differ between battery and AC, is untested.
- Since no backend reliably clears 1.2x, **PyTorch stays the default**. ONNX Runtime and OpenVINO give the
  same counts, so they are safe to try with `--backend` on a given machine.
- The runs are in `eval_results/backends.csv` (battery), `backends_ac.csv` and `backends_ac2.csv`, with power
  state in `eval_results/power_log.csv`. In `backends_ac2.csv`, rows after 13:39:55 were recorded after the
  laptop was unplugged.

OpenVINO was also run on all 182 Part A test images, as a check that it counts the same:

- **FP32:** each of the four density configs (DM-Count and CSRNet, at `--max-side 1000` and native size)
  gives the same MAE as PyTorch to 0.01. The paired difference is 0.000, and no image's count differs by
  more than 0.0002.
- **DM-Count bf16:** the MAE is 67.2 at `--max-side 1000` (paired difference -0.15, 95% interval -0.33 to
  +0.03) and 61.5 at native size (-0.10, -0.29 to +0.09). Its largest per-image difference is 0.57%, and one
  image is over 0.5%. That is why bf16 is opt-in and not the default.

**Threaded inference.** `--threaded auto` (the default) runs the model on a background thread for cameras
and in the main loop for files, so `--output` keeps every frame of a file.
- The thread works on a copy of the newest frame and drops older ones, so the display never falls behind
  the camera.
- The screen and the console show the result's age: how long ago the frame it came from was read.
- Until the first result arrives, frames are shown without stats and nothing is sent to the Arduino.
- `--realtime` plays a file at its own frame rate and skips frames when the program falls behind, so the
  threaded path can be tested without a webcam.

Test: a 60 s, 25 fps, 1280x720 video made from 20 other Part A test images (3 s each), built with
`python benchmark_backends.py video`. DM-Count, OpenVINO FP32, `--realtime --headless --max-seconds 60`, run 4
times in the order threaded, unthreaded, unthreaded, threaded, on battery:

| Run | Mode | Frames shown | Model results | Results/s | Frame gap median | p99 | max | Result age |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | threaded | 1,497 | 47 | 0.78 | 40 ms | 69 ms | 97 ms | 1.30-2.44 s, median 1.88 (58 samples) |
| 2 | unthreaded | 49 | 49 | 0.82 | 1,219 ms | 1,282 ms | 1,300 ms | 1.21-1.27 s (2 samples) |
| 3 | unthreaded | 49 | 49 | 0.82 | 1,219 ms | 1,243 ms | 1,248 ms | 1.20-1.35 s (2 samples) |
| 4 | threaded | 1,500 | 48 | 0.79 | 40 ms | 70 ms | 83 ms | 1.31-2.44 s, median 1.84 (59 samples) |

Result ages are sampled from the console log, which prints every 25 frames.

**What this shows:**

- **Threading keeps the video smooth.** Frames came 40 ms apart, matching the file's 25 fps, against more
  than 1.2 s apart in the main loop.
- **Threading does not make results faster or fresher.** Results per second fell slightly, from 0.82 to
  0.78, because drawing 25 frames a second competes for the same CPU cores.
- **Threaded results are older when you look at them.** A result is about 1.3 s old when it arrives and
  2.4 s old when the next one replaces it. In the main loop the screen freezes for about 1.2 s between
  results, so what is on screen ages over a similar range.
- **The model is most of the cost.** At 720p, OpenVINO took about 1.0 s of each 1.2 s main-loop frame.
  Preprocessing took 35 ms and analysis plus drawing about 30 ms.

**Limits:**

- These are timings from one laptop on battery, not a deployment figure. A plugged-in or different machine
  will differ.
- The live runs used `--backend openvino`, the default at the time. With PyTorch, the default now, each
  result takes longer, so results per second are lower. The threaded frame gaps should be similar, but that
  has not been measured.
- A real camera has not been tested. Cameras buffer a few frames, so in the main loop a real camera would
  show frames older than `--realtime` does. The threaded path avoids that by reading every frame.
- On start-up, the left/right decision holds HOLD for the 2 s cooldown. In the threaded runs, the first
  result arrived before that, so the first redirect appeared with the second result.

To reproduce:

```bash
python benchmark_backends.py export --data C:\datasets\ShanghaiTech   # export, and check against PyTorch
python benchmark_backends.py bench --data C:\datasets\ShanghaiTech    # resumable: eval_results/backends.csv
python benchmark_backends.py report
python evaluate.py --data C:\datasets\ShanghaiTech --models dmcount csrnet --backend openvino
python benchmark_backends.py video --data C:\datasets\ShanghaiTech    # test_media/partA_720p.mp4
python crowd_monitor.py --source test_media/partA_720p.mp4 --detector dmcount --realtime --threaded on --headless --max-seconds 60
```

Background & Motivation

Our initial implementation focused on deploying a YOLO-based object detection model on an IoT surveillance setup to count people in real-world environments.

While YOLO performs well in sparse to moderately crowded scenes, it showed critical limitations when applied to extremely dense crowds, such as religious gatherings, festivals, and mass events.

This project documents our engineering transition from traditional object detection to density map regression using CSRNet, and highlights the improvements achieved.

🧪 Phase 1: YOLO-based Crowd Counting on IoT
🔧 Approach

Model: YOLO (bounding-box based object detection)

Deployment: IoT camera / edge device

Output: Person count via detected bounding boxes

❌ Limitations Observed

When tested on high-density environments:

Severe occlusion between people

Extremely small visible head regions

Bounding boxes overlapped or disappeared

False negatives increased rapidly

Count accuracy degraded beyond usable limits

📉 Result:
YOLO consistently underestimated the crowd size and failed in scenarios with 500+ people per frame.

🔍 Key Insight

In extreme crowd density, individual detection is not feasible.
The problem must be reframed from “detecting people” to “estimating crowd density”.

This insight led us to explore Density Map Regression, which is specifically designed for high-occlusion scenarios.

🧠 Phase 2: Transition to CSRNet (Density Map Regression)
🔧 New Approach

Model: CSRNet (Convolutional Neural Network for Crowd Counting)

Core Idea:

Each person contributes to a continuous density map

No bounding boxes required

Output:

Density heatmap

Total crowd count = sum of density values

🚀 Why CSRNet Works Better Than YOLO
Aspect	YOLO (Object Detection)	CSRNet (Density Map)
Handles occlusion	❌ Poor	✅ Better (not measured separately)
Works in dense crowds	❌ Fails	✅ Designed for it
Bounding boxes needed	✅ Yes	❌ No
MAE on Part A images with >1,000 people	❌ 1,279.4	✅ 234.5 (19 images, native size)
Suitable for Kumbh Mela-like scenes	❌ Finds few people, no labelled Kumbh data	Not measured: no labelled Kumbh data
📊 Improvements

📈 Measured on ShanghaiTech Part A: MAE fell from 431.0 (YOLOv8n) to 74.3 (CSRNet) and 61.6 (DM-Count) at native size. See "Measured accuracy" above.

🌡️ Density heatmap visualization (reliability not measured)

🧩 Region-wise crowd density estimation (accuracy not measured)

🧠 Performance under extreme occlusion (not measured separately; the largest Part A crowds are under-counted on average)

🔁 Scaling to real-world surveillance feeds (not tested)

🏗️ System Architecture Evolution
YOLO-Based IoT Pipeline (Initial)
Camera → YOLO Detection → Bounding Boxes → Count

CSRNet-Based Pipeline (Improved)
Camera → Frame Extraction → CSRNet → Density Map → Count + Heatmap

📂 Datasets Used

To align with real-world dense crowd scenarios, we relied on datasets specifically designed for extreme density:

UCF-QNRF (primary inspiration)

ShanghaiTech Part A

NWPU-Crowd (optional benchmarking)

Only ShanghaiTech Part A has been evaluated in this repository. UCF-QNRF and NWPU-Crowd have not.

🎯 Real-World Impact

These are the project's goals. None has been measured in a real deployment:

Accurate crowd estimation in mass gatherings

Early identification of high-risk dense zones

Improved decision-making for crowd management and safety

Feasibility for smart city and public safety IoT deployments

🔮 Future Work

Optimize density networks for edge/IoT inference (started: ONNX and OpenVINO CPU backends, see "Running live on a CPU")

Real-time alerts for critical density thresholds (done for one camera: see "Alerts and logging" and "Dashboard")

Multi-camera crowd fusion

Edge–cloud hybrid deployment

📝 Conclusion

By moving from object detection (YOLO) to density map regression, this project reframes the problem from detecting people to estimating crowd density. On labelled ShanghaiTech Part A images, density networks count dense crowds far more accurately than YOLOv8n.

It has not been measured on Kumbh footage or in a live deployment, and the largest crowds are under-counted on average.
