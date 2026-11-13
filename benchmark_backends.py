"""
Speed of the density networks on CPU inference backends, against PyTorch.

  export - write exported_models/<model>_<weights>.onnx (dynamic height and width) and check
           it against PyTorch on real images
  bench  - time each backend on 20 Part A test images at --max-side 1000, and on the same
           images resized to 1280x720 (what a 720p camera feeds the monitor)
  report - the backend table from the bench CSV
  video  - a 1280x720 test video made from 20 other Part A test images, for
           crowd_monitor.py --realtime

Only the network call is timed: preprocessing is the same code for every backend. Each
image gets one discarded warm-up run per backend (a new input shape costs extra on its first
run), then one timed run, with the backend order rotated per image. Results go to a CSV
after every run, so an interrupted bench resumes where it stopped.

Examples:
    python benchmark_backends.py export
    python benchmark_backends.py bench --data C:\\datasets\\ShanghaiTech
    python benchmark_backends.py report
    python benchmark_backends.py video --data C:\\datasets\\ShanghaiTech
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

from crowd_monitor import (DENSITY_MODELS, MODEL_DIR, DensityDetector, exported_model_path,
                           onnx_runner, openvino_runner)
from evaluate import find_samples

# ---------------- DEFAULT SETTINGS ----------------
THREADS = 8                 # same for every backend; torch's default on this laptop
BENCH_CSV = os.path.join("eval_results", "backends.csv")
BENCH_FIELDS = ["model", "set", "image", "pass", "backend", "input_h", "input_w",
                "seconds", "count", "raw_count", "wall"]
BACKENDS = ["torch", "onnx", "openvino", "openvino-static", "openvino-bf16"]
N_IMAGES = 20
SAMPLE_SEED = 0
CAMERA_SIZE = (1280, 720)   # w, h of the "camera" set and the test video
PASSES = 2
VIDEO_FPS = 25
VIDEO_HOLD = 3.0            # seconds each image stays on screen in the test video
VIDEO_OUT = os.path.join("test_media", "partA_720p.mp4")

ADOPT_SPEEDUP = 1.2         # median speed-up a backend needs over torch...
ADOPT_COUNT_DIFF = 0.005    # ...with every count within this fraction of torch's
BOOTSTRAP_ROUNDS = 2000
STALL_GAP = 60              # s between consecutive results that counts as a stall
# ---------------------------------------------------


onnx_path = exported_model_path


def load_net(model, weights):
    import torch
    from lwcc import LWCC
    torch.set_num_threads(THREADS)
    return LWCC.load_model(DENSITY_MODELS[model], weights).eval()


# ---------------- Samples ----------------

def split_samples(samples, n=N_IMAGES, seed=SAMPLE_SEED):
    """Bench images and video images: two disjoint random sets, each in file order."""
    order = np.random.default_rng(seed).permutation(len(samples))
    bench, video = sorted(order[:n]), sorted(order[n:2 * n])
    return [samples[i][0] for i in bench], [samples[i][0] for i in video]


def image_sets(paths):
    """{set name: [(image name, BGR frame)]}: as stored, and stretched to the camera size."""
    frames = [(os.path.basename(p), cv2.imread(p)) for p in paths]
    if any(f is None for _, f in frames):
        sys.exit("Could not read a bench image")
    camera = [(n, cv2.resize(f, CAMERA_SIZE, interpolation=cv2.INTER_AREA)) for n, f in frames]
    return {"partA": frames, "camera720": camera}


# ---------------- Export ----------------

def export_onnx(model, weights, path):
    import torch
    net = load_net(model, weights)
    h, w = torch.export.Dim("h", min=16, max=4096), torch.export.Dim("w", min=16, max=4096)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(net, (torch.zeros(1, 3, 560, 1008),), path, input_names=["image"],
                      output_names=["density"], dynamic_shapes={"x": {2: h, 3: w}},
                      dynamo=True, external_data=False, verbose=False)
    return net


def check_export(net, path, frames, max_side):
    """ONNX Runtime vs PyTorch on real frames: (name, input shape, torch count, onnx count, max |diff|)."""
    import onnxruntime as ort
    import torch
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    detector = DensityDetector(model=net, max_side=max_side)
    out = []
    for name, frame in frames:
        x = detector.preprocess(frame)
        with torch.no_grad():
            a = net(x)[0, 0].numpy()
        b = sess.run(None, {"image": x.numpy()})[0][0, 0]
        out.append((name, tuple(x.shape[2:]), float(np.clip(a, 0, None).sum()),
                    float(np.clip(b, 0, None).sum()), float(np.abs(a - b).max())))
    return out


# ---------------- Backends: preprocessed input -> raw density map ----------------

def build_runners(model, weights, backends, static_hw):
    """{backend: fn(x tensor) -> raw map}. openvino-static only accepts the static_hw shape."""
    import torch
    net = load_net(model, weights)
    path = onnx_path(model, weights)
    runners = {}
    if "torch" in backends:
        def run_torch(x):
            with torch.no_grad():
                return net(x)[0, 0].numpy()
        runners["torch"] = run_torch
    makers = {"onnx": lambda: onnx_runner(path, THREADS),
              "openvino": lambda: openvino_runner(path, THREADS),
              "openvino-static": lambda: openvino_runner(path, THREADS, static_hw=static_hw),
              "openvino-bf16": lambda: openvino_runner(path, THREADS, precision="bf16")}
    for b in backends:
        if b in makers:
            run = makers[b]()
            runners[b] = lambda x, run=run: run(x.numpy())[0, 0]
    return net, runners


def rotated(items, k):
    k %= len(items)
    return items[k:] + items[:k]


# ---------------- Bench ----------------

def load_bench_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def row_key(model, set_name, image, pass_no, backend):
    return (model, set_name, image, str(pass_no), backend)


def bench(args):
    bench_paths, _ = split_samples(find_samples(args.data))
    sets = image_sets(bench_paths[:args.limit or None])
    done = {row_key(r["model"], r["set"], r["image"], r["pass"], r["backend"])
            for r in load_bench_rows(args.out)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    is_new = not os.path.isfile(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCH_FIELDS)
        if is_new:
            writer.writeheader()
        for model in args.models:
            todo = [row_key(model, s, n, p, b) for p in range(PASSES) for s in sets
                    for n, _ in sets[s] for b in args.backends]
            if all(k in done for k in todo):
                print(f"{model}: already complete in {args.out}")
                continue
            if not os.path.isfile(onnx_path(model, args.density_weights)):
                sys.exit(f"No {onnx_path(model, args.density_weights)}; run: python benchmark_backends.py export")
            print(f"Loading {model} on {', '.join(args.backends)}...", flush=True)
            # the camera set is one shape, so it is the one a static model can be built for
            import torch
            probe = DensityDetector(model=torch.nn.Identity(), max_side=args.max_side)
            static_hw = tuple(probe.preprocess(sets["camera720"][0][1]).shape[2:])
            net, runners = build_runners(model, args.density_weights, args.backends, static_hw)
            detector = DensityDetector(model=net, max_side=args.max_side)
            for p in range(PASSES):
                for set_name, frames in sets.items():
                    for i, (name, frame) in enumerate(frames):
                        x = detector.preprocess(frame)
                        hw = tuple(x.shape[2:])
                        for backend in rotated(list(args.backends), i + p):
                            if row_key(model, set_name, name, p, backend) in done:
                                continue
                            if backend == "openvino-static" and hw != static_hw:
                                continue  # one compiled shape only: n/a on Part A's mixed sizes
                            run = runners[backend]
                            run(x)  # warm-up, discarded
                            start = time.perf_counter()
                            raw = run(x)
                            seconds = time.perf_counter() - start
                            row = dict(model=model, set=set_name, image=name, input_h=hw[0],
                                       input_w=hw[1], seconds=f"{seconds:.4f}", backend=backend,
                                       count=float(np.clip(raw, 0, None).sum()),
                                       raw_count=float(raw.sum()), wall=f"{time.time():.1f}")
                            row["pass"] = p
                            writer.writerow(row)
                            f.flush()
                            done.add(row_key(model, set_name, name, p, backend))
                    print(f"{model} pass {p + 1}/{PASSES} {set_name}: done "
                          f"({time.strftime('%H:%M:%S')})", flush=True)
    print(f"Per-run results: {args.out}")


# ---------------- Report ----------------

def paired_speedups(rows, backend):
    """{image: [torch s / backend s]} for every (image, pass) both have."""
    t = {(r["image"], r["pass"]): float(r["seconds"]) for r in rows if r["backend"] == "torch"}
    out = {}
    for r in rows:
        k = (r["image"], r["pass"])
        if r["backend"] == backend and k in t:
            out.setdefault(r["image"], []).append(t[k] / float(r["seconds"]))
    return out


def median_interval(per_image, rounds=BOOTSTRAP_ROUNDS, seed=0):
    """Median of all values and a 95% bootstrap interval that resamples whole images."""
    groups = list(per_image.values())
    rng = np.random.default_rng(seed)
    meds = [np.median(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
            for _ in range(rounds)]
    low, high = np.percentile(meds, [2.5, 97.5])
    return float(np.median(np.concatenate(groups))), float(low), float(high)


def count_diffs(rows, backend):
    """Largest |count - torch count| over every (image, pass), absolute and as a fraction."""
    t = {(r["image"], r["pass"]): float(r["count"]) for r in rows if r["backend"] == "torch"}
    worst_abs = worst_rel = 0.0
    for r in rows:
        k = (r["image"], r["pass"])
        if r["backend"] == backend and k in t:
            d = abs(float(r["count"]) - t[k])
            worst_abs = max(worst_abs, d)
            worst_rel = max(worst_rel, d / max(t[k], 1e-6))
    return worst_abs, worst_rel


def summarise_bench(rows, backends=BACKENDS):
    """Table lines, the set of (model, backend) that qualify on every set, and checks."""
    lines = ["| Model | Input | Backend | Runs | Median s | p95 s | Speed-up (95% interval) "
             "| Max count diff | Qualifies |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    passed, checks = {}, []
    for model in dict.fromkeys(r["model"] for r in rows):
        for set_name in dict.fromkeys(r["set"] for r in rows if r["model"] == model):
            sub = [r for r in rows if r["model"] == model and r["set"] == set_name]
            shapes = {(r["input_w"], r["input_h"]) for r in sub}
            label = (f"{set_name} ({len(shapes)} sizes)" if len(shapes) > 1
                     else f"{set_name} ({'x'.join(next(iter(shapes)))})")
            zeros = sorted({r["image"] for r in sub if float(r["count"]) == 0})
            if zeros:
                checks.append(f"{model} {set_name}: exact 0 count on {zeros}")
            for backend in backends:
                b = [r for r in sub if r["backend"] == backend]
                if not b:
                    lines.append(f"| {model} | {label} | {backend} | 0 | n/a | n/a | n/a | n/a | n/a |")
                    continue
                s = np.array([float(r["seconds"]) for r in b])
                if backend == "torch":
                    speed, diff, ok = "1.00", "-", "-"
                else:
                    med, low, high = median_interval(paired_speedups(sub, backend))
                    d_abs, d_rel = count_diffs(sub, backend)
                    good = med >= ADOPT_SPEEDUP and d_rel <= ADOPT_COUNT_DIFF
                    passed.setdefault((model, backend), []).append(good)
                    speed = f"{med:.2f} ({low:.2f}-{high:.2f})"
                    diff, ok = f"{d_abs:.2f} ({d_rel * 100:.3f}%)", "yes" if good else "no"
                lines.append(f"| {model} | {label} | {backend} | {len(b)} | {np.median(s):.3f} "
                             f"| {np.percentile(s, 95):.3f} | {speed} | {diff} | {ok} |")
    qualified = {k for k, v in passed.items() if all(v)}
    walls = sorted(float(r["wall"]) for r in rows)
    gaps = np.diff(walls)
    if len(gaps) and gaps.max() > STALL_GAP:
        checks.append(f"stall: {int((gaps > STALL_GAP).sum())} gaps over {STALL_GAP}s between "
                      f"results, longest {gaps.max():.0f}s")
    return lines, qualified, checks


def report(args):
    rows = load_bench_rows(args.out)
    if not rows:
        sys.exit(f"No results in {args.out}; run the bench first")
    lines, qualified, checks = summarise_bench(rows)
    print("\n".join(lines))
    print(f"\nQualify on every input set (speed-up >= {ADOPT_SPEEDUP}x, every count within "
          f"{ADOPT_COUNT_DIFF * 100:.1f}% of torch): "
          f"{', '.join(f'{m} {b}' for m, b in sorted(qualified)) or 'none'}")
    if checks:
        print("\nCheck these:\n" + "\n".join(checks))


# ---------------- Test video ----------------

def make_video(args):
    _, video_paths = split_samples(find_samples(args.data))
    os.makedirs(os.path.dirname(args.video_out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, CAMERA_SIZE)
    if not writer.isOpened():
        sys.exit(f"Could not open a video writer for {args.video_out}")
    per_image = int(round(VIDEO_HOLD * VIDEO_FPS))
    for p in video_paths:
        frame = cv2.resize(cv2.imread(p), CAMERA_SIZE, interpolation=cv2.INTER_AREA)
        for _ in range(per_image):
            writer.write(frame)
    writer.release()
    cap = cv2.VideoCapture(args.video_out)
    frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Wrote {args.video_out}: {frames} frames at {fps:.0f} fps ({frames / fps:.0f} s), "
          f"images {', '.join(os.path.basename(p) for p in video_paths)}")


# ---------------- Main ----------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["export", "bench", "report", "video"])
    p.add_argument("--data", help="folder containing part_A_final (bench, video, export check)")
    p.add_argument("--models", nargs="+", choices=list(DENSITY_MODELS), default=["dmcount", "csrnet"])
    p.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    p.add_argument("--density-weights", default="SHA")
    p.add_argument("--max-side", type=int, default=1000)
    p.add_argument("--out", default=BENCH_CSV, help="bench CSV; resumes if it exists")
    p.add_argument("--video-out", default=VIDEO_OUT)
    p.add_argument("--limit", type=int, default=0, help="bench only the first N images (a check, not a result)")
    return p.parse_args(argv)


def main(argv=None):
    # the ONNX exporter prints emoji, which the Windows console code page can't encode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    if args.command == "export":
        check = None
        if args.data:
            bench_paths, _ = split_samples(find_samples(args.data))
            check = image_sets(bench_paths[:3])
        for model in args.models:
            path = onnx_path(model, args.density_weights)
            net = export_onnx(model, args.density_weights, path)
            print(f"Exported {path} ({os.path.getsize(path) / 1e6:.0f} MB)")
            for frames in (check or {}).values():
                for name, hw, a, b, d in check_export(net, path, frames, args.max_side):
                    print(f"  {name} {hw[1]}x{hw[0]}: torch {a:.2f}  onnx {b:.2f}  max |diff| {d:.2e}")
    elif args.command == "bench":
        if not args.data:
            sys.exit("bench needs --data")
        bench(args)
    elif args.command == "report":
        report(args)
    elif args.command == "video":
        if not args.data:
            sys.exit("video needs --data")
        make_video(args)


if __name__ == "__main__":
    main()
