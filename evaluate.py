"""
Measure counting accuracy against labelled data: ShanghaiTech Part A test set (182 images).

Scores the detectors from crowd_monitor.py:
  yolo             - number of people detected at the --conf threshold
  csrnet / dmcount - sum of the density map, at each size in --sides (0 = native size)

Density networks are scored two ways: "count" is what crowd_monitor.py uses (negative
values clipped to 0) and "raw" is the plain sum of the network output, which is what the
published benchmarks report.

Results go to a per-image CSV after every image, so an interrupted run resumes where it
stopped. The CSV is keyed by config name, which includes the weights and threshold; use a
new --out if you change anything else about a run.

Examples:
    python evaluate.py --data C:\\datasets\\ShanghaiTech
    python evaluate.py --data C:\\datasets\\ShanghaiTech --models dmcount --sides 1000
    python evaluate.py --data C:\\datasets\\ShanghaiTech --limit 5     # quick check, not a result
"""
import argparse
import csv
import math
import os
import re
import sys
import time

import cv2
import numpy as np

from crowd_monitor import CONF_TH, DensityDetector, YoloDetector

# ---------------- DEFAULT SETTINGS ----------------
PART_DIRS = ("part_A_final", "part_A")            # the zip's name, then the older one
GT_DIRS = ("ground_truth", "ground-truth")
EXPECTED_TEST_IMAGES = 182

OUT_CSV = os.path.join("eval_results", "shanghaitech_A_test.csv")
CSV_FIELDS = ["config", "image", "truth", "count", "raw_count", "seconds"]

BOOTSTRAP_ROUNDS = 2000
WORST_SHOWN = 3        # largest errors listed per config, for checking by eye
RAW_DIFF_SHOWN = 0.5   # people; show a "raw" row only if it differs from "count" by this MAE
# ---------------------------------------------------

DATASET_HELP = """ShanghaiTech Part A test set not found at {path!r}: {problem}

This script needs the real labelled dataset and will not run on anything else.

Get it from the dataset authors' links (Dropbox or Baidu, about 174 MB, no account needed
for Dropbox): https://github.com/desenzhou/ShanghaiTechDataset

Unzip it and pass the folder that contains part_A_final:
    python evaluate.py --data C:\\datasets\\ShanghaiTech

Expected layout (part_A/ and ground-truth/ are also accepted):
    part_A_final/test_data/images/IMG_1.jpg
    part_A_final/test_data/ground_truth/GT_IMG_1.mat"""


# ---------------- Pure logic (unit tested) ----------------

def metrics(preds, truths):
    """Standard crowd-counting metrics. Positive bias means over-counting."""
    errors = np.asarray(preds, dtype=np.float64) - np.asarray(truths, dtype=np.float64)
    return dict(n=len(errors), mae=float(np.abs(errors).mean()),
                rmse=float(np.sqrt((errors ** 2).mean())), bias=float(errors.mean()))


def mae_interval(preds, truths, rounds=BOOTSTRAP_ROUNDS, seed=0):
    """95% bootstrap interval for MAE: how far 182 images pin the figure down."""
    errors = np.abs(np.asarray(preds, dtype=np.float64) - np.asarray(truths, dtype=np.float64))
    picks = np.random.default_rng(seed).integers(0, len(errors), (rounds, len(errors)))
    low, high = np.percentile(errors[picks].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def image_number(path):
    return int(re.search(r"(\d+)", os.path.basename(path)).group(1))


def config_name(model, side, density_weights, conf, backend="torch", precision="f32"):
    if model == "yolo":
        return f"yolo-conf{conf:.2f}"
    suffix = "" if backend == "torch" else f"+{backend}"  # torch keeps the original names
    if backend == "openvino" and precision != "f32":
        suffix += f"-{precision}"
    return f"{model}-{density_weights}@{side or 'native'}{suffix}"


def expand_configs(models, sides):
    """Every model/size pair to run. YOLO has its own input size, so it runs once."""
    configs = []
    for model in models:
        configs += [(model, 0)] if model == "yolo" else [(model, s) for s in sides]
    return configs


# ---------------- Dataset ----------------

def load_points(mat_path):
    """Head positions from a ShanghaiTech label file, as an N x 2 array of (x, y)."""
    from scipy.io import loadmat
    info = loadmat(mat_path)["image_info"][0, 0][0, 0]
    points = np.asarray(info["location"], dtype=np.float64).reshape(-1, 2)
    stored = int(np.asarray(info["number"]).ravel()[0])
    if stored != len(points):
        sys.exit(f"{mat_path}: label says {stored} people but lists {len(points)} points")
    return points


def find_samples(data, expected=EXPECTED_TEST_IMAGES):
    """[(image path, label path)] in image order. Exits with download help if incomplete."""
    def fail(problem):
        sys.exit(DATASET_HELP.format(path=data, problem=problem))

    test_dir = next((os.path.join(data, p, "test_data") for p in PART_DIRS
                     if os.path.isdir(os.path.join(data, p, "test_data"))), None)
    if test_dir is None:
        fail("no part_A_final/test_data folder")
    gt_dir = next((os.path.join(test_dir, g) for g in GT_DIRS
                   if os.path.isdir(os.path.join(test_dir, g))), None)
    image_dir = os.path.join(test_dir, "images")
    if gt_dir is None or not os.path.isdir(image_dir):
        fail("test_data needs both images and ground_truth folders")

    images = sorted((f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")),
                    key=image_number)
    samples = [(os.path.join(image_dir, f),
                os.path.join(gt_dir, f"GT_{os.path.splitext(f)[0]}.mat")) for f in images]
    unlabelled = [os.path.basename(img) for img, gt in samples if not os.path.isfile(gt)]
    if unlabelled:
        fail(f"{len(unlabelled)} images have no label file, e.g. {unlabelled[0]}")
    if len(samples) != expected:
        fail(f"found {len(samples)} test images, expected {expected}")
    return samples


# ---------------- Predictors: frame -> (count, raw count) ----------------

def build_predictor(model, side, args):
    if model == "yolo":
        detector = YoloDetector(args.weights, args.conf)

        def predict(frame):
            found = len(detector.detect(frame))
            return found, found
    else:
        detector = DensityDetector(model, args.density_weights, side, backend=args.backend,
                                   precision=args.precision)

        def predict(frame):
            raw = detector.predict(frame)
            return float(np.clip(raw, 0, None).sum()), float(raw.sum())
    return predict


# ---------------- Running and reporting ----------------

def load_rows(path):
    """Rows saved by an earlier run, as {(config, image): row}."""
    if not os.path.isfile(path):
        return {}
    with open(path, newline="") as f:
        return {(r["config"], r["image"]): r for r in csv.DictReader(f)}


def run_config(config, predict, samples, truths, rows, writer, flush):
    """Predict every image not already in `rows`, saving each result as it arrives."""
    for i, (image, _) in enumerate(samples, 1):
        name = os.path.basename(image)
        if (config, name) in rows:
            continue
        frame = cv2.imread(image)
        if frame is None:
            sys.exit(f"Could not read {image}")
        start = time.perf_counter()
        count, raw = predict(frame)
        seconds = time.perf_counter() - start
        if not (math.isfinite(count) and math.isfinite(raw)):
            sys.exit(f"{config} gave a non-finite count on {name}: {count}, {raw}")
        row = dict(config=config, image=name, truth=truths[name], count=count, raw_count=raw,
                   seconds=seconds)
        rows[(config, name)] = row
        writer.writerow(row)
        flush()
        if i % 20 == 0 or i == len(samples):
            print(f"{config}: {i}/{len(samples)} images ({seconds:.2f}s on the last)", flush=True)


def summarise(config, rows, total=EXPECTED_TEST_IMAGES):
    """Result lines for one config, plus anything worth checking by eye."""
    picked = [r for (c, _), r in rows.items() if c == config]
    truth = [float(r["truth"]) for r in picked]
    count = [float(r["count"]) for r in picked]
    raw = [float(r["raw_count"]) for r in picked]
    seconds = np.array([float(r["seconds"]) for r in picked])

    m = metrics(count, truth)
    low, high = mae_interval(count, truth)
    label = config if m["n"] == total else f"{config} PARTIAL {m['n']}/{total}"
    lines = [f"| {label} | {m['mae']:.1f} ({low:.1f}-{high:.1f}) | {m['rmse']:.1f} "
             f"| {m['bias']:+.1f} | {np.median(seconds):.2f} |"]
    r = metrics(raw, truth)
    if abs(r["mae"] - m["mae"]) >= RAW_DIFF_SHOWN:
        low, high = mae_interval(raw, truth)
        lines.append(f"| {label} raw | {r['mae']:.1f} ({low:.1f}-{high:.1f}) | {r['rmse']:.1f} "
                     f"| {r['bias']:+.1f} | |")

    checks = []
    zeros = [x["image"] for x, c in zip(picked, count) if c == 0]
    if zeros:
        checks.append(f"{config}: exact 0 on {len(zeros)} images, e.g. {zeros[:5]}")
    order = np.argsort(-np.abs(np.array(count) - np.array(truth)))[:WORST_SHOWN]
    worst = ", ".join(f"{picked[i]['image']} truth {truth[i]:.0f} got {count[i]:.0f}" for i in order)
    checks.append(f"{config}: worst: {worst}")
    return lines, checks


def report(configs, rows):
    print("\n| Config | MAE (95% interval) | RMSE | Bias | Median s/image |")
    print("|---|---:|---:|---:|---:|")
    all_checks = []
    for config in configs:
        lines, checks = summarise(config, rows)
        print("\n".join(lines))
        all_checks += checks
    print("\nCheck these by eye:")
    print("\n".join(all_checks))


# ---------------- Main ----------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder containing part_A_final")
    p.add_argument("--models", nargs="+", choices=["yolo", "csrnet", "dmcount"],
                   default=["yolo", "csrnet", "dmcount"])
    p.add_argument("--sides", nargs="+", type=int, default=[1000, 0],
                   help="--max-side values for the density networks (0 = native size)")
    p.add_argument("--density-weights", default="SHA", help="SHA, SHB or QNRF (DM-Count only)")
    p.add_argument("--backend", choices=["torch", "onnx", "openvino"], default="torch",
                   help="csrnet/dmcount: PyTorch, or the ONNX export run in ONNX Runtime or OpenVINO")
    p.add_argument("--precision", choices=["f32", "bf16"], default="f32", help="--backend openvino only")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLO weights file")
    p.add_argument("--conf", type=float, default=CONF_TH)
    p.add_argument("--out", default=OUT_CSV, help="per-image results CSV; resumes if it exists")
    p.add_argument("--limit", type=int, default=0, help="only the first N images (a check, not a result)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    samples = find_samples(args.data)
    if args.limit:
        samples = samples[:args.limit]
    truths = {os.path.basename(img): len(load_points(gt)) for img, gt in samples}
    print(f"{len(samples)} images, {sum(truths.values())} labelled heads")

    configs = expand_configs(args.models, args.sides)
    names = [config_name(m, s, args.density_weights, args.conf, args.backend,
                         args.precision) for m, s in configs]
    rows = load_rows(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    is_new = not os.path.isfile(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        for (model, side), name in zip(configs, names):
            if all((name, os.path.basename(img)) in rows for img, _ in samples):
                print(f"{name}: already complete in {args.out}")
                continue
            print(f"Loading {name}...", flush=True)
            run_config(name, build_predictor(model, side, args), samples, truths, rows,
                       writer, f.flush)
    report(names, rows)
    print(f"\nPer-image results: {args.out}")


if __name__ == "__main__":
    main()
