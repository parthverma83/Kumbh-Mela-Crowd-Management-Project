import csv
import io
import os
import sys

import numpy as np
import pytest
import torch
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crowd_monitor import DensityDetector
from evaluate import (CSV_FIELDS, config_name, expand_configs, find_samples, load_points,
                      load_rows, mae_interval, metrics, run_config, summarise)


def write_label(path, points, number=None):
    """Same nesting as the real files: image_info -> 1x1 struct of location, number."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    fields = np.empty((1, 1), dtype=[("location", "O"), ("number", "O")])
    fields[0, 0]["location"] = points
    fields[0, 0]["number"] = np.array([[len(points) if number is None else number]])
    info = np.empty((1, 1), dtype=object)
    info[0, 0] = fields
    savemat(path, {"image_info": info})


def make_dataset(root, numbers, part="part_A_final", gt="ground_truth"):
    images = root / part / "test_data" / "images"
    labels = root / part / "test_data" / gt
    images.mkdir(parents=True)
    labels.mkdir()
    for n in numbers:
        (images / f"IMG_{n}.jpg").write_bytes(b"not read by find_samples")
        write_label(labels / f"GT_IMG_{n}.mat", [[1, 2], [3, 4]])
    return root


def test_metrics_known_values():
    m = metrics([12, 8, 10], [10, 10, 16])  # errors +2, -2, -6
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(10 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(44 / 3))
    assert m["bias"] == pytest.approx(-2)


def test_metrics_perfect_prediction_is_zero():
    assert metrics([5, 7], [5, 7]) == dict(n=2, mae=0.0, rmse=0.0, bias=0.0)


def test_mae_interval_brackets_the_mae_and_repeats():
    rng = np.random.default_rng(1)
    truth = rng.integers(50, 500, 182).astype(float)
    pred = truth + rng.normal(0, 40, 182)
    low, high = mae_interval(pred, truth)
    assert low < metrics(pred, truth)["mae"] < high
    assert mae_interval(pred, truth) == (low, high)


def test_load_points_reads_the_real_nesting(tmp_path):
    p = tmp_path / "GT_IMG_1.mat"
    write_label(p, [[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    points = load_points(p)
    assert points.shape == (3, 2)
    assert points[1].tolist() == [3.5, 4.5]


def test_load_points_keeps_an_empty_label(tmp_path):
    p = tmp_path / "GT_IMG_1.mat"
    write_label(p, [])
    assert load_points(p).shape == (0, 2)


def test_load_points_rejects_a_count_that_disagrees(tmp_path):
    p = tmp_path / "GT_IMG_1.mat"
    write_label(p, [[1, 2]], number=5)
    with pytest.raises(SystemExit, match="5 people"):
        load_points(p)


@pytest.mark.parametrize("part,gt", [("part_A_final", "ground_truth"), ("part_A", "ground-truth")])
def test_find_samples_accepts_both_layouts_in_numeric_order(tmp_path, part, gt):
    root = make_dataset(tmp_path, [10, 2, 1], part, gt)
    samples = find_samples(root, expected=3)
    assert [os.path.basename(i) for i, _ in samples] == ["IMG_1.jpg", "IMG_2.jpg", "IMG_10.jpg"]
    assert os.path.basename(samples[2][1]) == "GT_IMG_10.mat"


def test_find_samples_missing_dataset_gives_download_help(tmp_path):
    with pytest.raises(SystemExit, match="github.com/desenzhou/ShanghaiTechDataset"):
        find_samples(tmp_path / "nowhere")


def test_find_samples_refuses_an_incomplete_test_set(tmp_path):
    root = make_dataset(tmp_path, [1, 2])
    with pytest.raises(SystemExit, match="found 2 test images, expected 182"):
        find_samples(root)


def test_find_samples_refuses_an_unlabelled_image(tmp_path):
    root = make_dataset(tmp_path, [1, 2])
    os.remove(root / "part_A_final" / "test_data" / "ground_truth" / "GT_IMG_2.mat")
    with pytest.raises(SystemExit, match="no label file"):
        find_samples(root, expected=2)


def test_configs_run_yolo_once_and_name_their_weights():
    configs = expand_configs(["yolo", "dmcount"], [1000, 0])
    assert configs == [("yolo", 0), ("dmcount", 1000), ("dmcount", 0)]
    assert config_name("yolo", 0, "SHA", 0.4) == "yolo-conf0.40"
    assert config_name("csrnet", 1000, "SHA", 0.4) == "csrnet-SHA@1000"
    assert config_name("dmcount", 0, "QNRF", 0.4) == "dmcount-QNRF@native"


@pytest.fixture
def frame_files(tmp_path):
    import cv2
    paths = []
    for i in (1, 2, 3):
        p = tmp_path / f"IMG_{i}.jpg"
        cv2.imwrite(str(p), np.zeros((16, 16, 3), dtype=np.uint8))
        paths.append((str(p), "unused"))
    return paths


def run(samples, rows, predict, config="m"):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    truths = {os.path.basename(i): 10 for i, _ in samples}
    run_config(config, predict, samples, truths, rows, writer, buf.flush)
    return buf.getvalue()


def test_run_config_resumes_without_repeating_images(frame_files, tmp_path):
    rows = {("m", "IMG_1.jpg"): dict(config="m", image="IMG_1.jpg", truth=10, count=9,
                                     raw_count=9, seconds=1)}
    calls = []
    written = run(frame_files, rows, lambda f: calls.append(1) or (7.0, 7.0))
    assert len(calls) == 2
    assert len(rows) == 3
    assert "IMG_1.jpg" not in written  # kept from the earlier run, not written twice


def test_saved_rows_survive_a_round_trip(frame_files, tmp_path):
    rows = {}
    path = tmp_path / "r.csv"
    path.write_text(",".join(CSV_FIELDS) + "\n" + run(frame_files, rows, lambda f: (7.0, 6.5)))
    loaded = load_rows(path)
    assert len(loaded) == 3
    assert float(loaded[("m", "IMG_2.jpg")]["raw_count"]) == 6.5


def test_run_config_stops_on_a_non_finite_count(frame_files):
    with pytest.raises(SystemExit, match="non-finite"):
        run(frame_files, {}, lambda f: (float("nan"), 1.0))


def rows_of(counts, raws=None, truth=100, config="m"):
    raws = counts if raws is None else raws
    return {(config, f"IMG_{i}.jpg"): dict(config=config, image=f"IMG_{i}.jpg", truth=truth,
                                           count=c, raw_count=r, seconds=1.0)
            for i, (c, r) in enumerate(zip(counts, raws))}


def test_summary_marks_a_partial_run():
    lines, _ = summarise("m", rows_of([90, 110]))
    assert "PARTIAL 2/182" in lines[0]


def test_summary_is_not_partial_when_complete():
    lines, _ = summarise("m", rows_of([90, 110]), total=2)
    assert "PARTIAL" not in lines[0]


def test_summary_flags_exact_zeros():
    _, checks = summarise("m", rows_of([0, 100]), total=2)
    assert any("exact 0 on 1 images" in c for c in checks)


def test_summary_adds_a_raw_row_only_when_it_differs():
    same, _ = summarise("m", rows_of([90, 110], [90, 110]), total=2)
    assert len(same) == 1
    differs, _ = summarise("m", rows_of([90, 110], [50, 60]), total=2)
    assert len(differs) == 2 and "raw" in differs[1]


class NegativeNet(torch.nn.Module):
    """Like CSRNet's linear output: half the map is below zero."""

    def forward(self, x):
        _, _, h, w = x.shape
        out = torch.ones((1, 1, h // 8, w // 8))
        out[..., ::2] = -0.5
        return out


def test_max_side_zero_keeps_the_native_size():
    detector = DensityDetector(model=NegativeNet(), max_side=0)
    frame = np.zeros((301, 455, 3), dtype=np.uint8)  # not divisible by 16
    assert detector.preprocess(frame).shape[2:] == (301, 455)


def test_predict_is_raw_and_analyse_clips():
    detector = DensityDetector(model=NegativeNet(), max_side=320)
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    raw = detector.predict(frame)
    assert raw.min() < 0
    _, density = detector.analyse(frame)
    assert density.min() >= 0
    # evaluate.py derives "count" this way; it has to equal what the monitor would show
    assert density.sum() == pytest.approx(np.clip(raw, 0, None).sum(), rel=1e-3)
