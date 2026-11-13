import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_backends import count_diffs, paired_speedups, rotated, split_samples, summarise_bench
from crowd_monitor import DensityDetector
from evaluate import config_name


class TinyNet(torch.nn.Module):
    """A real conv net (so the export has weights), with a 1/8-resolution output."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 1, 3, padding=1)
        self.pool = torch.nn.AvgPool2d(8)

    def forward(self, x):
        return self.pool(self.conv(x))


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    folder = tmp_path_factory.mktemp("models")
    net = TinyNet().eval()
    h, w = torch.export.Dim("h", min=16, max=2048), torch.export.Dim("w", min=16, max=2048)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the exporter prints emoji
    torch.onnx.export(net, (torch.zeros(1, 3, 64, 96),), str(folder / "DM-Count_T.onnx"),
                      input_names=["image"], dynamic_shapes={"x": {2: h, 3: w}}, dynamo=True,
                      external_data=False, verbose=False)
    return net, folder


@pytest.mark.parametrize("backend", ["onnx", "openvino"])
def test_exported_backends_match_torch_at_two_sizes(exported, backend):
    net, folder = exported
    ref = DensityDetector(model=net, max_side=320)
    other = DensityDetector("dmcount", "T", max_side=320, backend=backend, model_dir=str(folder))
    rng = np.random.default_rng(0)
    for shape in [(240, 320, 3), (200, 150, 3)]:  # dynamic input size
        frame = rng.integers(0, 255, shape, dtype=np.uint8)
        a, b = ref.predict(frame), other.predict(frame)
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, atol=1e-5)
        assert other.analyse(frame)[1].shape == shape[:2]


def test_missing_export_says_how_to_make_it(tmp_path):
    with pytest.raises(SystemExit, match="benchmark_backends.py export"):
        DensityDetector("dmcount", "SHA", backend="onnx", model_dir=str(tmp_path))


def test_backend_runs_get_their_own_evaluate_config_names():
    assert config_name("dmcount", 0, "SHA", 0.4) == "dmcount-SHA@native"
    assert config_name("dmcount", 0, "SHA", 0.4, "torch") == "dmcount-SHA@native"
    assert config_name("dmcount", 1000, "SHA", 0.4, "openvino") == "dmcount-SHA@1000+openvino"
    assert config_name("yolo", 0, "SHA", 0.4, "openvino") == "yolo-conf0.40"


# ---------------- bench bookkeeping ----------------

def test_rotation_moves_every_backend_through_every_position():
    b = ["a", "b", "c"]
    assert [rotated(b, k)[0] for k in range(3)] == ["a", "b", "c"]
    assert rotated(b, 4) == ["b", "c", "a"]


def test_bench_and_video_images_do_not_overlap():
    samples = [(f"IMG_{i}.jpg", None) for i in range(182)]
    bench, video = split_samples(samples)
    assert len(bench) == len(video) == 20 and not set(bench) & set(video)
    assert split_samples(samples) == (bench, video)  # fixed seed


def bench_rows(backend, seconds, counts, wall0=0.0):
    return [{"model": "m", "set": "s", "image": f"I{i}", "pass": "0", "backend": backend,
             "input_h": "560", "input_w": "992", "seconds": str(s), "count": str(c),
             "raw_count": str(c), "wall": str(wall0 + i)}
            for i, (s, c) in enumerate(zip(seconds, counts))]


def test_speedups_and_count_diffs_are_paired_per_image():
    rows = bench_rows("torch", [2.0, 4.0], [100, 200]) + bench_rows("fast", [1.0, 1.0], [100, 201])
    assert paired_speedups(rows, "fast") == {"I0": [2.0], "I1": [4.0]}
    assert count_diffs(rows, "fast") == (1.0, pytest.approx(0.005))


def test_backend_qualifies_only_if_fast_enough_and_counts_match():
    torch_rows = bench_rows("torch", [2.0] * 4, [100] * 4)
    good = bench_rows("good", [1.0] * 4, [100.2] * 4)       # 2x, 0.2% off
    slow = bench_rows("slow", [1.9] * 4, [100] * 4)         # 1.05x
    off = bench_rows("off", [1.0] * 3 + [1.0], [100] * 3 + [101])  # one image 1% off
    _, qualified, _ = summarise_bench(torch_rows + good + slow + off,
                                      backends=["torch", "good", "slow", "off"])
    assert qualified == {("m", "good")}


def test_bench_summary_flags_stalls_and_exact_zeros():
    rows = bench_rows("torch", [1.0, 1.0], [0, 5]) + bench_rows("torch", [1.0], [5], wall0=500)
    rows[-1]["image"] = "I9"
    _, _, checks = summarise_bench(rows, backends=["torch"])
    assert any("exact 0" in c for c in checks)
    assert any("stall" in c for c in checks)


def test_bf16_runs_get_their_own_config_name():
    assert config_name("dmcount", 0, "SHA", 0.4, "openvino", "bf16") == "dmcount-SHA@native+openvino-bf16"
    assert config_name("dmcount", 0, "SHA", 0.4, "torch", "bf16") == "dmcount-SHA@native"

