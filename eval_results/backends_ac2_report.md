| Model | Input | Backend | Runs | Median s | p95 s | Speed-up (95% interval) | Max count diff | Qualifies |
|---|---|---|---:|---:|---:|---:|---:|---|
| dmcount | partA (14 sizes) | torch | 22 | 1.752 | 2.164 | 1.00 | - | - |
| dmcount | partA (14 sizes) | onnx | 21 | 1.503 | 1.841 | 1.20 (1.18-1.22) | 0.00 (0.000%) | no |
| dmcount | partA (14 sizes) | openvino | 22 | 1.435 | 2.059 | 1.17 (1.14-1.30) | 0.00 (0.000%) | no |
| dmcount | partA (14 sizes) | openvino-static | 0 | n/a | n/a | n/a | n/a | n/a |
| dmcount | partA (14 sizes) | openvino-bf16 | 22 | 0.757 | 1.160 | 2.39 (2.22-2.50) | 4.25 (0.399%) | yes |
| dmcount | camera720 (992x560) | torch | 20 | 1.545 | 1.687 | 1.00 | - | - |
| dmcount | camera720 (992x560) | onnx | 20 | 1.314 | 1.389 | 1.22 (1.19-1.24) | 0.00 (0.000%) | yes |
| dmcount | camera720 (992x560) | openvino | 20 | 1.456 | 2.274 | 1.08 (0.95-1.30) | 0.00 (0.000%) | no |
| dmcount | camera720 (992x560) | openvino-static | 20 | 1.432 | 2.382 | 1.12 (0.88-1.22) | 0.00 (0.000%) | no |
| dmcount | camera720 (992x560) | openvino-bf16 | 20 | 0.773 | 1.026 | 2.06 (1.74-2.52) | 3.49 (0.371%) | yes |

Qualify on every input set (speed-up >= 1.2x, every count within 0.5% of torch): dmcount openvino-bf16
