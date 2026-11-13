| Model | Input | Backend | Runs | Median s | p95 s | Speed-up (95% interval) | Max count diff | Qualifies |
|---|---|---|---:|---:|---:|---:|---:|---|
| dmcount | partA (14 sizes) | torch | 40 | 2.005 | 2.267 | 1.00 | - | - |
| dmcount | partA (14 sizes) | onnx | 40 | 1.593 | 1.942 | 1.26 (1.23-1.27) | 0.00 (0.000%) | yes |
| dmcount | partA (14 sizes) | openvino | 40 | 1.038 | 1.267 | 1.91 (1.84-1.94) | 0.00 (0.000%) | yes |
| dmcount | partA (14 sizes) | openvino-static | 0 | n/a | n/a | n/a | n/a | n/a |
| dmcount | partA (14 sizes) | openvino-bf16 | 40 | 0.536 | 0.633 | 3.70 (3.63-3.78) | 4.25 (0.399%) | yes |
| dmcount | camera720 (992x560) | torch | 40 | 1.780 | 2.038 | 1.00 | - | - |
| dmcount | camera720 (992x560) | onnx | 40 | 1.416 | 1.558 | 1.26 (1.25-1.28) | 0.00 (0.000%) | yes |
| dmcount | camera720 (992x560) | openvino | 40 | 0.903 | 0.982 | 1.96 (1.94-2.02) | 0.00 (0.000%) | yes |
| dmcount | camera720 (992x560) | openvino-static | 40 | 0.894 | 0.981 | 1.98 (1.93-2.00) | 0.00 (0.000%) | yes |
| dmcount | camera720 (992x560) | openvino-bf16 | 40 | 0.475 | 0.520 | 3.78 (3.70-3.86) | 3.49 (0.371%) | yes |
| csrnet | partA (14 sizes) | torch | 40 | 2.029 | 2.468 | 1.00 | - | - |
| csrnet | partA (14 sizes) | onnx | 40 | 1.641 | 1.968 | 1.23 (1.22-1.25) | 0.00 (0.000%) | yes |
| csrnet | partA (14 sizes) | openvino | 40 | 1.030 | 1.169 | 1.97 (1.91-2.04) | 0.00 (0.000%) | yes |
| csrnet | partA (14 sizes) | openvino-static | 0 | n/a | n/a | n/a | n/a | n/a |
| csrnet | partA (14 sizes) | openvino-bf16 | 40 | 0.532 | 0.610 | 3.85 (3.79-3.92) | 2.73 (1.312%) | no |
| csrnet | camera720 (992x560) | torch | 40 | 1.748 | 1.933 | 1.00 | - | - |
| csrnet | camera720 (992x560) | onnx | 40 | 1.416 | 1.578 | 1.22 (1.20-1.24) | 0.00 (0.000%) | yes |
| csrnet | camera720 (992x560) | openvino | 40 | 0.887 | 1.158 | 1.95 (1.91-1.97) | 0.00 (0.000%) | yes |
| csrnet | camera720 (992x560) | openvino-static | 40 | 0.886 | 1.002 | 1.97 (1.94-2.01) | 0.00 (0.000%) | yes |
| csrnet | camera720 (992x560) | openvino-bf16 | 40 | 0.464 | 0.521 | 3.77 (3.72-3.88) | 1.82 (1.146%) | no |

Qualify on every input set (speed-up >= 1.2x, every count within 0.5% of torch): csrnet onnx, csrnet openvino, csrnet openvino-static, dmcount onnx, dmcount openvino, dmcount openvino-bf16, dmcount openvino-static
