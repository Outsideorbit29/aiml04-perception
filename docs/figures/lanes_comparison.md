| Group | Subset | N | IoU classical | IoU yolop | IoU measured-only classical | IoU measured-only yolop | F1 classical | F1 yolop | P/R classical | P/R yolop |
|---|---|---|---|---|---|---|---|---|---|---|
| all | all | 500 | 0.423 | 0.415 | 0.239 | 0.224 | 0.231 | 0.515 | 0.24/0.23 | 0.60/0.45 |
| timeofday | dawn/dusk | 100 | 0.402 | 0.412 | 0.221 | 0.195 | 0.235 | 0.553 | 0.25/0.23 | 0.66/0.47 |
| timeofday | daytime | 200 | 0.445 | 0.419 | 0.288 | 0.253 | 0.192 | 0.517 | 0.19/0.20 | 0.59/0.46 |
| timeofday | night | 200 | 0.411 | 0.413 | 0.201 | 0.211 | 0.271 | 0.493 | 0.29/0.26 | 0.58/0.43 |
| weather | clear | 279 | 0.424 | 0.415 | 0.228 | 0.231 | 0.261 | 0.530 | 0.28/0.25 | 0.62/0.46 |
| weather | foggy | 1 | 0.569 | 0.573 | 0.000 | 0.000 | 0.667 | 0.667 | 1.00/0.50 | 1.00/0.50 |
| weather | overcast | 65 | 0.492 | 0.476 | 0.311 | 0.280 | 0.267 | 0.579 | 0.27/0.26 | 0.64/0.53 |
| weather | partly cloudy | 36 | 0.360 | 0.400 | 0.259 | 0.182 | 0.178 | 0.506 | 0.18/0.18 | 0.61/0.43 |
| weather | rainy | 36 | 0.400 | 0.452 | 0.208 | 0.240 | 0.208 | 0.432 | 0.22/0.20 | 0.51/0.37 |
| weather | snowy | 38 | 0.395 | 0.315 | 0.219 | 0.086 | 0.071 | 0.367 | 0.06/0.10 | 0.47/0.30 |
| weather | undefined | 45 | 0.387 | 0.367 | 0.233 | 0.210 | 0.159 | 0.458 | 0.16/0.16 | 0.56/0.39 |
| geometry | curve | 20 | 0.294 | 0.366 | 0.176 | 0.099 | 0.065 | 0.207 | 0.08/0.06 | 0.27/0.17 |
| geometry | no_gt | 108 | 0.317 | 0.117 | 0.104 | 0.008 | 0.000 | 0.000 | 0.00/- | 0.00/- |
| geometry | straight | 372 | 0.452 | 0.480 | 0.271 | 0.276 | 0.260 | 0.539 | 0.29/0.24 | 0.64/0.47 |

| Group | Subset | N | GT empty | GT sides | GT short | no pred classical | no pred yolop | synth classical | synth yolop | IoU when pred classical | IoU when pred yolop | both sides classical | both sides yolop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | all | 500 | 69 | 657 | 100 | 28 | 129 | 249 | 222 | 0.451 | 0.522 | 0.45 | 0.30 |
| timeofday | dawn/dusk | 100 | 13 | 133 | 29 | 4 | 23 | 48 | 50 | 0.422 | 0.504 | 0.48 | 0.27 |
| timeofday | daytime | 200 | 31 | 260 | 47 | 6 | 52 | 86 | 81 | 0.462 | 0.532 | 0.54 | 0.34 |
| timeofday | night | 200 | 25 | 264 | 24 | 18 | 54 | 115 | 91 | 0.456 | 0.520 | 0.34 | 0.28 |
| weather | clear | 279 | 29 | 379 | 52 | 20 | 67 | 144 | 122 | 0.459 | 0.521 | 0.41 | 0.32 |
| weather | foggy | 1 | 0 | 2 | 0 | 0 | 0 | 1 | 1 | 0.569 | 0.573 | 0.00 | 0.00 |
| weather | overcast | 65 | 7 | 87 | 16 | 3 | 10 | 28 | 31 | 0.519 | 0.552 | 0.52 | 0.37 |
| weather | partly cloudy | 36 | 6 | 51 | 9 | 1 | 8 | 14 | 20 | 0.372 | 0.445 | 0.58 | 0.22 |
| weather | rainy | 36 | 5 | 51 | 6 | 1 | 9 | 22 | 16 | 0.413 | 0.583 | 0.36 | 0.31 |
| weather | snowy | 38 | 13 | 30 | 3 | 1 | 21 | 18 | 14 | 0.412 | 0.526 | 0.50 | 0.08 |
| weather | undefined | 45 | 9 | 57 | 14 | 2 | 14 | 22 | 18 | 0.410 | 0.489 | 0.47 | 0.29 |
| geometry | curve | 20 | 1 | 36 | 0 | 3 | 2 | 8 | 14 | 0.349 | 0.409 | 0.45 | 0.20 |
| geometry | no_gt | 108 | 37 | 0 | 58 | 9 | 76 | 70 | 29 | 0.357 | 0.319 | 0.27 | 0.03 |
| geometry | straight | 372 | 31 | 621 | 42 | 16 | 51 | 171 | 179 | 0.474 | 0.546 | 0.50 | 0.38 |

| Method | Device | Frames | Total ms (mean) | p50 ms | p90 ms | FPS | Backend ms | Model ms | Fit ms |
|---|---|---|---|---|---|---|---|---|---|
| classical | CPU Intel(R) Xeon(R) CPU @ 2.00GHz | 500 | 40.1 | 37.7 | 51.3 | 24.9 | 21.0 | - | 19.1 |
| yolop | GPU Tesla T4 | 500 | 61.3 | 60.9 | 71.0 | 16.3 | 46.7 | 42.2 | 14.7 |

- BDD100K val, 500 images (seed 42, stratified by timeofday {'daytime': 200, 'night': 200, 'dawn/dusk': 100}); each image independent (`detect(stateless=True)`); camera geometry configs/lanes_bdd.yaml.
- IoU: ego_mask vs BDD drivable area 'direct', rows [0.58, 0.74] of the image height; mean of per-image IoU over images with GT in the band (no prediction = 0). Single-side fallback on (lane width 0.37 of the bird's-eye width, confidence x0.5): masks with one synthesized side are counted in 'synth'; 'IoU measured-only' scores those images as no prediction (old behaviour).
- F1: measured left/right ego boundaries only (synthesized sides never count) vs the nearest BDD parallel lane lines, TP if mean x error <= 20 px over the band; P/R = precision/recall. GT sides = scorable GT boundaries (>= 5 of 30 band rows); 'GT short' boundaries exist but are too short in the band and are not scored; geometry 'no_gt' = no scorable GT.
- Speed: detect() wall time after 10 warm-up calls, incl. pre/post-processing and polyline fit; Backend = binary stage, Model = onnxruntime session.run, Fit = shared warp + sliding window + polyfit. Classical on CPU, YOLOP (ONNX Runtime FP32) on GPU.
- YOLOP model only, fixed tensor, median of 50: 24.9 ms with host I/O, 22.9 ms with CUDA I/O binding.
