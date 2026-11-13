# M4.4 distance vs KITTI (1000 frames, git 61e8ac98, Tesla T4)

Boxes: KITTI GT 2D boxes (score 1), not detector output. MiDaS: per frame 1/Z = s*m + t on anchors' geometric distances, leave-one-out for anchors.

## GT = LiDAR (primary), each method on the objects it covers
| method | bin | n | coverage | AbsRel | RMSE (m) | bias | median rel |
|---|---|---|---|---|---|---|---|
| geometric | 0-10 | 364 | 0.81 | 0.085 | 0.94 | 0.001 | 0.071 |
| geometric | 10-30 | 1637 | 1.00 | 0.151 | 5.12 | 0.014 | 0.112 |
| geometric | 30-60 | 886 | 0.96 | 0.234 | 11.77 | 0.003 | 0.179 |
| geometric | all | 2887 | 0.96 | 0.168 | 7.58 | 0.009 | 0.120 |
| midas_small | 0-10 | 364 | 0.81 | 0.207 | 2.38 | 0.104 | 0.140 |
| midas_small | 10-30 | 1491 | 0.91 | 0.206 | 6.33 | 0.010 | 0.160 |
| midas_small | 30-60 | 804 | 0.87 | 0.302 | 19.55 | -0.006 | 0.219 |
| midas_small | all | 2659 | 0.88 | 0.236 | 11.78 | 0.018 | 0.172 |
| dpt_hybrid | 0-10 | 364 | 0.81 | 0.177 | 2.12 | 0.034 | 0.116 |
| dpt_hybrid | 10-30 | 1494 | 0.91 | 0.204 | 6.67 | 0.001 | 0.158 |
| dpt_hybrid | 30-60 | 808 | 0.87 | 0.316 | 18.50 | 0.096 | 0.223 |
| dpt_hybrid | all | 2666 | 0.88 | 0.234 | 11.37 | 0.034 | 0.165 |
| fused_midas_small | 0-10 | 446 | 0.99 | 0.121 | 1.22 | 0.035 | 0.084 |
| fused_midas_small | 10-30 | 1640 | 1.00 | 0.153 | 4.48 | -0.025 | 0.123 |
| fused_midas_small | 30-60 | 922 | 1.00 | 0.218 | 10.60 | -0.078 | 0.187 |
| fused_midas_small | all | 3008 | 1.00 | 0.168 | 6.76 | -0.032 | 0.130 |
| fused_dpt_hybrid | 0-10 | 446 | 0.99 | 0.121 | 1.36 | 0.011 | 0.083 |
| fused_dpt_hybrid | 10-30 | 1640 | 1.00 | 0.159 | 5.19 | -0.016 | 0.123 |
| fused_dpt_hybrid | 30-60 | 922 | 1.00 | 0.237 | 12.44 | -0.014 | 0.185 |
| fused_dpt_hybrid | all | 3008 | 1.00 | 0.177 | 7.90 | -0.012 | 0.132 |

## GT = LiDAR, common subset (objects every method estimated)
| method | bin | n | coverage | AbsRel | RMSE (m) | bias | median rel |
|---|---|---|---|---|---|---|---|
| geometric | 0-10 | 282 | - | 0.086 | 0.98 | -0.018 | 0.063 |
| geometric | 10-30 | 1488 | - | 0.151 | 5.11 | 0.017 | 0.111 |
| geometric | 30-60 | 762 | - | 0.234 | 11.74 | 0.014 | 0.177 |
| geometric | all | 2532 | - | 0.169 | 7.55 | 0.012 | 0.121 |
| midas_small | 0-10 | 282 | - | 0.193 | 2.49 | 0.082 | 0.134 |
| midas_small | 10-30 | 1488 | - | 0.207 | 6.34 | 0.010 | 0.160 |
| midas_small | 30-60 | 762 | - | 0.287 | 18.09 | -0.026 | 0.212 |
| midas_small | all | 2532 | - | 0.229 | 11.08 | 0.007 | 0.171 |
| dpt_hybrid | 0-10 | 282 | - | 0.164 | 2.15 | 0.011 | 0.109 |
| dpt_hybrid | 10-30 | 1488 | - | 0.201 | 6.31 | -0.002 | 0.157 |
| dpt_hybrid | 30-60 | 762 | - | 0.295 | 17.00 | 0.065 | 0.216 |
| dpt_hybrid | all | 2532 | - | 0.225 | 10.53 | 0.020 | 0.163 |
| fused_midas_small | 0-10 | 282 | - | 0.093 | 1.05 | -0.015 | 0.072 |
| fused_midas_small | 10-30 | 1488 | - | 0.153 | 4.41 | -0.025 | 0.124 |
| fused_midas_small | 30-60 | 762 | - | 0.208 | 9.83 | -0.091 | 0.181 |
| fused_midas_small | all | 2532 | - | 0.163 | 6.38 | -0.044 | 0.131 |
| fused_dpt_hybrid | 0-10 | 282 | - | 0.103 | 1.26 | -0.035 | 0.073 |
| fused_dpt_hybrid | 10-30 | 1488 | - | 0.157 | 4.73 | -0.018 | 0.124 |
| fused_dpt_hybrid | 30-60 | 762 | - | 0.219 | 10.65 | -0.035 | 0.175 |
| fused_dpt_hybrid | all | 2532 | - | 0.170 | 6.89 | -0.025 | 0.134 |

## GT = LiDAR, MiDaS split by alignment mode (affine vs scale-only fit)
| method | bin | n | coverage | AbsRel | RMSE (m) | bias | median rel |
|---|---|---|---|---|---|---|---|
| midas_small@affine | 0-10 | 210 | 0.47 | 0.215 | 2.61 | 0.130 | 0.137 |
| midas_small@affine | 10-30 | 843 | 0.51 | 0.211 | 6.42 | 0.026 | 0.163 |
| midas_small@affine | 30-60 | 402 | 0.43 | 0.365 | 24.98 | 0.053 | 0.230 |
| midas_small@affine | all | 1455 | 0.48 | 0.255 | 14.04 | 0.049 | 0.175 |
| midas_small@scale | 0-10 | 154 | 0.34 | 0.196 | 2.01 | 0.068 | 0.149 |
| midas_small@scale | 10-30 | 648 | 0.39 | 0.199 | 6.21 | -0.012 | 0.154 |
| midas_small@scale | 30-60 | 402 | 0.43 | 0.240 | 11.86 | -0.064 | 0.207 |
| midas_small@scale | all | 1204 | 0.40 | 0.213 | 8.26 | -0.019 | 0.169 |
| dpt_hybrid@affine | 0-10 | 210 | 0.47 | 0.185 | 2.30 | 0.081 | 0.109 |
| dpt_hybrid@affine | 10-30 | 847 | 0.52 | 0.191 | 6.55 | 0.025 | 0.150 |
| dpt_hybrid@affine | 30-60 | 406 | 0.44 | 0.316 | 19.84 | 0.091 | 0.211 |
| dpt_hybrid@affine | all | 1463 | 0.48 | 0.225 | 11.61 | 0.052 | 0.156 |
| dpt_hybrid@scale | 0-10 | 154 | 0.34 | 0.166 | 1.85 | -0.030 | 0.136 |
| dpt_hybrid@scale | 10-30 | 647 | 0.39 | 0.220 | 6.82 | -0.031 | 0.171 |
| dpt_hybrid@scale | 30-60 | 402 | 0.43 | 0.316 | 17.03 | 0.101 | 0.233 |
| dpt_hybrid@scale | all | 1203 | 0.40 | 0.245 | 11.06 | 0.013 | 0.183 |

## GT = LiDAR p10 (near surface), each method
| method | bin | n | coverage | AbsRel | RMSE (m) | bias | median rel |
|---|---|---|---|---|---|---|---|
| geometric | 0-10 | 403 | 0.82 | 0.080 | 0.88 | 0.033 | 0.069 |
| geometric | 10-30 | 1626 | 1.00 | 0.150 | 5.46 | 0.036 | 0.104 |
| geometric | 30-60 | 859 | 0.96 | 0.230 | 11.58 | 0.007 | 0.174 |
| geometric | all | 2888 | 0.96 | 0.164 | 7.54 | 0.027 | 0.112 |
| midas_small | 0-10 | 400 | 0.82 | 0.222 | 2.45 | 0.139 | 0.161 |
| midas_small | 10-30 | 1481 | 0.91 | 0.207 | 6.44 | 0.030 | 0.157 |
| midas_small | 30-60 | 779 | 0.87 | 0.303 | 19.75 | 0.003 | 0.217 |
| midas_small | all | 2660 | 0.88 | 0.237 | 11.76 | 0.038 | 0.172 |
| dpt_hybrid | 0-10 | 400 | 0.82 | 0.180 | 2.10 | 0.065 | 0.113 |
| dpt_hybrid | 10-30 | 1483 | 0.91 | 0.202 | 6.78 | 0.023 | 0.146 |
| dpt_hybrid | 30-60 | 784 | 0.87 | 0.319 | 18.80 | 0.108 | 0.221 |
| dpt_hybrid | all | 2667 | 0.88 | 0.233 | 11.41 | 0.054 | 0.158 |
| fused_midas_small | 0-10 | 485 | 0.99 | 0.125 | 1.25 | 0.068 | 0.086 |
| fused_midas_small | 10-30 | 1630 | 1.00 | 0.148 | 4.50 | -0.006 | 0.116 |
| fused_midas_small | 30-60 | 894 | 1.00 | 0.215 | 10.55 | -0.072 | 0.180 |
| fused_midas_small | all | 3009 | 1.00 | 0.164 | 6.66 | -0.014 | 0.122 |
| fused_dpt_hybrid | 0-10 | 485 | 0.99 | 0.120 | 1.31 | 0.044 | 0.084 |
| fused_dpt_hybrid | 10-30 | 1630 | 1.00 | 0.156 | 5.32 | 0.004 | 0.119 |
| fused_dpt_hybrid | 30-60 | 894 | 1.00 | 0.235 | 12.43 | -0.009 | 0.180 |
| fused_dpt_hybrid | all | 3009 | 1.00 | 0.174 | 7.84 | 0.007 | 0.125 |

## GT = label_2 location z (secondary, box centre: expect negative bias)
| method | bin | n | coverage | AbsRel | RMSE (m) | bias | median rel |
|---|---|---|---|---|---|---|---|
| geometric | 0-10 | 281 | 0.76 | 0.132 | 1.41 | -0.049 | 0.110 |
| geometric | 10-30 | 1641 | 1.00 | 0.170 | 4.92 | -0.047 | 0.142 |
| geometric | 30-60 | 965 | 0.96 | 0.238 | 11.73 | -0.029 | 0.197 |
| geometric | all | 2887 | 0.96 | 0.189 | 7.74 | -0.041 | 0.154 |
| midas_small | 0-10 | 286 | 0.78 | 0.216 | 2.42 | 0.020 | 0.179 |
| midas_small | 10-30 | 1500 | 0.91 | 0.216 | 6.24 | -0.047 | 0.180 |
| midas_small | 30-60 | 875 | 0.87 | 0.300 | 19.03 | -0.045 | 0.228 |
| midas_small | all | 2661 | 0.88 | 0.243 | 11.90 | -0.039 | 0.192 |
| dpt_hybrid | 0-10 | 286 | 0.78 | 0.201 | 2.27 | -0.046 | 0.167 |
| dpt_hybrid | 10-30 | 1503 | 0.91 | 0.210 | 5.79 | -0.065 | 0.179 |
| dpt_hybrid | 30-60 | 880 | 0.88 | 0.310 | 18.03 | 0.053 | 0.236 |
| dpt_hybrid | all | 2669 | 0.88 | 0.242 | 11.25 | -0.024 | 0.189 |
| fused_midas_small | 0-10 | 363 | 0.99 | 0.158 | 1.55 | -0.027 | 0.134 |
| fused_midas_small | 10-30 | 1643 | 1.00 | 0.176 | 4.63 | -0.083 | 0.156 |
| fused_midas_small | 30-60 | 1000 | 1.00 | 0.228 | 10.88 | -0.108 | 0.207 |
| fused_midas_small | all | 3006 | 1.00 | 0.191 | 7.17 | -0.085 | 0.167 |
| fused_dpt_hybrid | 0-10 | 363 | 0.99 | 0.159 | 1.70 | -0.051 | 0.131 |
| fused_dpt_hybrid | 10-30 | 1643 | 1.00 | 0.179 | 4.74 | -0.079 | 0.153 |
| fused_dpt_hybrid | 30-60 | 1000 | 1.00 | 0.246 | 12.70 | -0.046 | 0.202 |
| fused_dpt_hybrid | all | 3006 | 1.00 | 0.199 | 8.14 | -0.065 | 0.165 |

## Speed (Tesla T4)
| model | input | estimate ms (pre+model+post) | FPS | model-only ms by input |
|---|---|---|---|---|
| midas_small | [256, 832] | 25.4 | 39.4 | 256x448: 13.8, 256x832: 20.7 |
| dpt_hybrid | [384, 1248] | 404.2 | 2.5 | 384x672: 210.2, 384x1248: 409.9 |

No LiDAR support: {"total_objects": 3077, "no_lidar": 18, "by_label_z_bin": {"30-60": 13, "10-30": 3, "other": 2}, "by_class": {"car": 13, "rider": 2, "truck": 2, "person": 1}, "by_occluded": {"1": 12, "0": 6}, "zero_points": 9}
Label z - LiDAR z by class: {"car": {"n": 2279, "median_m": 1.479263129398234, "p90_m": 1.9324381199317664}, "person": {"n": 524, "median_m": 0.05593728043975155, "p90_m": 0.1304350103703392}, "rider": {"n": 156, "median_m": 0.056145327532282074, "p90_m": 0.26569103811504036}, "truck": {"n": 100, "median_m": 4.999785382657954, "p90_m": 6.085294216043107}}
Failed models: none
