# BDD100K EDA summary

## train (all 70k)
- images 69863, boxes 1273579, boxes/image mean 18.23, median 17, p95 36, max 88, empty 10
- box sizes (all classes): {'small': 705984, 'medium': 405780, 'large': 161815}
- timeofday: {'daytime': 36728, 'night': 27971, 'dawn/dusk': 5027, 'undefined': 137}
- weather: {'clear': 37344, 'overcast': 8770, 'undefined': 8119, 'snowy': 5549, 'rainy': 5070, 'partly cloudy': 4881, 'foggy': 130}

| class | boxes | share | images with class | small | medium | large | occluded |
|---|---|---|---|---|---|---|---|
| car | 700703 | 55.02% | 68943 | 301797 | 264792 | 134114 | 68% |
| truck | 27892 | 2.19% | 18078 | 5006 | 12043 | 10843 | 67% |
| bus | 11977 | 0.94% | 9281 | 2018 | 4917 | 5042 | 67% |
| person | 92159 | 7.24% | 22469 | 45098 | 41757 | 5304 | 58% |
| rider | 4560 | 0.36% | 3655 | 1640 | 2199 | 721 | 91% |
| bicycle | 7124 | 0.56% | 4398 | 1751 | 4186 | 1187 | 86% |
| motorcycle | 3023 | 0.24% | 2327 | 883 | 1532 | 608 | 78% |
| traffic light | 187871 | 14.75% | 39401 | 167287 | 20481 | 103 | 3% |
| traffic sign | 238270 | 18.71% | 57188 | 180504 | 53873 | 3893 | 12% |

## train subset (20k)
- images 20000, boxes 364698, boxes/image mean 18.23, median 17, p95 36, max 72, empty 2
- box sizes (all classes): {'small': 202218, 'medium': 116289, 'large': 46191}
- timeofday: {'daytime': 10514, 'night': 8007, 'dawn/dusk': 1439, 'undefined': 40}
- weather: {'clear': 10691, 'overcast': 2511, 'undefined': 2324, 'snowy': 1589, 'rainy': 1451, 'partly cloudy': 1397, 'foggy': 37}

| class | boxes | share | images with class | small | medium | large | occluded |
|---|---|---|---|---|---|---|---|
| car | 200532 | 54.99% | 19712 | 86340 | 75916 | 38276 | 68% |
| truck | 7924 | 2.17% | 5139 | 1440 | 3399 | 3085 | 68% |
| bus | 3428 | 0.94% | 2671 | 579 | 1398 | 1451 | 67% |
| person | 26382 | 7.23% | 6461 | 12989 | 11933 | 1460 | 58% |
| rider | 1358 | 0.37% | 1084 | 491 | 655 | 212 | 91% |
| bicycle | 2085 | 0.57% | 1300 | 542 | 1208 | 335 | 86% |
| motorcycle | 839 | 0.23% | 655 | 235 | 429 | 175 | 75% |
| traffic light | 53713 | 14.73% | 11320 | 47692 | 5998 | 23 | 4% |
| traffic sign | 68437 | 18.77% | 16373 | 51910 | 15353 | 1174 | 12% |

## val (10k)
- images 10000, boxes 185930, boxes/image mean 18.59, median 17, p95 36, max 66, empty 0
- box sizes (all classes): {'small': 103084, 'medium': 59707, 'large': 23139}
- timeofday: {'daytime': 5258, 'night': 3929, 'dawn/dusk': 778, 'undefined': 35}
- weather: {'clear': 5346, 'overcast': 1239, 'undefined': 1157, 'snowy': 769, 'rainy': 738, 'partly cloudy': 738, 'foggy': 13}

| class | boxes | share | images with class | small | medium | large | occluded |
|---|---|---|---|---|---|---|---|
| car | 102837 | 55.31% | 9882 | 45114 | 38573 | 19150 | 69% |
| truck | 4243 | 2.28% | 2733 | 752 | 1862 | 1629 | 68% |
| bus | 1660 | 0.89% | 1299 | 278 | 705 | 677 | 69% |
| person | 13425 | 7.22% | 3261 | 6507 | 6165 | 753 | 58% |
| rider | 658 | 0.35% | 527 | 242 | 313 | 103 | 91% |
| bicycle | 1039 | 0.56% | 592 | 257 | 622 | 160 | 90% |
| motorcycle | 460 | 0.25% | 346 | 134 | 231 | 95 | 79% |
| traffic light | 26884 | 14.46% | 5651 | 23723 | 3150 | 11 | 4% |
| traffic sign | 34724 | 18.68% | 8211 | 26077 | 8086 | 561 | 12% |

## Class-handling numbers (train subset)

| class | car/class ratio | inv-sqrt-freq weight |
|---|---|---|
| car | 1.0 | 0.157 |
| truck | 25.3 | 0.787 |
| bus | 58.5 | 1.197 |
| person | 7.6 | 0.432 |
| rider | 147.7 | 1.902 |
| bicycle | 96.2 | 1.535 |
| motorcycle | 239.0 | 2.42 |
| traffic light | 3.7 | 0.302 |
| traffic sign | 2.9 | 0.268 |
