# Flash Attention Benchmark Results

**Hardware**: Intel Arc Pro B60 (Battlemage BMG-G21)
**Date**: $(date)
**Backend**: SYCL

========================================
Testing Llama-3.2-1B (Q8_0)
========================================

## Prompt Length: 128, Generation: 32

### Flash Attention: ON

Running: Llama-3.2-1B | p=128 n=32 b=1 fa=1
ggml_sycl: XMX detection: device=Intel(R) Arc(TM) Pro B60 Graphics, has_xmx=1, tile_kind=16x16x16 (PVC)
ggml_sycl: Using XMX (cooperative matrix) for flash attention with 16x16x16 (PVC) tiles
ggml_sycl: XMX Output O[0:4] = [0.002350, 0.013489, -0.045410, -0.001938]
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       1 |        1 |  1 |           pp128 |         43.56 ± 0.01 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       1 |        1 |  1 |            tg32 |         43.67 ± 0.10 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=128 n=32 b=8 fa=1
ggml_sycl: XMX detection: device=Intel(R) Arc(TM) Pro B60 Graphics, has_xmx=1, tile_kind=16x16x16 (PVC)
ggml_sycl: Using XMX (cooperative matrix) for flash attention with 16x16x16 (PVC) tiles
ggml_sycl: XMX Output O[0:4] = [0.003265, 0.013184, -0.045410, -0.001892]
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       8 |        8 |  1 |           pp128 |         75.07 ± 0.01 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       8 |        8 |  1 |            tg32 |         43.64 ± 0.03 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=128 n=32 b=16 fa=1
ggml_sycl: XMX detection: device=Intel(R) Arc(TM) Pro B60 Graphics, has_xmx=1, tile_kind=16x16x16 (PVC)
ggml_sycl: Using XMX (cooperative matrix) for flash attention with 16x16x16 (PVC) tiles
ggml_sycl: XMX Output O[0:4] = [0.002350, 0.013489, -0.045410, -0.001953]
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      16 |       16 |  1 |           pp128 |        660.57 ± 0.17 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      16 |       16 |  1 |            tg32 |         43.59 ± 0.05 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=128 n=32 b=32 fa=1
ggml_sycl: XMX detection: device=Intel(R) Arc(TM) Pro B60 Graphics, has_xmx=1, tile_kind=16x16x16 (PVC)
ggml_sycl: Using XMX (cooperative matrix) for flash attention with 16x16x16 (PVC) tiles
ggml_sycl: XMX Output O[0:4] = [0.002350, 0.013489, -0.045410, -0.001953]
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      32 |       32 |  1 |           pp128 |       1300.92 ± 2.36 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      32 |       32 |  1 |            tg32 |         43.48 ± 0.12 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=128 n=32 b=64 fa=1
ggml_sycl: XMX detection: device=Intel(R) Arc(TM) Pro B60 Graphics, has_xmx=1, tile_kind=16x16x16 (PVC)
ggml_sycl: Using XMX (cooperative matrix) for flash attention with 16x16x16 (PVC) tiles
ggml_sycl: XMX Output O[0:4] = [0.002350, 0.013489, -0.045410, -0.001953]
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      64 |       64 |  1 |           pp128 |      2220.78 ± 23.14 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      64 |       64 |  1 |            tg32 |         43.65 ± 0.02 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=128 n=32 b=128 fa=1
ggml_sycl: XMX detection: device=Intel(R) Arc(TM) Pro B60 Graphics, has_xmx=1, tile_kind=16x16x16 (PVC)
ggml_sycl: Using XMX (cooperative matrix) for flash attention with 16x16x16 (PVC) tiles
ggml_sycl: XMX Output O[0:4] = [0.002350, 0.013489, -0.045410, -0.001953]
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |     128 |      128 |  1 |           pp128 |       3672.12 ± 7.59 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |     128 |      128 |  1 |            tg32 |         43.54 ± 0.07 |

build: e3abf94a5 (7860)

### Flash Attention: OFF

Running: Llama-3.2-1B | p=128 n=32 b=1 fa=0
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       1 |        1 |           pp128 |         55.00 ± 0.00 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       1 |        1 |            tg32 |         54.91 ± 0.08 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=128 n=32 b=8 fa=0
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       8 |        8 |           pp128 |         78.18 ± 0.00 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       8 |        8 |            tg32 |         54.89 ± 0.14 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=128 n=32 b=16 fa=0
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      16 |       16 |           pp128 |        809.92 ± 0.27 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      16 |       16 |            tg32 |         55.04 ± 0.02 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=128 n=32 b=32 fa=0
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      32 |       32 |           pp128 |       1568.34 ± 0.49 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      32 |       32 |            tg32 |         54.97 ± 0.06 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=128 n=32 b=64 fa=0
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      64 |       64 |           pp128 |       2926.69 ± 1.71 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      64 |       64 |            tg32 |         55.00 ± 0.01 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=128 n=32 b=128 fa=0
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |     128 |      128 |           pp128 |       4949.51 ± 4.64 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |     128 |      128 |            tg32 |         55.07 ± 0.02 |

build: e3abf94a5 (7860)


## Prompt Length: 512, Generation: 128

### Flash Attention: ON

Running: Llama-3.2-1B | p=512 n=128 b=1 fa=1
ggml_sycl: XMX detection: device=Intel(R) Arc(TM) Pro B60 Graphics, has_xmx=1, tile_kind=16x16x16 (PVC)
ggml_sycl: Using XMX (cooperative matrix) for flash attention with 16x16x16 (PVC) tiles
ggml_sycl: XMX Output O[0:4] = [0.002350, 0.013489, -0.045410, -0.001938]
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       1 |        1 |  1 |           pp512 |         40.75 ± 0.02 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       1 |        1 |  1 |           tg128 |         43.48 ± 0.01 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=512 n=128 b=8 fa=1
ggml_sycl: XMX detection: device=Intel(R) Arc(TM) Pro B60 Graphics, has_xmx=1, tile_kind=16x16x16 (PVC)
ggml_sycl: Using XMX (cooperative matrix) for flash attention with 16x16x16 (PVC) tiles
ggml_sycl: XMX Output O[0:4] = [0.003265, 0.013184, -0.045410, -0.001892]
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       8 |        8 |  1 |           pp512 |         73.96 ± 0.00 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       8 |        8 |  1 |           tg128 |         43.56 ± 0.02 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=512 n=128 b=16 fa=1
ggml_sycl: XMX detection: device=Intel(R) Arc(TM) Pro B60 Graphics, has_xmx=1, tile_kind=16x16x16 (PVC)
ggml_sycl: Using XMX (cooperative matrix) for flash attention with 16x16x16 (PVC) tiles
ggml_sycl: XMX Output O[0:4] = [0.002350, 0.013489, -0.045410, -0.001953]
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      16 |       16 |  1 |           pp512 |        617.85 ± 0.68 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      16 |       16 |  1 |           tg128 |         43.61 ± 0.04 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=512 n=128 b=32 fa=1
ggml_sycl: XMX detection: device=Intel(R) Arc(TM) Pro B60 Graphics, has_xmx=1, tile_kind=16x16x16 (PVC)
ggml_sycl: Using XMX (cooperative matrix) for flash attention with 16x16x16 (PVC) tiles
ggml_sycl: XMX Output O[0:4] = [0.002350, 0.013489, -0.045410, -0.001953]
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      32 |       32 |  1 |           pp512 |       1217.86 ± 2.66 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      32 |       32 |  1 |           tg128 |         43.59 ± 0.05 |

build: e3abf94a5 (7860)

Running: Llama-3.2-1B | p=512 n=128 b=64 fa=1
ggml_sycl: XMX detection: device=Intel(R) Arc(TM) Pro B60 Graphics, has_xmx=1, tile_kind=16x16x16 (PVC)
ggml_sycl: Using XMX (cooperative matrix) for flash attention with 16x16x16 (PVC) tiles
ggml_sycl: XMX Output O[0:4] = [0.002350, 0.013489, -0.045410, -0.001953]
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      64 |       64 |  1 |           pp512 |       2004.42 ± 1.70 |
