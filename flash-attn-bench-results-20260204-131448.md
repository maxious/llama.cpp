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
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       1 |        1 |  1 |           pp128 |         53.28 ± 0.02 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       1 |        1 |  1 |            tg32 |         53.55 ± 0.01 |

build: d780109d1 (8048)

Running: Llama-3.2-1B | p=128 n=32 b=8 fa=1
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       8 |        8 |  1 |           pp128 |         77.35 ± 0.06 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       8 |        8 |  1 |            tg32 |         53.43 ± 0.09 |

build: d780109d1 (8048)

Running: Llama-3.2-1B | p=128 n=32 b=16 fa=1
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      16 |       16 |  1 |           pp128 |        774.54 ± 3.77 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      16 |       16 |  1 |            tg32 |         53.81 ± 0.04 |

build: d780109d1 (8048)

Running: Llama-3.2-1B | p=128 n=32 b=32 fa=1
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      32 |       32 |  1 |           pp128 |       1483.45 ± 3.88 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      32 |       32 |  1 |            tg32 |         53.51 ± 0.03 |

build: d780109d1 (8048)

Running: Llama-3.2-1B | p=128 n=32 b=64 fa=1
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      64 |       64 |  1 |           pp128 |      2576.77 ± 10.82 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      64 |       64 |  1 |            tg32 |         53.35 ± 0.19 |

build: d780109d1 (8048)

Running: Llama-3.2-1B | p=128 n=32 b=128 fa=1
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |     128 |      128 |  1 |           pp128 |      4020.20 ± 43.31 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |     128 |      128 |  1 |            tg32 |         53.50 ± 0.07 |

build: d780109d1 (8048)

### Flash Attention: OFF

Running: Llama-3.2-1B | p=128 n=32 b=1 fa=0
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       1 |        1 |           pp128 |         54.29 ± 0.01 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       1 |        1 |            tg32 |         54.35 ± 0.01 |

build: d780109d1 (8048)

Running: Llama-3.2-1B | p=128 n=32 b=8 fa=0
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       8 |        8 |           pp128 |         77.67 ± 0.01 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       8 |        8 |            tg32 |         54.33 ± 0.01 |

build: d780109d1 (8048)

Running: Llama-3.2-1B | p=128 n=32 b=16 fa=0
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      16 |       16 |           pp128 |        809.77 ± 0.08 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      16 |       16 |            tg32 |         54.42 ± 0.02 |

build: d780109d1 (8048)

Running: Llama-3.2-1B | p=128 n=32 b=32 fa=0
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      32 |       32 |           pp128 |       1575.11 ± 1.49 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      32 |       32 |            tg32 |         54.30 ± 0.01 |

build: d780109d1 (8048)

Running: Llama-3.2-1B | p=128 n=32 b=64 fa=0
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      64 |       64 |           pp128 |       2945.23 ± 2.21 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |      64 |       64 |            tg32 |         54.27 ± 0.01 |

build: d780109d1 (8048)

Running: Llama-3.2-1B | p=128 n=32 b=128 fa=0
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |     128 |      128 |           pp128 |      5147.85 ± 12.75 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |     128 |      128 |            tg32 |         54.25 ± 0.02 |

build: d780109d1 (8048)


## Prompt Length: 512, Generation: 128

### Flash Attention: ON

Running: Llama-3.2-1B | p=512 n=128 b=1 fa=1
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | SYCL       |  99 |       1 |        1 |  1 |           pp512 |         51.63 ± 0.03 |
