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
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=128 n=32 b=8 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=128 n=32 b=16 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=128 n=32 b=32 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=128 n=32 b=64 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=128 n=32 b=128 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

### Flash Attention: OFF

Running: Llama-3.2-1B | p=128 n=32 b=1 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=128 n=32 b=8 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=128 n=32 b=16 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=128 n=32 b=32 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=128 n=32 b=64 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=128 n=32 b=128 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |


## Prompt Length: 512, Generation: 128

### Flash Attention: ON

Running: Llama-3.2-1B | p=512 n=128 b=1 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=512 n=128 b=8 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=512 n=128 b=16 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=512 n=128 b=32 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=512 n=128 b=64 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

### Flash Attention: OFF

Running: Llama-3.2-1B | p=512 n=128 b=1 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=512 n=128 b=8 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=512 n=128 b=16 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=512 n=128 b=32 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=512 n=128 b=64 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |


## Prompt Length: 1024, Generation: 256

### Flash Attention: ON

Running: Llama-3.2-1B | p=1024 n=256 b=1 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=1024 n=256 b=8 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=1024 n=256 b=16 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=1024 n=256 b=32 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

### Flash Attention: OFF

Running: Llama-3.2-1B | p=1024 n=256 b=1 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=1024 n=256 b=8 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=1024 n=256 b=16 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |

Running: Llama-3.2-1B | p=1024 n=256 b=32 fa=0
main: error: failed to load model '/home/maxious/koboldcpp/Llama-3.2-1B.Q8_0.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |


========================================
Testing GLM-4.7-Flash (Q4_K_M)
========================================

## Prompt Length: 128, Generation: 32

### Flash Attention: ON

Running: GLM-4.7-Flash | p=128 n=32 b=1 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/GLM-4.7-Flash-SynthLabs-REAP-25-Q4_K_M.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: GLM-4.7-Flash | p=128 n=32 b=8 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/GLM-4.7-Flash-SynthLabs-REAP-25-Q4_K_M.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: GLM-4.7-Flash | p=128 n=32 b=16 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/GLM-4.7-Flash-SynthLabs-REAP-25-Q4_K_M.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: GLM-4.7-Flash | p=128 n=32 b=32 fa=1
main: error: failed to load model '/home/maxious/koboldcpp/GLM-4.7-Flash-SynthLabs-REAP-25-Q4_K_M.gguf'
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | --------------: | -------------------: |

Running: GLM-4.7-Flash | p=128 n=32 b=64 fa=1
