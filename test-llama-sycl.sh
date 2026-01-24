cd /home/maxious/llama.cpp && source /opt/intel/oneapi/setvars.sh intel64 2>/dev/null && \
  export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 && \
  export ZES_ENABLE_SYSMAN=1 && \
  export ONEAPI_DEVICE_SELECTOR=level_zero:0 && \
  timeout 30 ./build-sycl/bin/llama-completion \
    --model ~/koboldcpp/Llama-3.2-1B.Q8_0.gguf \
    --prompt "1+1=" --n-predict 4 -ngl 99 2>&1