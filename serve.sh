
vllm serve canopylabs/orpheus-3b-0.1-ft \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.7 \
    --max-num-batched-tokens 16384 -tp 8