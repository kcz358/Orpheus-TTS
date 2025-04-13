
vllm serve canopylabs/orpheus-tts-0.1-finetune-prod \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.7 \
    --max-num-batched-tokens 16384 -tp 8