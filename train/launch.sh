torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:10999 \
    --nnodes 1 \
    --nproc_per_node 8 \
    train/main.py \
    --env_conf train/llama2-7b.json \
    --n_samples_per_gpu 128 \
    --gamma 2.0 \
    --trainable_layers "[0,1,2]" \
    --trainable_tokens 32 \
    --merge_method slerp