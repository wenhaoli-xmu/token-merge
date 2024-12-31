torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:10999 \
    --nnodes 1 \
    --nproc_per_node 8 \
    train/main.py \
    --env_conf train/llama2-7b.json \
    --n_sample 64 \
    --accum_grad 8
