torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:10999 \
    --nnodes 1 \
    --nproc_per_node 1 \
    train/main.py \
    --env_conf train/llama2-7b.json \
    --n_sample 1024 \
    --accum_grad 8
