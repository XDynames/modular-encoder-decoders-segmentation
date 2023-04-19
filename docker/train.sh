docker run \
    --rm \
    --shm-size 80G \
    --gpus "all" \
    -v /mnt/data/aarc/recordings/:/data/ \
    -v /home/james/Documents/ac-imitation-learning/configs/:/configs/ \
    imitation-learning \
    python ./src/main.py fit --config /configs/train.yaml
