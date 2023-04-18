docker run \
    --rm \
    --shm-size 80G \
    --gpus "all" \
    -v /home/james/Documents/recordings/:/data/ \
    -v /home/james/Documents/imitation-learning/configs/:/configs/ \
    imitation-learning \
    python ./src/main.py fit --config /configs/train.yaml
