GPU_ID=6
BATCH_SIZE=1
WORKER_NUMBER=6
LEARNING_RATE=0.001
DECAY_STEP=5
EPOCHS=7
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset pascal_voc --net vgg16 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda --epochs $EPOCHS
