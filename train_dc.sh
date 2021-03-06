GPU_ID=2
BATCH_SIZE=1
WORKER_NUMBER=6
LEARNING_RATE=0.001
DECAY_STEP=5
EPOCHS=7
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset pascal_voc --net vgg16_dc \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda --epochs $EPOCHS --r true --checkpoint 10021 --checkepoch 3
