GPU_ID=$1
NET=$2
SESSION=1
EPOCH=7
CHECKPOINT=10021
CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset pascal_voc --net $NET \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
