#!/bin/sh

BATCH_SIZE=64
WORKER_SIZE=4
GPU_SIZE=2
CPU_SIZE=4

nsml run -d 20_scls_stt -g $GPU_SIZE -c $CPU_SIZE -a "--batch_size $BATCH_SIZE --workers $WORKER_SIZE --use_attention"
