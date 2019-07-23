#!/bin/sh

BATCH_SIZE=64
WORKER_SIZE=4
GPU_SIZE=2
CPU_SIZE=4
DATASET=20_scls_stt

nsml run -g $GPU_SIZE -c $CPU_SIZE -d $DATASET -a "--batch_size $BATCH_SIZE --workers $WORKER_SIZE --use_attention"
