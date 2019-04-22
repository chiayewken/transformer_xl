#!/bin/bash

GSDATA=$2
GSEXP=$2
DATASET=$3
LOCAL_DIR=data/$3
DROPOUT=$4

echo mode: $1
echo gsutil dir: ${GSDATA}
echo dataset: ${DATASET}
echo local_dir: ${LOCAL_DIR}
echo dropout_rate: ${DROPOUT}

# TPU setting
NUM_HOST=1 # Colab TPUv2 -> 1 | Depends on TPU configuration eg pods have more
NUM_CORE=8 # TPUv2 -> 8 | TPUv3 -> 16
TEST_NUM_HOST=1
TEST_NUM_CORE=8 # TPUv2 -> 8 | TPUv3 -> 16

# Model
# DIV_VAL=4  # Default = 1 | reduced vocab size due to BPE, no need to reduce emb_dim
N_LAYER=18
D_MODEL=1024
D_EMBED=1024
N_HEAD=16
D_HEAD=64
D_INNER=4096

# Training
# TGT_LEN=384
# MEM_LEN=384
# Align to sota/wt103.sh
TGT_LEN=256
MEM_LEN=256
# BSZ 64 + Colab TPU ->  Used 8.91G of 8.00G hbm
TRAIN_BSZ=32
VALID_BSZ=32
# num_passes repeats data
# default is 10 to mitigate TPU dropping remainder on small datasets
# should be able to set to 1 on large datasets
NUM_PASSES=1

# Testing
#TEST_TGT_LEN=128
TEST_TGT_LEN=256
TEST_MEM_LEN=1600
TEST_CLAMP_LEN=1000
TEST_BSZ=8

if [[ $1 == 'train_data' ]]; then
    # python data_utils.py \
    python data_utils_bpe.py \
        --data_dir=${LOCAL_DIR}/ \
        --dataset=${DATASET} \
        --tgt_len=${TGT_LEN} \
        --per_host_train_bsz=${TRAIN_BSZ} \
        --per_host_valid_bsz=${VALID_BSZ} \
        --num_core_per_host=${NUM_CORE} \
        --num_passes=${NUM_PASSES} \
        --use_tpu=True \
        ${@:2}

    # include corpus metadata in gcloud bucket
    gsutil -m cp ${LOCAL_DIR}/tfrecords/* ${GSDATA}/${DATASET}-tfrecords/
    gsutil -m cp ${LOCAL_DIR}/cache.pkl ${GSDATA}/${DATASET}-tfrecords/
    gsutil -m cp ${LOCAL_DIR}/corpus-info.json ${GSDATA}/${DATASET}-tfrecords/

elif [[ $1 == 'test_data' ]]; then
    # python data_utils.py \
    python data_utils_bpe.py \
        --data_dir=${LOCAL_DIR}/ \
        --dataset=${DATASET} \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=${TEST_BSZ} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --num_passes=1 \
        --use_tpu=True \
        ${@:2}

    SRC_PATTERN=test.bsz-${TEST_BSZ}.tlen-${TEST_TGT_LEN}.core-${TEST_NUM_CORE}*
    gsutil -m cp ${LOCAL_DIR}/tfrecords/${SRC_PATTERN} ${GSDATA}/${DATASET}-tfrecords/

elif [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --data_dir=${GSDATA}/${DATASET}-tfrecords \
        --record_info_dir=${LOCAL_DIR}/tfrecords/ \
        --corpus_info_path=${LOCAL_DIR}/corpus-info.json \
        --model_dir=${GSEXP}/${DATASET}_warmup_64k \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --proj_same_dim=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=${DROPOUT} \
        --dropatt=${DROPOUT} \
        --init_std=0.005 \
        --learning_rate=0.00025 \
        --warmup_steps=64000 \
        --train_steps=4000000 \
        --tgt_len=${TGT_LEN} \
        --mem_len=${MEM_LEN} \
        --train_batch_size=${TRAIN_BSZ} \
        --num_hosts=${NUM_HOST} \
        --num_core_per_host=${NUM_CORE} \
        --iterations=1000 \
        --save_steps=10000 \
        --use_tpu=True \
        --do_eval=False \
        ${@:2}

elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python train.py \
        --data_dir=${GSDATA}/${DATASET}-tfrecords \
        --record_info_dir=${LOCAL_DIR}/tfrecords/ \
        --corpus_info_path=${LOCAL_DIR}/corpus-info.json \
        --model_dir=${GSEXP}/${DATASET} \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --proj_same_dim=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${VALID_BSZ} \
        --num_host=${TEST_NUM_HOST} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --use_tpu=True \
        --do_train=False \
        --do_eval_only=True \
        --eval_split=valid \
        ${@:2}

elif [[ $1 == 'dynamic_eval' ]]; then
    # Uses GPU only
    NUM_CORE=1

    # Testing
    TEST_TGT_LEN=128
    TEST_MEM_LEN=1600
    TEST_CLAMP_LEN=1000

    TEST_CKPT_PATH=${LOCAL_DIR}/model.ckpt-0
    TEST_BSZ=1
    TEST_NUM_CORE=1
    export CUDA_VISIBLE_DEVICES=0


    echo 'Preprocess test set...'
    python data_utils_bpe.py \
        --data_dir=${LOCAL_DIR}/ \
        --dataset=${DATASET} \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=1 \
        --num_passes=1 \
        --use_tpu=False


    python data_utils_bpe.py \
        --data_dir=${LOCAL_DIR}/ \
        --dataset=wt103 \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=0 \
        --num_passes=1 \
        --use_tpu=False

    echo 'Run evaluation on test set...'
    python dynamiceval_tf.py \
        --data_dir=${LOCAL_DIR}/tfrecords \
        --record_info_dir=${LOCAL_DIR}/tfrecords/ \
        --corpus_info_path=${LOCAL_DIR}/corpus-info.json \
        --eval_ckpt_path=${TEST_CKPT_PATH} \
        --model_dir=EXP-${DATASET} \
        --learning_rate=0.000002\
        --decay_rate=0 \
        --epsilon=0.00001 \
        --rms=True\
        --untie_r=True \
        --proj_share_all_but_first=True \
        --num_core_per_host=${TEST_NUM_CORE} \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --eval_split=test\
        --same_length=True

else
    echo 'unknown argument 1'
fi
