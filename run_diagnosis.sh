#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,4,5,6,7

export DATA_DIR="eICU/"
export TF_ENABLE_ONEDNN_OPTS=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 


EXPIRED_LR=0.00011
EXPIRED_DROPOUT=0.72
FINE_TUNING_LR=0.00007
FINE_TUNING_DROPOUT=0.8
MDP_LR=2e-4
MDP_DROPOUT=0.5


OUTPUT_DIR="/dirs/gated_prompt"
mkdir -p "$OUTPUT_DIR"


LOG_DIR="logs/gated/"
mkdir -p "$LOG_DIR"


expired_ALL() {
    local fold=$1
    local device="$CUDA_VISIBLE_DEVICES"
    local dropout=$3
    local learning_rate=$4
    local prior_type="bert_prior"
    local LABEL_KEY=diagnosis
    local log_dir="${LOG_DIR}_los_base_${dropout}_${learning_rate}_fold_$fold.log"
    

    echo "prior guide unpdate, bert_prior, adr and gate" >> "$log_dir"
    

    python train_gated.py \
        --data_dir "$DATA_DIR" \
        --fold "$fold" \
        --output_dir "$OUTPUT_DIR" \
        --use_prior \
        --use_guide \
        --output_hidden_states \
        --output_attentions \
        --do_train \
        --do_eval \
        --do_test \
        --label_key "$LABEL_KEY" \
        --max_steps 100000 \
        --hidden_dropout_prob "$dropout" \
        --num_stacks 2 \
        --reg_coef 1.5 \
        --prior_type "$prior_type" \
        --learning_rate "$MDP_LR" \
        --task_desc "diagnosis_base_${dropout}_${learning_rate}" \
        --log_dir "$log_dir" \
        --device "0" \
        --mdp \
        --share_weights\
        --do_edge_prompt \
        --do_code_prompt \
        --do_gate_mechanism \
        --use_adr_pooler
}


for FOLD in 0 1 2 3 4; do
    expired_ALL "$FOLD" "$CUDA_VISIBLE_DEVICES" "$MDP_DROPOUT" "$MDP_LR"
done
