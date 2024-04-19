prune_model='/path/to/prune_model/'
data='/path/to/data/'
output='/path/to/posttrain_model/'
export CUDA_VISIBLE_DEVICES=0

python post_train.py \
    --prune_model ${prune_model} \
    --data_path ${data} \
    --num_epochs 2 \
    --peft \
    --lora_r 64 \
    --lora_target_modules 'q_proj,k_proj,o_proj,gate_proj,down_proj,up_proj' \
    --learning_rate 1e-2 \
    --lr_scheduler_type 'cosine' \
    --micro_batch_size 8 \
    --batch_size 64 \
    --output_dir ${output} \
