export CUDA_VISIBLE_DEVICES=0

model="/path/to/model"
python main_bbo.py \
    --model ${model} \
    --remain_ratio 0.8 \
    --nsamples 1024 \
    --save_model \
