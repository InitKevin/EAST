cd ../
# --gpu_list=0 \
python multigpu_train.py \
    --input_size=512 \
    --batch_size_per_gpu=1 \
    --checkpoint_path=./logs/east_icdar2015_resnet_v1_50_rbox/ \
    --text_scale=512 \
    --training_data_path=./data/ocr/icdar2015/ \
    --geometry=RBOX \
    --learning_rate=0.0001 \
    --num_readers=1 \
    --pretrained_model_path=./logs/resnet_v1_50.ckpt