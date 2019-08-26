if [ "$1" == "debug" ] || [ "$1" == "console" ]; then
    echo "###### 调试模式 ######"
    python train.py \
    --debug=True \
    --gpu_list=0 \
    --input_size=512 \
    --batch_size_per_gpu=1 \
    --save_checkpoint_steps=1 \
    --save_summary_steps=1 \
    --checkpoint_path=./model/checkpoint/ \
    --text_scale=512 \
    --training_data_path=./data/train \
    --geometry=RBOX \
    --learning_rate=0.0001 \
    --num_readers=1 \
    --pretrained_model_path=./model/resnet_v1_50.ckpt
    exit
fi

Date=$(date +%Y%m%d%H%M)
echo "###### 生产模式 ######"
    nohup \
    python train.py \
    --debug=False \
    --gpu_list=0 \
    --input_size=512 \
    --batch_size_per_gpu=1 \
    --save_checkpoint_steps=1000 \
    --save_summary_steps=100 \
    --checkpoint_path=./model/checkpoint/ \
    --text_scale=512 \
    --training_data_path=./data/train \
    --geometry=RBOX \
    --learning_rate=0.0001 \
    --num_readers=1 \
    --pretrained_model_path=./model/resnet_v1_50.ckpt \
    >> ./logs/east_$Date.log 2>&1 &