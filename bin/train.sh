if [ "$1" = "stop" ]; then
    echo "!!!停止了训练!!!"
    ps aux|grep python|grep name=east|awk '{print $2}'|xargs kill -9
    exit
fi


if [ "$1" == "debug" ] || [ "$1" == "console" ]; then
    echo "###### 调试模式 ######"
    python -m main.train \
    --name=east \
    --debug=True \
    --gpu_list=0 \
    --max_steps=3 \
    --batch_size=1 \
    --num_readers=1 \
    --input_size=512 \
    --validate_steps=1 \
    --validate_batch_num=1 \
    --early_stop=1 \
    --save_summary_steps=1 \
    --model_path=./model \
    --tboard_dir=./logs/tboard \
    --text_scale=512 \
    --training_data_path=./data/train \
    --validate_data_path=./data/validate \
    --geometry=RBOX \
    --learning_rate=0.0001 \
    --pretrained_model_path=./model/resnet_v1_50.ckpt
    exit
fi

Date=$(date +%Y%m%d%H%M)
echo "###### 生产模式 ######"
    nohup \
    python -m main.train \
    --name=east \
    --debug=False \
    --gpu_list=0 \
    --max_steps=200000 \
    --batch_size=32 \
    --num_readers=5 \
    --input_size=512 \
    --validate_steps=1000 \
    --validate_batch_num=30 \
    --early_stop=100 \
    --save_summary_steps=100 \
    --model_path=./model \
    --tboard_dir=./logs/tboard \
    --text_scale=512 \
    --training_data_path=./data/train \
    --validate_data_path=./data/validate \
    --geometry=RBOX \
    --learning_rate=0.0001 \
    --pretrained_model_path=./model/resnet_v1_50.ckpt \
    >> ./logs/east_$Date.log 2>&1 &