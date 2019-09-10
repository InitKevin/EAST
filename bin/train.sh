if [ "$1" = "stop" ]; then
    echo "!!!停止了训练!!!"
    ps aux|grep python|grep name=east_train|awk '{print $2}'|xargs kill -9
    exit
fi


if [ "$1" == "debug" ] || [ "$1" == "console" ]; then
    echo "###### 调试模式 ######"
    python -m main.train \
    --name=east_train \
    --debug=True \
    --gpu_list=0 \
    --max_steps=3 \
    --batch_size=2 \
    --num_readers=1 \
    --input_size=512 \
    --validate_steps=1 \
    --validate_batch_num=1 \
    --early_stop=3 \
    --save_summary_steps=1 \
    --model_path=./model \
    --tboard_dir=./logs/tboard \
    --text_scale=512 \
    --training_data_path=./data/train \
    --validate_data_path=./data/validate \
    --geometry=RBOX \
    --learning_rate=0.0001 \
    --lambda_AABB=1000 \
    --lambda_theta=100000 \
    --lambda_score=1\
    --pretrained_model_path=./model/resnet_v1_50.ckpt
    exit
fi

Date=$(date +%Y%m%d%H%M)
echo "###### 生产模式 ######"
    nohup \
    python -m main.train \
    --name=east_train \
    --debug=True \
    --gpu_list=0 \
    --max_steps=200000 \
    --batch_size=14 \
    --num_readers=100 \
    --input_size=512 \
    --validate_steps=1000 \
    --validate_batch_num=8 \
    --early_stop=100 \
    --save_summary_steps=100 \
    --model_path=./model \
    --tboard_dir=./logs/tboard \
    --text_scale=512 \
    --training_data_path=./data/train \
    --validate_data_path=./data/validate \
    --geometry=RBOX \
    --learning_rate=0.0001 \
    --lambda_AABB=100 \
    --lambda_theta=10000 \
    --lambda_score=1\
    --pretrained_model_path=./model/resnet_v1_50.ckpt \
    >> ./logs/east_$Date.log 2>&1 &