# EAST代码改造

## 环境

- tensorflow 1.8+
- python 3.6+

## 子目录约定
- bin/  各类启动shell脚本
- data/ 存放训练、验证数据
  - data/train/images   训练图片 
  - data/train/labels   训练标签，是一个[x1,y1,x2,y2,x3,y3,x4,y4]的格式
- test/ 各类测试脚本和程序
- model/ 存放模型的目录
- logs/ 日志存目录

## 服务器端查看tboard

使用tboard.sh来启动，对应绑定端口为8080:ai；8081:ai2；依次递推

## 常用脚本
- port  查看服务器端口
- gpu   查看gpu使用情况
- log   查看近期200条日志（最新的日志文件）
- logf  动态查看最新日志文件 

## 改造工作及其工作
- 要先跑起来EAST本身的代码，跑的标准是，可以用它的标注样本跑一个训练epoch出来即可
- 深入理解代码，并且严格和论文对比，搞清实现的原理
- 根据EAST的格式，修改代码，适配我们的样本格式
- 按照我们的格式，跑起来训练过程
- 排查各类问题、坑，按照既定的F1值进行调参
- 与之前的ocr web进行集成

## 代码理解改造难点
- 网络结构，特别是上卷积和concat的细节
- 样本的生成，特别是缩进1/3的边缘，生成各边界距离的feature map
- 损失函数的实现理解
- 改进的NMS的实现的理解
- 验证代码的实现的理解（可能没有，需要参考ctpn的实现）
* 样本如何准备的，一个4点标注的标签，怎样就变成了一个score map+4个距离map+1个旋转角度 map？
* 在一个批次的时候，喂给训练的模型的到底是什么？是抽样的一些像素么？如果是？是怎么样的抽样方式
* 代码中使用的reset net v50，究竟是哪些层被抽取出来？east中有5层被合并，这5层如何和reset net对应？
* 预测的最后阶段，是一个什么网络实现了输出一个score map+4个距离map+1个旋转角度 map？总得有一个啥网络来实现这个
* 损失函数是如何实现的？如何定义超参？默认值是多少？
* 预测的时候，对应的local aware NMS是如何实现的？
* 如何判断最后的预测效果，对应的评价标准是什么？可以参考CTPN的评价标准对比一下。
* 我们的标准，如何顺利的被转成east需要的样本格式？我们那340万张样本图片

# 开发日志

### 8.26

- 修改了加载我们的样本格式，目录结构为 data/train/images和data/train/labels
- 修改了train.sh中各类参数定义，完善了脚本
- 阅读了代码，增加了许多注释，方便理解代码
- 尝试在GPU上跑，笔记本上貌似跑不动，直接放弃，在GPU上跑，又出现读样本进程成为僵尸的问题
- train.sh上增加了调试和生产模式，以及停止功能

### 8.28

- 增加了日志系统，去除了prints...
- 增加了evaluator类，迁移自[evaluator.py](https://github.com/piginzoo/ctpn/blob/banjin-dev/utils/evaluate/evaluator.py)
- 实现了evaluator.validate方法，用来调用detect方法来实现lanms
- 增加了early stop机制，也是迁移自ctpn代码
- 重构了summary写入的内容和时机

### 9.3

- 调整了损失函数的lambda超参，是通过观察tensorboard中的各个子loss给出的，最终让3个loss接近
- 调整了批次大小从64-32，到最后的14，否则会OOM
- 修正了实际训练过程中出现的一些异常case，导致训练中断
- 修改了各个参数，大家work数量保证训练时候不用等待数据加载，训练一个step的周期目前降到1s内

# 原作者的日志，仅作保留

## EAST: An Efficient and Accurate Scene Text Detector

### Introduction
This is a tensorflow re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2).
The features are summarized blow:
+ Online demo
	+ http://east.zxytim.com/
	+ Result example: http://east.zxytim.com/?r=48e5020a-7b7f-11e7-b776-f23c91e0703e
	+ CAVEAT: There's only one cpu core on the demo server. Simultaneous access will degrade response time.
+ Only **RBOX** part is implemented.
+ A fast Locality-Aware NMS in C++ provided by the paper's author.
+ The pre-trained model provided achieves **80.83** F1-score on ICDAR 2015
	Incidental Scene Text Detection Challenge using only training images from ICDAR 2015 and 2013.
  see [here](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_samples&task=1&m=29855&gtv=1) for the detailed results.
+ Differences from original paper
	+ Use ResNet-50 rather than PVANET
	+ Use dice loss (optimize IoU of segmentation) rather than balanced cross entropy
	+ Use linear learning rate decay rather than staged learning rate decay
+ Speed on 720p (resolution of 1280x720) images:
	+ Now
		+ Graphic card: GTX 1080 Ti
		+ Network fprop: **~50 ms**
		+ NMS (C++): **~6ms**
		+ Overall: **~16 fps**
	+ Then
		+ Graphic card: K40
		+ Network fprop: ~150 ms
		+ NMS (python): ~300ms
		+ Overall: ~2 fps

Thanks for the author's ([@zxytim](https://github.com/zxytim)) help!
Please cite his [paper](https://arxiv.org/abs/1704.03155v2) if you find this useful.

### Contents
1. [Installation](#installation)
2. [Download](#download)
2. [Demo](#demo)
3. [Test](#train)
4. [Train](#test)
5. [Examples](#examples)

### Installation
1. Any version of tensorflow version > 1.0 should be ok.

### Download
1. Models trained on ICDAR 2013 (training set) + ICDAR 2015 (training set): [BaiduYun link](http://pan.baidu.com/s/1jHWDrYQ) [GoogleDrive](https://drive.google.com/open?id=0B3APw5BZJ67ETHNPaU9xUkVoV0U)
2. Resnet V1 50 provided by tensorflow slim: [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

### Train
If you want to train the model, you should provide the dataset path, in the dataset path, a separate gt text file should be provided for each image
and run

```
python multigpu_train.py --gpu_list=0 --input_size=512 --batch_size_per_gpu=14 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
--text_scale=512 --training_data_path=/data/ocr/icdar2015/ --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \
--pretrained_model_path=/tmp/resnet_v1_50.ckpt
```

If you have more than one gpu, you can pass gpu ids to gpu_list(like --gpu_list=0,1,2,3)

**Note: you should change the gt text file of icdar2015's filename to img_\*.txt instead of gt_img_\*.txt(or you can change the code in icdar.py), and some extra characters should be removed from the file.
See the examples in training_samples/**

### Demo
If you've downloaded the pre-trained model, you can setup a demo server by
```
python3 run_demo_server.py --checkpoint-path /tmp/east_icdar2015_resnet_v1_50_rbox/
```
Then open http://localhost:8769 for the web demo. Notice that the URL will change after you submitted an image.
Something like `?r=49647854-7ac2-11e7-8bb7-80000210fe80` appends and that makes the URL persistent.
As long as you are not deleting data in `static/results`, you can share your results to your friends using
the same URL.

URL for example below: http://east.zxytim.com/?r=48e5020a-7b7f-11e7-b776-f23c91e0703e
![web-demo](demo_images/web-demo.png)


### Test
run
```
python eval.py --test_data_path=/tmp/images/ --gpu_list=0 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
--output_dir=/tmp/
```

a text file will be then written to the output path.


### Examples
Here are some test examples on icdar2015, enjoy the beautiful text boxes!
![image_1](demo_images/img_2.jpg)
![image_2](demo_images/img_10.jpg)
![image_3](demo_images/img_14.jpg)
![image_4](demo_images/img_26.jpg)
![image_5](demo_images/img_75.jpg)

### Troubleshooting
+ How to compile lanms on Windows ?
  + See https://github.com/argman/EAST/issues/120

Please let me know if you encounter any issues(my email boostczc@gmail dot com).
