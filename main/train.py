import time,os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from utils import log_util
from nets import model
from utils import icdar
from utils.early_stop import EarlyStop
from utils import evaluator
import logging
import datetime

FLAGS = tf.app.flags.FLAGS
logger = logging.getLogger(__name__)

def tower_loss(images, score_maps, geo_maps, training_masks, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        # 模型定义！！！，f_score是和原图大小一样的是否是前景的概率图， f_geometry是5张图，4张是上下左右值，1张是旋转角度值
        f_score, f_geometry = model.model(images, is_training=True)

    #              def loss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo,training_mask):
    model_loss = model.loss(score_maps, f_score,    geo_maps,   f_geometry,training_masks)

    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    tf.summary.image('input', images)
    tf.summary.image('score_map', score_maps)
    tf.summary.image('score_map_pred', f_score * 255)
    tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
    tf.summary.image('geo_map_#0_pred', f_geometry[:, :, :, 0:1])
    tf.summary.image('geo_map_#1_pred', f_geometry[:, :, :, 0:1])
    tf.summary.image('training_masks', training_masks)
    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss,f_score, f_geometry


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads



def create_summary_writer():
    # 按照日期，一天生成一个Summary/Tboard数据目录
    # Set tf summary
    if not os.path.exists(FLAGS.tboard_dir): os.makedirs(FLAGS.tboard_dir)
    today = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = os.path.join(FLAGS.tboard_dir,today)
    summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())
    return summary_writer


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    gpus = list(range(len(FLAGS.gpu_list.split(','))))

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    if FLAGS.geometry == 'RBOX':
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    else:
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)

    # 这个是定义召回率、精确度和F1
    v_recall = tf.Variable(0.001, trainable=False)
    v_precision = tf.Variable(0.001, trainable=False)
    v_f1 = tf.Variable(0.001, trainable=False)
    tf.summary.scalar("Recall",v_recall)
    tf.summary.scalar("Precision",v_precision)
    tf.summary.scalar("F1",v_f1)

    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

    # split
    input_images_split = tf.split(input_images, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]

                # 模型定义！！！
                #                                         def tower_loss(images,score_maps, geo_maps, training_masks, reuse_variables=None):
                total_loss, model_loss,f_score, f_geometry  = tower_loss(iis,   isms,       igms,     itms,           reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = create_summary_writer()

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        # pretrained_model_path实际上是resnet50的pretrain模型
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
        logger.debug("成功加载resnet预训练模型：%s",FLAGS.pretrained_model_path)

    early_stop = EarlyStop(FLAGS.early_stop)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.model_name!="None":
            model_meta_file_path = os.path.join(FLAGS.model_path,FLAGS.model_name) + ".meta"
            logger.debug('尝试从[%s]中恢复训练到半截的模型',model_meta_file_path)
            if not os.path.exists(model_meta_file_path):
                logger.error("模型路径不存在，训练终止")
                exit(-1)
            # 这个是之前的checkpoint模型，可以半截接着训练
            ckpt = tf.train.latest_checkpoint(FLAGS.model_path)
            saver.restore(sess, ckpt)
            logger.debug('预训练模型[%s]加载完毕，可以继续训练了', model_meta_file_path)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)
            logger.debug("从头开始训练...，model_name=%s",FLAGS.model_name)

        data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size,
                                         type="train")

        validate_data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size,
                                         type="validate")

        # 开始训练啦！
        for step in range(FLAGS.max_steps):

            # 取出一个batch的数据
            start = time.time()
            data = next(data_generator)
            logger.debug("[训练] 第%d步，加载了一批(%d)图片(%f秒)，准备训练...",step,FLAGS.batch_size,(time.time()-start))

            # 训练他们
            run_start = time.time()
            ml, tl, _ ,summary_str = sess.run([model_loss,
                                  total_loss,
                                  train_op,
                                  summary_op],
                                 feed_dict={
                                    input_images: data[0],
                                    input_score_maps: data[2],
                                    input_geo_maps: data[3],
                                    input_training_masks: data[4]})
            if np.isnan(tl):
                logger.debug('Loss diverged, stop training')
                break

            logger.debug("[训练] 跑完批次的梯度下降,耗时:%f",time.time()-run_start)


            # if step % FLAGS.validate_steps == 0:
            #     logger.debug("保存checkpoint:",FLAGS.model_path + 'model.ckpt')
            #     saver.save(sess, FLAGS.model_path + 'model.ckpt', global_step=global_step)
            # 默认是1000步，validate一下
            if step!=0 and step % FLAGS.validate_steps == 0:
                precision, recall, f1 = evaluator.validate(sess,
                                                           FLAGS.validate_batch_num,
                                                           FLAGS.batch_size,
                                                           validate_data_generator,
                                                           f_score,
                                                           f_geometry,
                                                           input_images)
                # 更新三个scalar tensor
                sess.run([tf.assign(v_f1, f1),
                          tf.assign(v_recall, recall),
                          tf.assign(v_precision, precision)])

                logger.debug("评估完毕:在第%d步,F1:%f,Recall:%f,Precision:%f", step,f1,recall,precision)
                if is_need_early_stop(early_stop, f1, saver, sess, step): break  # 用负的编辑距离

            if step!=0 and step % FLAGS.save_summary_steps == 0:
                logger.debug("写入summary文件，第%d步",step)
                summary_writer.add_summary(summary_str, global_step=step)
                avg_time_per_step = (time.time() - start)/FLAGS.save_summary_steps
                avg_examples_per_second = (FLAGS.save_summary_steps * FLAGS.batch_size * len(gpus))/(time.time() - start)
                start = time.time()
                logger.debug('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                    step, ml, tl, avg_time_per_step, avg_examples_per_second))


            logger.debug("[训练] 第%d步结束，整体耗时(包括加载数据):%f",step,(time.time()-start))


def is_need_early_stop(early_stop,value,saver,sess,step):
    decision = early_stop.decide(value)

    if decision == EarlyStop.CONTINUE:
        logger.info("新Value值比最好的要小，继续训练...")
        return False

    if decision == EarlyStop.BEST:
        logger.info("新Value值[%f]大于过去最好的Value值，早停计数器重置，并保存模型", value)
        saver.save(sess, os.path.join(FLAGS.model_path,'model.ckpt'), global_step=step)
        return False

    if decision == EarlyStop.STOP:
        logger.warning("超过早停最大次数，也尝试了多次学习率Decay，无法在提高：第%d次，训练提前结束", step)
        return True

    logger.error("无法识别的EarlyStop结果：%r",decision)
    return True

def init_flags():
    tf.app.flags.DEFINE_string('name', 'east', '')
    tf.app.flags.DEFINE_integer('input_size', 512, '')
    tf.app.flags.DEFINE_integer('batch_size', 32, '')
    tf.app.flags.DEFINE_integer('num_readers', 16, '')
    tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
    tf.app.flags.DEFINE_integer('max_steps', 100000, '')
    tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
    tf.app.flags.DEFINE_string('gpu_list', '0', '')
    tf.app.flags.DEFINE_boolean('debug',False,'')
    tf.app.flags.DEFINE_string('model_path', '', '')
    tf.app.flags.DEFINE_string('model_name', 'None', '')
    tf.app.flags.DEFINE_string('tboard_dir', '', '')
    tf.app.flags.DEFINE_integer('validate_steps', 1000, '')
    tf.app.flags.DEFINE_integer('validate_batch_num', 30, '') # 一共检查多少个批次
    tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
    tf.app.flags.DEFINE_integer('early_stop', 100, '')
    tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
    tf.app.flags.DEFINE_string('training_data_path', '', '')
    tf.app.flags.DEFINE_string('validate_data_path', '', '')
    tf.app.flags.DEFINE_integer('lambda_AABB',1000 , '')
    tf.app.flags.DEFINE_integer('lambda_theta',100000 , '')
    tf.app.flags.DEFINE_integer('lambda_score', 1, '')


    # tf中定义了 tf.app.flags.FLAGS ，用于接受从终端传入的命令行参数，
    # “DEFINE_xxx”函数带3个参数，分别是变量名称，默认值，用法描述
    tf.app.flags.DEFINE_integer('max_image_large_side', 1280,
                                'max image size of training')
    tf.app.flags.DEFINE_integer('max_text_size', 800,
                                'if the text in the input image is bigger than this, then we resize'
                                'the image according to this')
    tf.app.flags.DEFINE_integer('min_text_size', 10,
                                'if the text size is smaller than this, we ignore it during training')
    tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.1,
                              'when doing random crop from input image, the'
                              'min length of min(H, W')
    tf.app.flags.DEFINE_string('geometry', 'RBOX','which geometry to generate, RBOX or QUAD')

if __name__ == '__main__':
    init_flags()
    log_util.init_logger()
    tf.app.run()
