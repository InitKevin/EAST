# coding=utf-8
import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets import resnet_v1

FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])# 使用双线性插值调整images为size

# 定义一个图像做标准化的函数
def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    # x.get_shape()，只有tensor才可以使用这种方法，返回的是一个元组，需要通过as_list()的操作转换成list
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    # 函数用途简单说就是把一个张量划分成几个子张量,value：准备切分的张量,num_or_size_splits:准备切成几份,axis:准备在第几个维度上进行切割
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)# 把多个array沿着第4个维度接在一起


def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    # 对RGB像素做标准化，即减去均值
    images = mean_image_subtraction(images)
    # 先将图片经过resnet_v1网络，得到resnet_v1的全部stage的输出，存在end_points里面
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    print("网络定义完毕，返回的end_points:",end_points)

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997, # 衰减系数
        'epsilon': 1e-5, # 在x的方差中添加了一个小的浮点数，避免被零除
        'scale': True, # 如果为True，则乘以gamma。如果为False，gamma则不使用。当下一层是线性的时（例如nn.relu），由于缩放可以由下一层完成，所以可以禁用该层。
        'is_training': is_training # bool值，用于指定操作是用于训练还是推断
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu, # 激活函数
                            normalizer_fn=slim.batch_norm, # 正则化函数
                            normalizer_params=batch_norm_params, # slim.batch_norm中的参数，以字典形式表示
                            weights_regularizer=slim.l2_regularizer(weight_decay)): # 权重的正则化器
            # 取2，3，4，5次池化后的输出
            f = [end_points['pool5'],
                 end_points['pool4'],
                 end_points['pool3'],
                 end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                # 由网络结构图可知h0=f0
                if i == 0:
                    h[i] = f[i]
                # 对其他的hi有，hi = conv (concat (fi, unpool (hi-1) ) )
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                # 由网络结构可知，对于h0，h1，h2都要先经过unpool在与fi进行叠加
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            # score map
            # g[3]就是最后得到的upool的合并完的feature map了，然后做了一个1x1的卷积，channel是1，得到了啥？
            # 得到了一张1通道的图，跟原图大小一样，每个"像素"的值都是一个概率，是一个"概率"的图
            #
            # slim.conv2d(inputs,num_outputs(output channel),kernel_size,stride=1,padding='SAME',
            F_score = slim.conv2d(g[3],1,1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 参数注释：前三个参数依次为网络的输入，输出的通道，卷积核大小，activation_fn : 激活函数，默认是nn.relu，normalizer_fn : 正则化函数，默认为None

            # 4 channel of axis aligned bbox and 1 channel rotation angle
            # text boxes
            geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            # text rotation
            angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            # 这里将坐标与角度信息合并输出
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

            # 乖乖：都是通过卷积得到的啊，最后得到了啥：F_score（1张）、geo_map（4张）、angle_map（1张），恩，张就是指图，跟原图带下一样的伪图

    return F_score, F_geometry #1+5channel的图像


def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    coefficient:协同因素
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + \
            tf.reduce_sum(y_pred_cls * training_mask) + \
            eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss



def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice(筛子) loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    # score交叉熵
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> 距离top的长度, d2->right, d3->bottom, d4->left, theta->倾斜角度
    d1_gt,   d2_gt,   d3_gt,   d4_gt,   theta_gt   = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)

    # 为何是+，噢，是因为，预测和标注的，不是坐标，而是宽高
    area_gt   = (d1_gt   + d3_gt  ) * (d2_gt   + d4_gt  )
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)

    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)

    # 计算IoU
    area_intersect = w_union * h_union # 相交面积
    area_union = area_gt + area_pred - area_intersect
    # -log(IoU)
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))

    # 角度误差函数
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))

    # 加权和得到geo loss
    L_g = L_AABB + 20 * L_theta

    # 考虑training_mask，背景像素不参与误差计算
    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
