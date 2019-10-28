import cv2
import time
import os
import numpy as np
import tensorflow as tf
from utils import data_util
import lanms
import model
import logging
from utils.icdar import restore_rectangle


def init_flags():
    tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
    tf.app.flags.DEFINE_string('gpu_list', '0', '')
    tf.app.flags.DEFINE_string('output_dir', '/tmp/debug/images/', '')
    tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

FLAGS = tf.app.flags.FLAGS
logger = logging.getLogger("Eval")

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


count=0
def detect(score_map, geo_map,image,score_map_thresh=0.8,box_thresh=0.1, nms_thres=0.2,label=None):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4: # [1 400 288 1]=>[400 288]
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ] # (h, w, 5)
    # filter the score map，返回的是xy_text二维坐标数据
    xy_text = np.argwhere(score_map > score_map_thresh) #返回大于阈值的坐标，(x,y)二维的坐标,score map [400 288]
    xy_text = xy_text[:,:2]                             # 从[x,y,0]=>[x,y], [N,3]=>[N,2]
    # sort the text boxes via the y axis，argsort函数返回的是数组值从小到大的索引值
    xy_text = xy_text[np.argsort(xy_text[:,0])]         #返回还是二维坐标数据组，只不过是按照x排序了，argsort是从小到大
    logger.debug("从%r中挑选置信度大于0.8的点，得到%r",score_map.shape,xy_text.shape)


    global count
    count+=1
    if count>50: count=0


    # restore
    start = time.time()
    # ::-1 把数组倒过来
    # 为何最表要乘以4？ !!!明白了！因为按照那个预测，得到的只是原图1/4大小的坐标，所以，要乘以4
    # 可是为何是1/4，这就要从resnet50说起，我们取的是conv_2x/是conv_3x/是conv_4x/是conv_5x的输出，注意conv_2x，他输出的时候，大小已经是原图的1/4啦。
    # 参考这个：https://www.cnblogs.com/ymjyqsx/p/7587033.html
    text_box_restored = restore_rectangle(xy_text[:,::-1]*4,# 为何要乘以4，上面解释了
                                          geo_map[           # 把这些点对应的值拿出来，geo_map:(h, w, 5)
                                            xy_text[:, 0],   # x坐标
                                            xy_text[:, 1],   # y坐标，注意，这个是原图1/4的大小对应的坐标
                                            :
                                          ]) # N*4*2
    logger.debug("从预测得到了%d个框:%r",text_box_restored.shape[0],text_box_restored.shape)

    # 返回的box是 8个点的坐标值+1个是否文字的置信度
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]].reshape(-1)
    logger.debug("从geo map还原矩形框的时间：%d",time.time() - start)
    debug(image.copy(), boxes, "before_nms.jpg",count)

    # nms part
    start = time.time()
    # 终于开始做激动人心的local aware NMS了！
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    logger.debug("NMS的完成，结果：%r，时间：%d",boxes.shape,time.time() - start)
    debug(image.copy(),boxes,"nms_merged.jpg",count)

    if boxes.shape[0] == 0:
        logger.warning("经过NMS合并，结果居然为0个框")
        return None

    # here we filter some low score boxes by the average score map,
    # this is different from the orginal paper
    # 这里需要解释一下，实际上是用nms过滤后的点，做了一个mask，这个mask是啥啊，就是那些框，框出来这些点来
    # 然后这些点每个点都有自己的socore把，恩，平均他们一下，作为这个点的score值
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)#<----注意下一个细节，结果都除以4了，又，之前乘过4，诡异哈？
        boxes[i, 8] = cv2.mean(score_map, mask)[0] #<------解释上一行，因为score_map还是原图1/4的大小
    boxes = boxes[boxes[:, 8] > box_thresh] # 把那些置信度低的去掉再
    debug(image.copy(), boxes, "filter_low_score.jpg",count,label=label)

    logger.debug("处理后，得到检测框：%r",boxes.shape)
    return boxes

# 调试50张（循环覆盖），可用使用python simple-http 8080（python自带的）启动一个简单的web服务器，来调试
def debug(image,boxes,name,index,label=None):
    for i, box in enumerate(boxes):
        cv2.polylines(image, box[:8].reshape((-1, 4, 2)).astype(np.int32),isClosed=True,color=(0,0,255),thickness=1) #red
        
    # 如果标签不为空，画之
    if label is not None:
        for i, lbox in enumerate(label):
            cv2.polylines(image, label[:8].reshape((-1, 4, 2)).astype(np.int32),isClosed=True,color=(255,0,0),thickness=1) #green

    cv2.imwrite("debug/{}_{}".format(index,name),image)


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list


    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # [1/4H, 1/4W,1], [1/4*h,1/4*w,4]
        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.model_path)
            model_path = os.path.join(FLAGS.model_path, os.path.basename(ckpt_state.model_model_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                start_time = time.time()
                # 调整图像为32的倍数，但是基本上保持原图大小

                im_resized, (ratio_h, ratio_w) = data_util.resize_image(im)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start

                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))

                # save to file
                if boxes is not None:
                    res_file = os.path.join(
                        FLAGS.output_dir,
                        '{}.txt'.format(
                            os.path.basename(im_fn).split('.')[0]))

                    with open(res_file, 'w') as f:
                        for box in boxes:
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                            ))
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])

if __name__ == '__main__':
    init_flags()
    tf.app.run()
