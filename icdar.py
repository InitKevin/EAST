# coding:utf-8
import glob
import csv
import cv2
import time
import os
import numpy as np
import pdb
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon

import tensorflow as tf

from data_util import GeneratorEnqueuer

# tf中定义了 tf.app.flags.FLAGS ，用于接受从终端传入的命令行参数，
# “DEFINE_xxx”函数带3个参数，分别是变量名称，默认值，用法描述
tf.app.flags.DEFINE_string('training_data_path', '/data/ocr/icdar2015/',
                           'training dataset to use')
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
tf.app.flags.DEFINE_string('geometry', 'RBOX',
                           'which geometry to generate, RBOX or QUAD')

FLAGS = tf.app.flags.FLAGS


# data/images
def get_images():
    files = []
    image_dir = os.path.join(FLAGS.training_data_path,"images")
    print("尝试加载目录中的图像：",image_dir)
    for ext in ['jpg', 'png', 'jpeg', 'JPG','png']:
        patten = os.path.join(image_dir, '*.{}'.format(ext))
        print("检索模式：",patten)
        files.extend(glob.glob(patten))

    if FLAGS.debug:
        print("调试模式，仅加载10张图片")
        _len = min(len(files),10)
        files = files[:_len]

    print("加载完毕%d图像" % len(files))
    return files


def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

# 计算一个框的面积, poly shape:[4,2]，4是4个点，2是x和y
def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    # 为何这么诡异的公式呢，参考这个：https://www.cnblogs.com/zzcpy/p/10524348.html
    # Area = (1/2) * |x1y2 + x2y3 + x3y4 + x4y1 - x2y1 - x3y2 - x4y3 - x1y4 |
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]), # (x1-x2)*(y1+y0)
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]), # (x2-x1)*(y2+y1)
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]), # (x3-x2)*(y3+y2)
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])  # (x0-x3)*(y0+y3)
    ]
    return np.sum(edge)/2.

# 剔除越界的，面积小于1的框
def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        print("polys shape is 0:",polys.shape)
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1) # polys shape:[N,4,2]，4是4个点，2是x和y
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1) # 这个是怕这个图的标注越界？可是怎么可能会越界呢？不懂？？？

    validated_polys = []
    validated_tags = []
    print("begin to process poly & tag")
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:# 如果面积小于1
            # print poly
            print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    print("finish polys process:%d,%d" % (len(validated_polys),len(validated_tags)))
    return np.array(validated_polys), np.array(validated_tags)

# 这个函数真心没看懂，远航君给讲了一下，豁然开朗
# 就是把文字框往x，y轴做投影，形成两个数组h_array，w_array，这俩数组，凡是在文本框投影的地方，都是1，剩余的地方都是0
# 然后，他会找那些0的地方，随机做切分，这样可以切出一个个包含了文本框的子图来
def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''

    # pad_h就是高的1/10，pad_w是宽的1/10
    h, w, _ = im.shape
    pad_h = h//10 #//是整除，就是缩小十倍区中 30//7 = 4
    pad_w = w//10
    # ？？？在干嘛，这个h_array，w_array是俩数组，一维的，比w和h大一点。为何要做一些padding，不懂。
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)

    # 这个就是把文本框向x，y，也就是宽、高方向上投影
    # 这样w_array，h_array被分割成一段段的，是文本框投影的地方都是1，   000001111111000011100000000111111100000001111111100000 这个样子
    for poly in polys:
        # poly.shape=[4,2]
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx+pad_w:maxx+pad_w] = 1 # 在x上投影，框投影的地方，都是1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny+pad_h:maxy+pad_h] = 1

    # 这步得到所有的0的index，索引
    # np.where得到的是下标
    # b = array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0])
    # >> > np.where(b == 0)
    # (array([0, 1, 2, 6, 7, 12, 13, 14, 15, 18]),) <------ 注意这个结果要[0]一下
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0] # 为何要[0]一下，是因为np.where，返回的[0]才是下标
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    print("maxretry:",max_tries)
    for i in range(max_tries):

        # xx是随机找到2个x的坐标，x坐标是从w_array里面找到的为0值的数组下标，就是远航君说的随机找2个值为0点的意思，我有个问题，如果2个值挨着怎么办？都是一个区间里
        xx = np.random.choice(w_axis, size=2) # 比如得到[36,58]
        xmin = np.min(xx) - pad_w # 都偏移宽的1/0？为何？ ,
        xmax = np.max(xx) - pad_w # 得到[16,38]，假设图像是200x200，xmin - (200/10), xmax - (200/10)
        xmin = np.clip(xmin, 0, w-1) # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
        xmax = np.clip(xmax, 0, w-1) # 参考：https://blog.csdn.net/qq1483661204/article/details/78150203

        #
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1) # 诡异？ ymin是一个数啊，clip后，实际上我理解就是取整到[0,h-1]之间，把这个数
        ymax = np.clip(ymax, 0, h-1) # 比如 np.clip(-33,0,219)=> 0

        if xmax - xmin < FLAGS.min_crop_side_ratio*w or ymax - ymin < FLAGS.min_crop_side_ratio*h:
            # area too small
            continue

        # !!! 这个才是重点，剪裁了，裁出一块区域，包含着文本框
        # polys.shape=>[文本框个数, 4, 2]，4是4个点，2是x&y
        # poly_axis_in_area得到的实际上是一个整张图的包含了切出来的区域的掩码(就是true/false)
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & \
                                (polys[:, :, 0] <= xmax) & \
                                (polys[:, :, 1] >= ymin) & \
                                (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []

        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                print("crop_background")
                return im[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        # 把子图切出来
        im = im[ymin:ymax+1, xmin:xmax+1, :]
        # 得到切出来的子图中的那些框和tags
        polys = polys[selected_polys]
        tags  = tags [selected_polys]
        # 别忘了，要调整一下切出来的子图的坐标
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        print("crop return:", im.shape, polys.shape, tags.shape)
        return im, polys, tags

    return im, polys, tags

# 不想细看了，大概也能理解，就是缩小1/3后的那个框的4个坐标，返回的是
def shrink_poly(poly, r):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                    np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


# p3 到p1-p2组成的直线的距离
def point_dist_to_line(p1, p2,      p3):
    # compute the distance from p3 to p1-p2
    # cross:向量积，数学中又称外积、叉积,运算结果是一个向量而不是一个标量。并且两个向量的叉积与这两个向量和垂直。模长是|a|*|b|*cos夹角，方向上右手法则
    # 叉乘的二维的一个含义是，"在二维中，两个向量的向量积的模的绝对值等于由这两天向量组成的平行四边形的面积"
    # np.linalg.norm(np.cross(p2 - p1, p1 - p3)) 就是p1p3,p1p2夹成的平行四边形的面积
    # 除以
    # np.linalg.norm(p2 - p1)，是p1p2的长度，
    # 得到的，就是P3到p1,p2组成的的距离，
    # 你可以自己画一个平行四边形，面积是 底x高，现在面积已知，底就是p1p2，那高，就是p3到p1p2的距离
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

# 求拟合曲线的k和b
def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        # https://blog.csdn.net/vola9527/article/details/40402189
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]

# 找到交叉点，line1,line都是用k、b表示的
# line：[k,0/1,b]
def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]: #k,也就是斜率一样，两条线平行啊，没交叉点啊
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0: # 都是平行于x轴，没交叉点啊
        print('Cross point does not exist')
        return None
    if line1[1] == 0: #??? 中间的一位是0的是啥含义来着？忘了
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1-b2)/(k1-k2) # 求解交叉点
        y = k1*x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
    return verticle


def rectangle_from_parallelogram(poly):
    '''
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    '''
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1-p0, p3-p0)/(np.linalg.norm(p0-p1) * np.linalg.norm(p3-p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0-p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0-p1) > np.linalg.norm(p0-p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点 - find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle/np.pi * 180 > 45:
            # 这个点为p2 - this point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
        else:
            # 这个点为p3 - this point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :],
                                  new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :],
                                  new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))

    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :],
                                  new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :],
                                  new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))

    return np.concatenate([new_p_0, new_p_1])


def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)

# this is core method for samples making
# 这个应该是最最核心的样本数据准备的过程了,im_size现在是512x512
def generate_rbox(im_size, polys, tags):
    h, w = im_size

    print("开始生成rbox数据")

    # 初始化3个蒙版，都是512x512
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)

    # polys.shape => [框数，4，2]
    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]

        r = [None, None, None, None]
        for i in range(4):
            # np.linalg.norm(求范数)：https://blog.csdn.net/hqh131360239/article/details/79061535
            # 这里求范数，就是在求文本框点之间的距离，r得到的是挨着我的点里面最小的那个距离，从第一个开始
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        # score map
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)
        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)

        # if the poly is too small, then ignore it during training，如果太小，就不参与训练了
        # 终于明白training_mask的妙用了，就是控制那些点不参与训练
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < FLAGS.min_text_size:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        # argwhere返回满足条件的数组元的索引
        # 啥意思？poly_mask == (poly_idx + 1)这个条件不理解？我理解这句话没啥用啊？？？
        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))

        # if geometry == 'RBOX':
        # 对任意两个顶点的组合生成一个平行四边形 - generate a parallelogram for any combination of two vertices
        fitted_parallelograms = []
        for i in range(4):
            # 4个点
            p0 = poly[i] # poly.shape [4,2]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]


            '''
                                 p0
                                 #
                               #####
                            ##########
                         ##############
                      ###################
                   ########################
              p3 ###########################
                     #########################
                         ####################### p1
                             ##################  
                                 ############
                                     ######
                                        #
                                       p2
            '''
            # 求拟合曲线的k和b，返回的是[k,0/1,b]
            edge          = fit_line([p0[0], p1[0]], [p0[1], p1[1]]) #左上，右上 0,1
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]]) #左上，左下 0,3
            forward_edge  = fit_line([p1[0], p2[0]], [p1[1], p2[1]]) #右上，右下 1,2

            # 看p2到p0p1的距离 > p3到p0p1的距离
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                # 平行线经过p2 - parallel lines through p2，对，就是这个意思
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    #
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # 经过p3 - after p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]

            # move forward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
        # sort thie polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectange = rectangle_from_parallelogram(parallelogram)
        rectange, rotate_angle = sort_rectangle(rectange)

        p0_rect, p1_rect, p2_rect, p3_rect = rectange
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            # right
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            # down
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            # left
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            # angle
            geo_map[y, x, 4] = rotate_angle
    return score_map, geo_map, training_mask

'''
返回的是这4个data
input_images: data[0],
input_score_maps: data[2],
input_geo_maps: data[3],
input_training_masks: data[4]})
'''
def generator(input_size=512, batch_size=32,
              background_ratio=3./8,
              random_scale=np.array([0.5, 1, 2.0, 3.0]),
              vis=False):
    # 获得训练集路径下所有图片名字
    image_list = np.array(get_images())
    # index：总样本数
    index = np.arange(0, image_list.shape[0])
    # pdb.set_trace()
    while True:
        np.random.shuffle(index)
        images = []
        image_fns = []
        score_maps = []
        geo_maps = []
        training_masks = []
        count=0
        for i in index:
            try:
                # 读取图片
                im_fn = image_list[i]
                _image = cv2.imread(im_fn)
                print ("image name：",im_fn)
                h, w, _ = _image.shape

                # 读取标签txt
                label_name = os.path.splitext(os.path.basename(im_fn))[0]
                label_dir = os.path.join(FLAGS.training_data_path, "labels")
                txt_fn = os.path.join(label_dir,label_name+".txt")


                if not os.path.exists(txt_fn):
                    print('text file {} does not exists'.format(txt_fn))
                    continue
                print("成功加载标签文件：", txt_fn)

                # 读出对应label文档中的内容
                # text_polys：样本中文字坐标:[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]，text_polys shape:[N,4,2]，4是4个点，2是x和y
                # text_tags：文字框内容是否可辨识
                # 例子：
                #   377, 117, 463, 117, 465, 130, 378, 130, GenaxisTheatre
                #   493, 115, 519, 115, 519, 131, 493, 131, [06]
                text_polys, text_tags = load_annoataion(txt_fn) #4点标注，是不规则四边形，而不是一个旋转的矩形

                # 保存其中的有效标签框，并修正文本框坐标溢出边界现象，多边形面积>1
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

                # 这段是在resize图像，不知道为何这样做，是为了做样本增强么？resize4个选项，相当于数据多了4倍。不过，我感觉没必要
                # if text_polys.shape[0] == 0:
                #     continue
                # random scale this image，为何要随机resize一下，做样本增强么？，random_scale=[0.5, 1, 2.0, 3.0]
                rd_scale = np.random.choice(random_scale)
                im = cv2.resize(_image, dsize=None, fx=rd_scale, fy=rd_scale)
                text_polys *= rd_scale

                # random crop a area from image
                # np.random.rand() < background_ratio
                if 1 < background_ratio: #background_ratio=3/8，background_ratio是个啥东东？

                    # crop background
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                    if text_polys.shape[0] > 0:
                        print("cannot find background",im_fn)
                        continue

                    # pad and resize image,最终图片变成512x512，图像不变形，padding补足
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = cv2.resize(im_padded, dsize=(input_size, input_size)) # input_size = 512

                    score_map        = np.zeros((input_size, input_size), dtype=np.uint8)
                    geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
                    geo_map          = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                    training_mask    = np.ones((input_size, input_size), dtype=np.uint8)

                else: # > 3/8

                    # 这个是切出一个子图来，就用这个子图做训练了，我理解，还是跟数据增强差不多，可以大幅的提高图像的利用率啊
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                    if text_polys.shape[0] == 0:
                        print("text_polys shapie is 0:", im_fn.shape,text_polys.shape)
                        continue

                    h, w, _ = im.shape

                    # 这步操作就是最终在不变形的情况下，把子图resize成512x512，空白处padding 0
                    # pad the image to the training input size or the longer side of image
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size]) # 就是看子图的宽、高、512里面谁最大，选谁做新图的长和宽
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy() # 把切出来的那个子图拷贝进去
                    im = im_padded
                    # resize the image to input size
                    new_h, new_w, _ = im.shape
                    resize_h = input_size # 强制改成512了啊！
                    resize_w = input_size
                    im = cv2.resize(im, dsize=(resize_w, resize_h))
                    print("resize图像为512：",im.shape)

                    # 把标注坐标缩放
                    resize_ratio_3_x = resize_w/float(new_w)
                    resize_ratio_3_y = resize_h/float(new_h)
                    text_polys[:, :, 0] *= resize_ratio_3_x
                    text_polys[:, :, 1] *= resize_ratio_3_y
                    new_h, new_w, _ = im.shape

                    score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)
                    print("RBox数据生成完毕(score_map, geo_map, training_mask)：",score_map.shape,geo_map.shape,training_mask.shape)

                if vis:
                    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
                    # axs[0].imshow(im[:, :, ::-1])
                    # axs[0].set_xticks([])
                    # axs[0].set_yticks([])
                    # for poly in text_polys:
                    #     poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                    #     poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                    #     axs[0].add_artist(Patches.Polygon(
                    #         poly * 4, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                    #     axs[0].text(poly[0, 0] * 4, poly[0, 1] * 4, '{:.0f}-{:.0f}'.format(poly_h * 4, poly_w * 4),
                    #                    color='purple')
                    # axs[1].imshow(score_map)
                    # axs[1].set_xticks([])
                    # axs[1].set_yticks([])
                    axs[0, 0].imshow(im[:, :, ::-1])
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    for poly in text_polys:
                        poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                        poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                        axs[0, 0].add_artist(Patches.Polygon(
                            poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                        axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
                    axs[0, 1].imshow(score_map[::, ::])
                    axs[0, 1].set_xticks([])
                    axs[0, 1].set_yticks([])
                    axs[1, 0].imshow(geo_map[::, ::, 0])
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    axs[1, 1].imshow(geo_map[::, ::, 1])
                    axs[1, 1].set_xticks([])
                    axs[1, 1].set_yticks([])
                    axs[2, 0].imshow(geo_map[::, ::, 2])
                    axs[2, 0].set_xticks([])
                    axs[2, 0].set_yticks([])
                    axs[2, 1].imshow(training_mask[::, ::])
                    axs[2, 1].set_xticks([])
                    axs[2, 1].set_yticks([])
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                images.append(im[:, :, ::-1].astype(np.float32))
                image_fns.append(im_fn)
                score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                count+=1

                # 凑过了batch_size，就被这批数据yield出去
                if len(images) == batch_size:
                    yield images, image_fns, score_maps, geo_maps, training_masks
                    images = []
                    image_fns = []
                    score_maps = []
                    geo_maps = []
                    training_masks = []
            except BaseException as e:
                print("Error happened:",str(e))
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    print ("get a generator output:" , generator_output)
                    break
                else:
                    #print("queue is empty, which cause we are wating....")
                    time.sleep(1)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()



if __name__ == '__main__':
    pass
