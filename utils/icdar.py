# coding:utf-8
import glob
import csv
import cv2
import time
import os
import numpy as np
from shapely.geometry import Polygon
import tensorflow as tf
from utils.data_util import GeneratorEnqueuer
import logging

logger = logging.getLogger("Generator")
FLAGS = tf.app.flags.FLAGS

def get_images(dir):
    files = []
    # print(dir)
    image_dir = os.path.join(dir,"images")
    # logger.debug("进程[%d]尝试加载目录中的图像：%s",os.getpid(),image_dir)
    for ext in ['jpg', 'png', 'jpeg', 'JPG','png']:
        patten = os.path.join(image_dir, '*.{}'.format(ext))
        # logger.debug("检索模式：%s",patten)
        files.extend(glob.glob(patten))

    if FLAGS.debug:
        logger.debug("调试模式，仅加载10张图片")
        _len = min(len(files),10)
        files = files[:_len]

    # logger.debug("进程[%d]加载完毕%d张图像路径..." , os.getpid(),len(files))
    return files
# data/images


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
    #
    # ============= 换一个思路 ===============
    # 其实就是个梯形计算公式
    # * 纵坐标看成底
    # * 横坐标看成高
    # 面积S1 = 就是 （上底 + 下底）* 高 * 1/2
    # 最后再 +、- 得到最终poly面积
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
        logger.debug("polys shape is 0:%r",polys.shape)
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1) # polys shape:[N,4,2]，4是4个点，2是x和y
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1) # 这个是怕这个图的标注越界？可是怎么可能会越界呢？不懂？？？

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:# 如果面积小于1
            # logger.debug poly
            logger.debug('文字区域面积小于1')
            continue
        if p_area > 0:
            logger.debug('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    # logger.debug("finish polys process:%d,%d" % (len(validated_polys),len(validated_tags)))
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

        if xmax - xmin < FLAGS.min_crop_side_ratio*w or \
           ymax - ymin < FLAGS.min_crop_side_ratio*h:
           # area too small
           continue

        # !!! 这个才是重点，剪裁了，裁出一块区域，包含着文本框
        # polys.shape=>[文本框个数, 4, 2]，4是4个点，2是x&y
        # poly_axis_in_area得到的实际上是一个整张图的包含了切出来的区域的"掩码"(就是true/false)
        if polys.shape[0] != 0: # 至少有1个框
            # 这句话就是表达，你们这些框，谁在我选出的这个区域里，4行表示每个点的x或者y，满足 xmin<x<xmax 且 ymin<y<ymax
            # polys[:, :, 0] >= xmin,这个会从【50，50，3】变成【50，50】的true false矩阵
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & \
                                (polys[:, :, 0] <= xmax) & \
                                (polys[:, :, 1] >= ymin) & \
                                (polys[:, :, 1] <= ymax)
            # 上面这句话是判断某个点在区域里，
            # [N,4,2]，下面这步，是说，框的4个点都在区域里，
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []

        if len(selected_polys) == 0:#说明你切出来的部分，是不包含任何框的
            # no text in this area
            if crop_background: #如果标志就是说，我要没有框的背景的话，就把这个没有框的图像返回去，selected_polys是个空数组哈
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
        # logger.debug("crop return:", im.shape, polys.shape, tags.shape)
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
        # logger.debug poly
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
    # cross:向量积，数学中又称外积、叉积,运算结果是一个向量而不是一个标量。并且两个向量的叉积与这两个向量和垂直。模长是|a|*|b|*sin夹角，方向上右手法则
    # 叉乘的二维的一个含义是，"在二维中，两个向量的向量积的模的绝对值等于由这两天向量组成的平行四边形的面积"
    # np.linalg.norm(np.cross(p2 - p1, p1 - p3)) 就是p1p3,p1p2夹成的平行四边形的面积
    # 除以
    # np.linalg.norm(p2 - p1)，是p1p2的长度，
    # 得到的，就是P3到p1,p2组成的的距离，
    # 你可以自己画一个平行四边形，面积是 底x高，现在面积已知，底就是p1p2，那高，就是p3到p1p2的距离
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

# 根据p1和p2来获取拟合的曲线 ax+by+c=0, 自己没看懂，还是田老师的帮助下才弄清楚
# param:
#    p1 -- x坐标集合
#    p2 -- y坐标集合
#    如果有两个点（x1, y1）和(x2, y2)，那么p1=[x1, x2], p2=[y1, y2]
# return:
#    [a, b, c]
def fit_line(p1, p2):
    # fit a line ax+by+c = 0 ,
    if p1[0] == p1[1]:
        # 两个点的x坐标相同，说明该直行是垂直于X轴
        return [1., 0., -p1[0]] # 1x + 0y + -p1(x) = 0
    else:
        # https://blog.csdn.net/vola9527/article/details/40402189
        # 得到过两点的，deg：自由度：为多项式最高次幂，结果为多项式的各个系数
        # 这个其实就是找到2个点的斜率和截距：k/b
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]

# 找到交叉点，line1,line都是用k、b表示的
# line：[k,0/1,b]
def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]: #k,也就是斜率一样，两条线平行啊，没交叉点啊!!!
        logger.debug('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0: # 都是平行于x轴，没交叉点啊!!!
        logger.debug('Cross point does not exist')
        return None
    if line1[1] == 0: #??? 中间的一位是0的是啥含义来着？忘了
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else: # 这个才是求那个点呢，这个是解一个2元1次方程组得到的，就是x1=x2,y1=y2，方程就是成了 y1=x1*k1+b1 ~ y1=x1*k2+b2，解出，x1,y1
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


# 读这个函数前，请默默自觉的在面前的纸上画一个平行四边形，从左上角开始标4个点：p0,p1,p2,p3，顺时针
# 我画了个，您参考可以：http://www.piginzoo.com/images/20190828/1566987583219.jpg
def rectangle_from_parallelogram(poly):
    '''
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    '''
    p0, p1, p2, p3 = poly

    #  np.dot(p1-p0, p3-p0)
    # -----------------------= cos(p03~p32的夹角)
    #   |p0-p1| * |p3-p0|
    # 这步是算出平行四边形左下角的夹角
    angle_p0 = np.arccos(np.dot(p1-p0, p3-p0)/(np.linalg.norm(p0-p1) * np.linalg.norm(p3-p0)))

    # 平行四边形左下角，小于90度
    if angle_p0 < 0.5 * np.pi:
        #横着的一个平行四边形
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0-p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)# <-----这个是核心，是过p0做了一个垂直线，参考我这张图：http://www.piginzoo.com/images/20190828/1566987583219.jpg
                                                   # 可能你担心好好一个平行四边形，你不是给切出了一块么，其实，没事，切不到原的文字框的，你看我这张图就能明白
            new_p3 = line_cross_point(p2p3, p2p3_verticle) # 好嘛~，终于得到我梦寐以求的矩形的左下角的点了，我梦寐以求的是这个矩形啊，new_p3只是副产品

            ############ 这里有大疑问 ??? 这个会影响回归的效果，也就是计算那个框的精确性 ###########
            ## p2，恩，接下来搞丫🏃p2，这块我理解不了，和我的图对比，我应该去搞P1啊，如果是他这样，会切掉一部分我的文本区域啊？？？！！！（大惑）
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)
            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        # 竖着的一个平行四边形
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    # 平行四边形左下角，大于90度
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

# 把矩形的4个点重新排序？！排个屁啊，前面不都已经靠min(x+y)算过谁是左上角了么？我就奇了个怪了???
#
def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1]) # y至最大的的那个点的index

    # "有2个点的y一样，都是和最大的y一样"，啥意思？，就是这条边和x轴平行
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1)) # 又是之前的伎俩，x+y最小的那个点的index[0~3]
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.#<----看，这个0就是我们要那个和x轴的夹角，这里当然是0啦
    # 不是平行x轴的
    else:
        # 找到最低点右边的点 - find the point that sits right to the lowest point
        # p_lowest是啥来着，是最靠下的那个点的坐标
        p_lowest_right = (p_lowest - 1) % 4 # -1，就是下标-1，就是顺时针小1的那个点
        p_lowest_left  = (p_lowest + 1) % 4 # +1，就是下标+1，就是顺时针大1的那个点
        #   -(poly[p_lowest][1] - poly[p_lowest_right][1])
        # --------------------------------------------------- => 就是矩形靠下边的那个线的斜率，求一下arctan，得到角度
        #    (poly[p_lowest][0] - poly[p_lowest_right][0])
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))

        # assert angle > 0
        if angle <= 0:
            logger.debug("角度<0: angle:%r,最低点：%r,最高点：%r",angle, poly[p_lowest], poly[p_lowest_right])

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

# 这个函数只有推断的时候用，训练不用
# 主要核心是调用lanms，来算出需要的框
# origin   [N,2]    放着前景的2维坐标，x从小到大
# geometry [h,w,5]  5张原图大小的"怪怪"图，你懂得
def restore_rectangle_rbox(origin, geometry):

    d = geometry[:, :4]    #geo_map:(h, w, 5)，[:4]是只切了前4个, d => [h*w,4]
    angle = geometry[:, 4] #geo_map:(h, w, 5)，[4]是第5个        angle=> [h*w,1]

    # for angle > 0
    # origin<===xy_text = np.argwhere(score_map > score_map_thresh)，而且是排过序的
    # 恩，结果是，那些是前景的点，并且按照置信度排过序，而且他们形成的矩形的角度大于零 （靠！条件真TM多）
    origin_0 = origin[angle >= 0] # 那些点
    d_0 = d[angle >= 0]           # 那些点的距离们
    angle_0 = angle[angle >= 0]   # 那些点的角度们

    # d: (6812, 4)
    # angle: (6812,)
    # origin_0: (3624, 3)
    # d_0: (3624, 4)
    # angle_0: (3624,)
    logger.debug("d:%r", d.shape)
    logger.debug("angle:%r", angle.shape)
    logger.debug("origin_0:%r",origin_0.shape)
    logger.debug("d_0:%r", d_0.shape)
    logger.debug("angle_0:%r", angle_0.shape)

    if origin_0.shape[0] > 0:
        # 什么鬼？这么复杂
        # 增加1个新维度[10,h,w]
        #p.shape=>(10, 3624)
        p = np.array([np.zeros(d_0.shape[0]),
                      -d_0[:, 0] - d_0[:, 2], # d维度是[h*w,4],d_0[:, 0]实际上是降维了[h*w]，或者说[h*w,1]，实际上得到是矩形的高
                       d_0[:, 1] + d_0[:, 3], # 矩形的长,维度是[h*w]
                      -d_0[:, 0] - d_0[:, 2], # 矩形的宽负数,维度是[h*w]
                       d_0[:, 1] + d_0[:, 3], # 矩形的长，维度是[h*w]
                      np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]),
                      d_0[:, 3],
                      -d_0[:, 2]])
        # 这里实在想不清楚了，需要调试一下
        logger.debug("======================================================")
        logger.debug("一共多少个前景点：d_0.shape[0]:%r",d_0.shape[0])
        logger.debug("每个点的-x-y:-d_0[:, 0] - d_0[:, 2].shape:%r", (- d_0[:, 0] - d_0[:, 2]).shape)
        logger.debug("得到的p的shape:%r",p.shape)
        # logger.debug("得到的p:%r", p)

        #[10,3624]=>[3642,5,2]
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2
        logger.debug("p的转置+reshape(-1,5,2):%r", p.shape)

        # 旋转矩阵：https://blog.csdn.net/csxiaoshui/article/details/65446125
        # [
        #     [x']    [cos,-sin,0]   [x]
        #     [y']  = [sin, cos,0] * [y]
        #     [1 ]    [0  ,    ,1]   [1]
        # ]
        # 先凑上述的矩阵，为何要重复5次？？？
        # np.repeat:对数组中的元素进行连续复制:[1,2,3]=>[1,1,1,2,2,2,3,3,3] : a.repeat(3)
        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0)) # N*5*2
        logger.debug("rotate_matrix_x:%r",rotate_matrix_x.shape)
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2
        logger.debug("rotate_matrix_x:%r",rotate_matrix_x.shape)
        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        logger.debug("rotate_matrix_y:%r", rotate_matrix_y.shape)
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))
        logger.debug("rotate_matrix_y:%r", rotate_matrix_y.shape)

        # 旋转,p是喜欢旋转矩阵，
        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1
        logger.debug("rotate_matrix_x * p:%r",(rotate_matrix_x * p).shape)
        logger.debug("np.sum(rotate_matrix_x * p, axis=2):%r",np.sum(rotate_matrix_x * p, axis=2).shape)
        logger.debug("np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]:%r",(np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]).shape)
        logger.debug("rotate_matrix_x * p:%rx%r", rotate_matrix_x.shape, p.shape)
        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)         # N*5*2
        logger.debug("p_rotate_x,p_rotate_y:%r,%r=>p_rotate,%r",p_rotate_x.shape, p_rotate_y.shape,p_rotate.shape)

        p3_in_origin = origin_0 - p_rotate[:, 4, :] # origin_0[N,2] - p_rotate[N,2]
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
        # 角度小于零，旋转矩阵不一样了
        p = np.array([-d_1[:, 1] - d_1[:, 3],
                      -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]),
                      -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]),
                      np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3],
                      np.zeros(d_1.shape[0]),
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

    # 返回的是4个点，[N,4]，矩形的4个点
    return np.concatenate([new_p_0, new_p_1])


def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)

# this is core method for samples making
# 这个应该是最最核心的样本数据准备的过程了,im_size现在是512x512
def generate_rbox(im_size, polys, tags):
    h, w = im_size

    # logger.debug("开始生成rbox数据：h:%d,w:%d",h,w)

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
            # 挨个算一下每个点，到别人最近的最小距离
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        # score map
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)
        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1) # ？？？

        # if the poly is too small, then ignore it during training，如果太小，就不参与训练了
        # 终于明白training_mask的妙用了，就是控制那些点不参与训练
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < FLAGS.min_text_size:
            cv2.fillPoly(training_mask,  poly.astype(np.int32)[np.newaxis, :, :],    0)

        # ???
        if tag:cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

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

            # 看这张图示：http://www.piginzoo.com/images/20190828/1566987583219.jpg
            # 求拟合曲线的k和b，返回的是 ax+by+c=0的系数表达
            edge          = fit_line([p0[0], p1[0]], [p0[1], p1[1]]) #左上，右上 0,1
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]]) #左上，左下 0,3
            forward_edge  = fit_line([p1[0], p2[0]], [p1[1], p2[1]]) #右上，右下 1,2

            # 看p2到p0p1的距离 > p3到p0p1的距离
            # 就是看p2,p3谁离直线p0p1远，就选谁画一条平行于p0p1的先作为新矩形的边
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                # 平行线经过p2 - parallel lines through p2，对，就是这个意思
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    # edge[0] = k,
                    # p2[1] - edge[0] * p2[0] = y - k*x = b
                    # edge_opposite实际上就是[k,-1,b],就是那条平行线的k、b
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # 经过p3 - after p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]

            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p2 = line_cross_point(forward_edge, edge_opposite) # 不对啊，edge_opposite是那条平行线，但是forward_edge和forward_edge不一定垂直啊？？？

            # 再求p0,p3平行于p1p2的距离谁远，然后做平行线，然后
            # 求这条平行线forward_opposite和edge_opposite的交点=> new p3，以及
            # 求这条平行线forward_opposite和edge的交点         => new p0
            # 我勒个去，我怎么觉得我圈出来一个平行四边形，而不是一个矩形啊，颠覆了我的假设认知了
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
            new_p0 = line_cross_point(forward_opposite, edge)#  求这条平行线forward_opposite和edge的交点         => new p0
            new_p3 = line_cross_point(forward_opposite, edge_opposite)# 求这条平行线forward_opposite和edge_opposite的交点=> new p3

            # 果然是平行四边形啊，作者起了这个名字"parallelograms"，爱死你了 (^_^)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])


            # 上面不是画了了一个平行四边形了么？可是，用另外用一个边，也可以画出一个平行四边形啊
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


            # 然后，我得到了2个平行四边形，我勒个去，我猜到了开头（以为要通过不规则四边形找一个规律的四边形），
            # 但是我没猜到结尾（我以为是画个矩形，却尼玛画出平行四边形，还是两个）


        # 找那个最大的平行四边形，恩，可以理解
        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
        # sort thie polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1) #axis=1，什么鬼？是把x、y加到了一起,[[1,1],[2,2]]=>[2,4]
        min_coord_idx = np.argmin(parallelogram_coord_sum) # 实际上是找左上角，一般来讲是x+y最小的是左上角，你别跟我扯极端情况，
                                                           # 我自己画了一下，这事不是那么绝对，但是大部分是别太变态的情况，是这样的
        # 按照那个点当做p0，剩下的点依次编号，重新调整0-3的标号
        parallelogram = parallelogram[
            [min_coord_idx,
             (min_coord_idx + 1) % 4,
             (min_coord_idx + 2) % 4,
             (min_coord_idx + 3) % 4]]

        # 算出套在平行四边形外面的框，我觉得里面的算法有问题，等XDJM们帮着我解惑？？？
        rectange = rectangle_from_parallelogram(parallelogram)

        # 调整一下p0~p3的顺序，并且算出对应的夹角，恩，是的，夹角是在这里算出来的
        rectange, rotate_angle = sort_rectangle(rectange)

        p0_rect, p1_rect, p2_rect, p3_rect = rectange

        # xy_in_poly就是框里面的那些点的x,y坐标，应该很多，挨个算每个点到这个矩形的距离
        # point_dist_to_line，这个函数之前用过，不多说了，最后一个参数是点，前两个参数，线上的2个点
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
def generator(input_size=512,
              batch_size=32,
              type=None,
              background_ratio=3./8,
              random_scale=np.array([0.5, 1, 2.0, 3.0]),
              vis=False):

    # 训练数据，返回那一坨东西，score map，geo_map,....
    if type=="train":
        data_dir=FLAGS.training_data_path
        name="训练"
    # 测试数据，就一个图和labels就成了
    else:
        data_dir=FLAGS.validate_data_path
        name="验证"

    logger.debug("启动[%s]数据集加载器",name)
    # 获得训练集路径下所有图片名字
    image_list = np.array(get_images(data_dir))
    # index：总样本数
    index = np.arange(0, image_list.shape[0])
    # pdb.set_trace(x`)
    while True:
        np.random.shuffle(index)
        images = []
        labels = []
        image_names = []
        score_maps = []
        geo_maps = []
        training_masks = []
        count=0
        for i in index:
            try:
                # 读取图片
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                # logger.debug ("[%s]成功加载图片文件[%s]：",name,im_fn)
                h, w, _ = im.shape

                # 读取标签txt
                label_name = os.path.splitext(os.path.basename(im_fn))[0]
                label_dir = os.path.join(data_dir, "labels")
                txt_fn = os.path.join(label_dir,label_name+".txt")


                if not os.path.exists(txt_fn):
                    logger.debug('标签文件不存在啊：%s',txt_fn)
                    continue

                # logger.debug("[%s]成功加载标签文件：%s",name,txt_fn)

                # 读出对应label文档中的内容
                # text_polys：样本中文字坐标:[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]，text_polys shape:[N,4,2]，4是4个点，2是x和y
                # text_tags：文字框内容是否可辨识
                # 例子：
                #   377, 117, 463, 117, 465, 130, 378, 130, GenaxisTheatre
                #   493, 115, 519, 115, 519, 131, 493, 131, [06]
                text_polys, text_tags = load_annoataion(txt_fn) #4点标注，是不规则四边形，而不是一个旋转的矩形

                # 保存其中的有效标签框，并修正文本框坐标溢出边界现象，多边形面积>1
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

                # 如果是训练数据，就弄出来这2数据就够
                if type=="validate":
                    images.append(im)
                    labels.append(text_polys)
                # 这个是train，训练数据，那就多了，你知道的。。。。
                else:

                    # !!!我注释掉了，哥不缺样本，没变要做增强，就用原图
                    # 这段是在resize图像，不知道为何这样做，是为了做样本增强么？resize4个选项，相当于数据多了4倍。不过，我感觉没必要
                    # if text_polys.shape[0] == 0:
                    #     continue
                    # random scale this image，为何要随机resize一下，做样本增强么？，random_scale=[0.5, 1, 2.0, 3.0]
                    #rd_scale = np.random.choice(random_scale)
                    #im = cv2.resize(_image, dsize=None, fx=rd_scale, fy=rd_scale)
                    #text_polys *= rd_scale

                    # 思考？
                    # 为何这要做切一部分，然后resize成512x512呢，难道是，EAST对太长的图像识别能力差，特别是两边的点
                    # 不对！我记得说的是一个框的两边的点，预测可能会不准。。。。所以才有个EAST增强。
                    # 不知道了......????

                    # 关于background_ratio：https://github.com/argman/EAST/issues/133
                    #
                    # "its kind of data augmentation, random scale is used to scale the image ,
                    # and then a patch is randomly cropped from the image, back_ground ratio is used to
                    # crop some background training data that does not contain text. Anyway ,
                    # maybe there is better strategy for augmentation" -- argman
                    # 我理解这个目的就是为了测试一些纯负样本，不过我给人觉得没必要啊，干嘛非要虐待自己，专搞一些背景出来虐自己呢？有啥好处呢？
                    # 这玩意预测出来，肯定score map都很低啊，而且，geo_map都为0，就是到各个框的上下左右都是0，何苦呢？不理解！！！？？？
                    start = time.time()
                    if np.random.rand() < background_ratio: #background_ratio=3/8，background_ratio是个啥东东？

                        # crop background，crop_background=True这个标志就是说，我要的是背景，不能给我包含任何框
                        im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                        if text_polys.shape[0] > 0: #看！即使你切出来一块有框的图，我也不要，对！我！不！要！
                            #logger.debug("没法搞到一块不包含文本框的纯背景啊:(%s",im_fn)
                            continue

                        # pad and resize image,最终图片变成512x512，图像不变形，padding补足
                        # 注意！这个图像处理是这样的，例如：
                        # 如果原图是300x400，那么变成了512x512，右面和下面都有黑色padding
                        # 如果原图是600x800，那么先是800x800的黑色背景，然后把这张图帖进去，然后下面空出200高的黑色padding，然后这图resize成512x512
                        new_h, new_w, _ = im.shape
                        max_h_w_i = np.max([new_h, new_w, input_size]) # input_size是命令行参数，默认就是512，也没人改
                        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                        im_padded[:new_h, :new_w, :] = im.copy()
                        im = cv2.resize(im_padded, dsize=(input_size, input_size)) # input_size = 512，看，这里强制给resize给

                        # 如果和下面的else对比一下，就能知道，他并没有产生rbox的数据（即调用generate_rbox）
                        score_map        = np.zeros((input_size, input_size), dtype=np.uint8)
                        geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
                        geo_map          = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                        training_mask    = np.ones((input_size, input_size), dtype=np.uint8)
                        # logger.debug("进程[%d],生成了一个不包含文本框的背景%s数据：score:%r,geo:%r,mask:%r,耗时:%f",
                        #              os.getpid(),score_map.shape,geo_map.shape,training_mask.shape,name,
                        #              (time.time()-start))

                    else: # > 3/8

                        # 这个是切出一个子图来，就用这个子图做训练了，我理解，还是跟数据增强差不多，可以大幅的提高图像的利用率啊
                        im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                        if text_polys.shape[0] == 0:
                            logger.debug("文本框数量为0，image:%r,文本框：%r", im_fn.shape,text_polys.shape)
                            continue

                        h, w, _ = im.shape

                        # 这步操作就是最终在不变形的情况下，把子图resize成512x512，空白处padding 0
                        # pad the image to the training input size or the longer side of image
                        # 注意！这个图像处理是这样的，例如：
                        # 如果原图是300x400，那么变成了512x512，右面和下面都有黑色padding
                        # 如果原图是600x800，那么先是800x800的黑色背景，然后把这张图帖进去，然后下面空出200高的黑色padding，然后这图resize成512x512
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


                        # 把标注坐标缩放
                        resize_ratio_3_x = resize_w/float(new_w)
                        resize_ratio_3_y = resize_h/float(new_h)
                        text_polys[:, :, 0] *= resize_ratio_3_x
                        text_polys[:, :, 1] *= resize_ratio_3_y
                        new_h, new_w, _ = im.shape

                        score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)
                        # logger.debug("进程[%d],生成%s数据(score,geo,mask)：%r,%r,%r，耗时:%f",
                        #              os.getpid(),name,
                        #              score_map.shape,
                        #              geo_map.shape,
                        #              training_mask.shape,
                        #              (time.time()-start))

                    images.append(im[:, :, ::-1].astype(np.float32))
                    image_names.append(im_fn)

                    #!!! 这里藏着一个惊天大秘密：( >﹏<。)
                    # geo_map[::4, ::4, :] => 表示隔4个采样一个点，把图从512x512=>128x128
                    # 你知道他为何这么做么？（坏笑）是因为他要和resnet对上，resnet取的是conv2,conv3,conv4,conv5，而conv2其实不是原图，而是原图的1/4，
                    # 所以，我们最终uppooling后，最后一个合并的是conv2，得到就是原图(512)的1/4(128)，这样，你的样本也要变成原有的1/4，
                    # Resnet50参考此图：https://www.cnblogs.com/ymjyqsx/p/7587033.html
                    # 恩，通过这个采样，不就达到目的了么？
                    # 这个细节，细思极恐，一不小心就被作者忽悠了
                    # 所以，这里要记着，EAST预测出来的东西，坐标都是原图1/4的，我回头去看看推断的时候，怎么处理这个问题？
                    score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                    geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                    training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                count+=1

                # 凑过了batch_size，就被这批数据yield出去
                # 你要理解哈，最终的是啥，是32个(32假设是批次），32个images,score_maps,geo_maps.....
                if len(images) == batch_size:
                    # logger.debug("[%s]返回一个批次数据：%d张", name, batch_size)
                    if type == "validate":
                        yield images, labels
                    else:
                        yield images, image_names, score_maps, geo_maps, training_masks

                    images = []
                    labels = []
                    image_names = []
                    score_maps = []
                    geo_maps = []
                    training_masks = []
            except BaseException as e:
                logger.debug("Error happened:%s",str(e))
                import traceback
                traceback.logger.debug_exc()
                continue


def get_batch(num_workers,**kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=100, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    #logger.debug("queue is empty, which cause we are wating....")
                    time.sleep(1)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()



if __name__ == '__main__':
    pass
