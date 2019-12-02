import numpy as np
def point_dist_to_line(p1, p2,      p3):
    # compute the distance from p3 to p1-p2
    # cross:向量积，数学中又称外积、叉积,运算结果是一个向量而不是一个标量。并且两个向量的叉积与这两个向量和垂直。模长是|a|*|b|*sin夹角，方向上右手法则
    # 叉乘的二维的一个含义是，"在二维中，两个向量的向量积的模的绝对值等于由这两天向量组成的平行四边形的面积"
    # np.linalg.norm(np.cross(p2 - p1, p1 - p3)) 就是p1p3,p1p2夹成的平行四边形的面积
    # 除以
    # np.linalg.norm(p2 - p1)，是p1p2的长度，
    # 得到的，就是P3到p1,p2组成的的距离，
    # 你可以自己画一个平行四边形，面积是 底x高，现在面积已知，底就是p1p2，那高，就是p3到p1p2的距离
    pp = []
    pp.append(p1)
    pp.append(p2)
    pp.append(p3)
    if len(pp) == 3:
        return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

if __name__ == '__main__':
    p1 = [0,1]
    p2= [1,0]
    p3 =None
    print(point_dist_to_line(p1, p2,      p3))
