# import os
# root_dir = "/Users/yanmeima/Desktop/east_data/ch4_test_images"
# for file_name in os.listdir(root_dir):
#     print("=============>", file_name)
#     newname = root_dir + '/' + file_name[4:]
#     file_name = root_dir + '/' + file_name
#     os.rename(file_name, newname)

import os
import cv2
# root_dir = "/Users/yanmeima/Desktop/发票照片new"
# #i = 0
# for file_name in os.listdir(root_dir):
#     print("=============>", file_name)
#     print(type(file_name))

#    i +=1
#     image = cv2.imread(os.path.join(root_dir,file_name))
#     cv2.imwrite(os.path.join('/Users/yanmeima/Desktop/bill/' +str(i)+'.jpg'), image)

import numpy as np
p0 = [2.33,3.44]
p1 = [4.32,5.23]
p2 = [3.04,4.22]
p3 = [4.86,8.11]
print(type(p0))
print(p0)
print(p0[0])
print(p0[1])

l = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
print(l)