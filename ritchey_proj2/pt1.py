"""
 Title: Project 2, Part 1: Stereo Matching
 Author: Dominic Ritchey, dmr5bq@virginia.edu
 Due: 16 March 2017
 File: pt1.py
"""

# imports
import numpy, scipy
import skimage, skimage.filters, skimage.io
import scipy.ndimage.filters
from time import time
from math import floor, sqrt
import cv2


# constants and parameters

IMG_SIZE_ERR = Exception("The images you provided do not match and therefore cannot be compared")

IN_PATH = 'assets/'
OUT_PATH = 'results/pt1/'

POS_INF = 0x7fffffffffffffff

MATCH_THRES = 35

RED = 0
GREEN = 1
BLUE = 2

img_num = '1'

filename_l = 'img' + img_num + '_l.png'
filename_r = 'img' + img_num + '_r.png'
filename_gt = 'img' + img_num + '_gt.npy'
filename_data = 'data_img' + img_num + ".txt"

# start timing
start_time = time()

# load integer images
img_l = skimage.io.imread(IN_PATH + filename_l)
img_r = skimage.io.imread(IN_PATH + filename_r)
img_gt = numpy.load(IN_PATH + filename_gt)

# convert images to float images
img_l = skimage.img_as_float(img_l)
img_r = skimage.img_as_float(img_r)

# get img dimensions and set maximum disparities
width = img_l.shape[1]
height = img_l.shape[0]

# calculate max_disparity
max_disparity = int(floor(height / 3))

# generate numpy arrays to store the DSI and depth_map
DSI = numpy.ones((height, width, max_disparity), dtype=numpy.float32)
output_bilateral = numpy.zeros((height, width))
output_gaussian = numpy.zeros((height, width))
output_joint_bilateral = numpy.zeros((height, width))

# build the disparity space image (DSI)
for y in range(height):
    for x in range(width):
        for d in range(max_disparity):

            # boundary case detection
            if x - d > 0:

                # calculate the euclidean distance for each color channel
                d_r = abs(img_l[y, x, RED] - img_r[y, x - d, RED])
                d_g = abs(img_l[y, x, GREEN] - img_r[y, x - d, GREEN])
                d_b = abs(img_l[y, x, BLUE] - img_r[y, x - d, BLUE])

                # save the summed distance into the DSI
                DSI[y, x, d] = (d_r + d_g + d_b)
            else:

                # for boundary case -> assign max_value
                DSI[y, x, d] = 1.0

DSI_max = numpy.amax(DSI)
DSI_gaussian = DSI.copy()
DSI_bilateral = DSI.copy()
DSI_joint_bilateral = DSI.copy()

for d in range(max_disparity):

    DSI_gaussian[:, :, d] = numpy.multiply(DSI_gaussian[:, :, d], 1 / DSI_max)
    DSI_gaussian[:, :, d] = skimage.filters.gaussian(DSI_gaussian[:, :, d], 1.5)
    DSI_gaussian[:, :, d] = numpy.multiply(DSI_gaussian[:, :, d], DSI_max)

    DSI_bilateral[:, :, d] = cv2.bilateralFilter(DSI_bilateral[:, :, d], 5, 100, 100)

    # DSI_joint_bilateral[:, :, d] = cv2.ximgproc.jointBilateralFilter(img_l.astype(numpy.float32),
     #                                                                DSI_joint_bilateral[:, :, d].astype(numpy.float32),
      #                                                               DSI_joint_bilateral[:, :, d].astype(numpy.float32),
       #                                                              5, 75, 75)

"""
    For each pixel, find the disparity d for which the DSI value
    is the lowest and save it to the output array
"""

for y in range(height):
    for x in range(width):
        # find the depth for which DSI value is minimal
        min_val_g = POS_INF
        min_val_b = POS_INF
        min_val_j = POS_INF
        min_d_g = max_disparity
        min_d_b = max_disparity
        min_d_j = max_disparity

        for d in range(max_disparity):
            if DSI_gaussian[y, x, d] < min_val_g:
                min_val_g = DSI_gaussian[y, x, d]
                min_d_g = d
        for d in range(max_disparity):
            if DSI_bilateral[y, x, d] < min_val_b:
                min_val_b = DSI_bilateral[y, x, d]
                min_d_b = d
        for d in range(max_disparity):
            if DSI_joint_bilateral[y, x, d] < min_val_j:
                min_val_j = DSI_joint_bilateral[y, x, d]
                min_d_j = d

        output_gaussian[y, x] = min_d_g
        output_bilateral[y, x] = min_d_b
        output_joint_bilateral[y, x] = min_d_j

"""
    Calculate the RMS error from the ground truth for the image set
"""

sum_of_sq_diff_g = 0
sum_of_sq_diff_b = 0
sum_of_sq_diff_j = 0

for y in range(height):
    for x in range(width):

        d_true = img_gt[y, x]

        d_calc_g = output_gaussian[y, x]
        d_calc_b = output_bilateral[y, x]
        d_calc_j = output_joint_bilateral[y, x]

        sum_of_sq_diff_g += (d_true - d_calc_g) ** 2
        sum_of_sq_diff_b += (d_true - d_calc_b) ** 2
        sum_of_sq_diff_j += (d_true - d_calc_j) ** 2

total_px = height * width

err_rms_g = sqrt(sum_of_sq_diff_g / total_px)
err_rms_b = sqrt(sum_of_sq_diff_b / total_px)
err_rms_j = sqrt(sum_of_sq_diff_j / total_px)

data_str = ''

data_str += 'Image {} is {}px x {}px in size\n\n'.format(img_num, width, height)

data_str += "The RMS error for {} is {}\n".format('bilateral filtering', err_rms_b)
data_str += "The RMS error for {} is {}\n".format('gaussian filtering', err_rms_g)
data_str += "The RMS error for {} is {}\n".format('joint bilateral filtering', err_rms_j)

"""
    Left-right consistency check using bilateral filtering
"""

d_map = numpy.zeros((height, width))
d_map_l = output_bilateral.copy()
d_map_r = numpy.zeros((height, width))

""" calculate the right-to-left depth map """

DSI_r = numpy.ones((height, width, max_disparity), dtype=numpy.float32)

# build the disparity space image (DSI)
for y in range(height):
    for x in range(width):
        for d in range(max_disparity):

            # boundary case detection
            if x + d < width:

                # calculate the euclidean distance for each color channel
                d_r = abs(img_l[y, x + d, RED] - img_r[y, x, RED])
                d_g = abs(img_l[y, x + d, GREEN] - img_r[y, x, GREEN])
                d_b = abs(img_l[y, x + d, BLUE] - img_r[y, x, BLUE])

                # save the summed distance into the DSI
                DSI_r[y, x, d] = (d_r + d_g + d_b)
            else:

                # for boundary case -> assign max_value
                DSI_r[y, x, d] = 1.0

for d in range(max_disparity):
    DSI_r[:, :, d] = cv2.bilateralFilter(DSI_r[:, :, d], 5, 100, 100)

for y in range(height):
    for x in range(width):
        # find the depth for which DSI value is minimal
        min_val = POS_INF
        min_d = max_disparity

        for d in range(max_disparity):
            if DSI_r[y, x, d] < min_val:
                min_val = DSI_r[y, x, d]
                min_d = d

        d_map_r[y, x] = min_d

for y in range(height):
    for x in range(width):

        for h in range(MATCH_THRES):
            d_l = d_map_l[y, x]

            if x + h < width:
                d_r_p = d_map_r[y, x + h]
            else:
                d_r_p = d_map_r[y, width - 1]

            if x - h >= 0:
                d_r_n = d_map_r[y, x - h]
            else:
                d_r_n = d_map_r[y, 0]

            if d_l == d_r_p or d_l == d_r_n:
                d_map[y, x] = d_map_l[y, x]
                break

data_str += "\nTime to complete: {}".format(time() - start_time)

with open(OUT_PATH + filename_data, 'w') as f:
    f.write(data_str)

scipy.misc.imsave(OUT_PATH + "img" + img_num + "_depth_gaussian_left_to_right.png", output_gaussian)
scipy.misc.imsave(OUT_PATH + "img" + img_num + "_depth_consistency_check.png", d_map)
scipy.misc.imsave(OUT_PATH + "img" + img_num + "_depth_bilateral_left_to_right.png", output_bilateral)
scipy.misc.imsave(OUT_PATH + "img" + img_num + "_depth_bilateral_right_to_left.png", d_map_r)
scipy.misc.imsave(OUT_PATH + "img" + img_num + "_depth_joint_bilateral_left_to_right.png", output_bilateral)
scipy.misc.imsave(OUT_PATH + "DSI-slices/" + "img" + img_num + "_DSI_1.png", DSI[:, :, 1])
scipy.misc.imsave(OUT_PATH + "DSI-slices/" + "img" + img_num + "_DSI_5.png", DSI[:, :, 5])
scipy.misc.imsave(OUT_PATH + "DSI-slices/" + "img" + img_num + "_DSI_10.png", DSI[:, :, 10])

print ("Time to complete: {}".format(time() - start_time))
