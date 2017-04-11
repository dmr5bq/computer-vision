"""
 Title: Project 2, Part 2: Panorama Stitching
 Author: Dominic Ritchey, dmr5bq@virginia.edu
 Due: 16 March 2017
 File: pt2.py
"""

import cv2
import numpy
from random import sample
from math import sqrt, ceil
import scipy, skimage, numpy.linalg, skimage.transform
from time import time

IN_PATH = 'assets/'
OUT_PATH = 'results/pt2/'

PIXEL_THRES = 5

filename_a = 'img_a.jpg'
filename_b = 'img_b.jpg'

A = cv2.imread(IN_PATH + filename_a, 1)
B = cv2.imread(IN_PATH + filename_b, 1)

start_time = time()


def get_keypoint_matches(img_a, img_b):

    sift = cv2.xfeatures2d.SIFT_create()

    keypoint_a, des_a = sift.detectAndCompute(img_a, None)
    keypoint_b, des_b = sift.detectAndCompute(img_b, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, des_b, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return good, keypoint_a, keypoint_b


def apply_homography(x, y, H):
    """Return a set of points (u,v) that are the result of warping (x, y) with H"""
    a = H[0, 0]
    b = H[0, 1]
    c = H[0, 2]
    d = H[1, 0]
    e = H[1, 1]
    f = H[1, 2]
    g = H[2, 0]
    h = H[2, 1]

    k = g * x + h * y + 1
    if k == 0:
        k = 0.01

    u = (a * x + b * y + c) / k
    v = (d * x + e * y + f) / k

    if round(u) == -0:
        u = abs(u)

    if round(v) == -0:
        v = abs(v)

    return u, v


def get_homography_mapping(dst_points, src_points):

    if len(dst_points) != 4 or len(src_points) != 4:
        raise Exception("get_homography_mapping requires two lists of four points from images A and B")

    H = numpy.ones((3, 3))

    A = numpy.zeros((8, 8))

    b = numpy.zeros(8)

    mat_ind = 0
    for i in range(len(dst_points)):

        x = src_points[i][0]
        y = src_points[i][1]

        u = dst_points[i][0]
        v = dst_points[i][1]

        A[mat_ind] = numpy.array([x, y, 1, 0, 0, 0, - u * x, - u * y])
        A[mat_ind + 1] = numpy.array([0, 0, 0, x, y, 1, - v * x, - v * y])

        b[mat_ind] = u
        b[mat_ind + 1] = v

        mat_ind += 2

    params = numpy.linalg.lstsq(A, b)[0]

    H[0, 0] = params[0]
    H[0, 1] = params[1]
    H[0, 2] = params[2]
    H[1, 0] = params[3]
    H[1, 1] = params[4]
    H[1, 2] = params[5]
    H[2, 0] = params[6]
    H[2, 1] = params[7]
    H[2, 2] = 1

    return H


def find_best_homography_from_matches(kp_a, kp_b, matches):

    trials = 100000 # int(5 * ceil((len(matches)) ** 2))

    homographies = []

    for i in range(trials):

        if i % 1000 == 0:
            print ("{} / {} homographies computed, {} seconds elapsed\n\n---\n".format(i, trials, round(time() - start_time)))

        result = find_homography_from_matches(kp_a, kp_b, matches)
        homographies.append(result)

    a = sorted(homographies, key=lambda res: res['count_inliers'], reverse=True)

    return a[0]['homography'], a[1]['homography'], a[2]['homography'], a[3]['homography'], a[4]['homography']


def find_homography_from_matches(kp_a, kp_b, matches):

    indices = sample(range(len(matches)), 4)

    list_b = []
    list_a = []

    for i in range(4):

        kb = kp_b[indices[i]]
        x = kb.pt[0]
        y = kb.pt[1]

        ka = kp_a[indices[i]]
        u = ka.pt[0]
        v = ka.pt[1]

        list_b.append((x, y))
        list_a.append((u, v))

    H = get_homography_mapping(list_a, list_b)

    count_inliers = 0

    for i in range(len(matches)):
        # ||Hb - a|| < PIXEL_THRES

        kb = kp_b[i]
        x = kb.pt[0]
        y = kb.pt[1]

        ka = kp_a[i]
        u = ka.pt[0]
        v = ka.pt[1]

        x_calc, y_calc = apply_homography(x, y, H)
        x_true, y_true = u, v

        error = sqrt((x_calc - x_true) ** 2 + (y_calc - y_true) ** 2)

        if error < PIXEL_THRES:
            count_inliers += 1

    return {'count_inliers': count_inliers, 'homography': H}


def composite_warped(a, b, H):
    "Warp images a and b to a's coordinate system using the homography H which maps b coordinates to a coordinates."
    out_shape = (a.shape[0], 2 * a.shape[1])                               # Output image (height, width)
    p = skimage.transform.ProjectiveTransform(numpy.linalg.inv(H))       # Inverse of homography (used for inverse warping)
    bwarp = skimage.transform.warp(b, p, output_shape=out_shape)         # Inverse warp b to a coords
    bvalid = numpy.zeros(b.shape, 'uint8')                               # Establish a region of interior pixels in b
    bvalid[1:-1,1:-1,:] = 255
    bmask = skimage.transform.warp(bvalid, p, output_shape=out_shape)    # Inverse warp interior pixel region to a coords
    apad = numpy.hstack((skimage.img_as_float(a), numpy.zeros(a.shape))) # Pad a with black pixels on the right
    return skimage.img_as_ubyte(numpy.where(bmask==1.0, bwarp, apad))    # Select either bwarp or apad based on mask

good, keypoint_a, keypoint_b = get_keypoint_matches(A, B)

H_list = find_best_homography_from_matches(keypoint_a, keypoint_b, good)

C = cv2.drawMatchesKnn(A,keypoint_a,B,keypoint_b,good,None,flags=2)

cv2.imwrite(OUT_PATH + "keypoint_matches_sift.png", C)
for i in range(len(H_list)):
    cv2.imwrite(OUT_PATH + "composite{}_{}.png".format(PIXEL_THRES,i), composite_warped(A, B, H_list[i]))

