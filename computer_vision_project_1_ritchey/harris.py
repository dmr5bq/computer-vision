#!/usr/bin/python
# coding: utf-8

import numpy, pylab, scipy.ndimage.filters
import skimage, skimage.io, skimage.filters
from math import sqrt, atan, pi
import time
import gc

"""
    Constants and parameters:
"""
THRESHOLD = 0.1

RED = 0
GREEN = 1
BLUE = 2

P_RED = 0.30
P_GREEN = 0.59
P_BLUE = 0.11

OUTPATH = 'results/'
PATH = 'assets/'
FILENAME = 'flower.jpg'
OUTFILENAME = ('harris_result?threshold=%s_' + FILENAME) % THRESHOLD

gc.enable()

start = time.time()

# PART 1: computing smoothed gradients
# - creating G[x,y], edge strength array

#   load image
img = skimage.io.imread(PATH + FILENAME)

#   convert image to float format
img = skimage.img_as_float(img)

# get image width and height
h = img.shape[0]
k = img.shape[1]

m = 4

# build arrays for data storage
L = numpy.zeros((h, k))
G = numpy.zeros((h, k))

# build the gaussian kernel for smoothing
H = numpy.multiply(numpy.array([
    [2, 4, 5, 4, 2],
    [4, 9, 12, 9, 4],
    [5, 12, 15, 12, 5],
    [4, 9, 12, 9, 4],
    [2, 4, 5, 4, 2]
]), 1 / 159)


# build the derivative arrays
D_x = numpy.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
D_y = numpy.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

corners = []

#   extract image luminance as a 2D float array
for x in range(h):
    for y in range(k):
        r = img[x, y, RED]
        g = img[x, y, GREEN]
        b = img[x, y, BLUE]

        lum = P_RED * r + P_GREEN * g + P_BLUE * b

        L[x, y] = min(lum, 1)

img = None
gc.collect()

print ("\n -- DONE constructing L")
print ("time elapsed: %s" % (time.time() - start))

#   find the x- and y-component of the gradient

# smooth with a Gaussian kernel using a convolution
# use scipy.signal.convolve2d() or scipy.ndimage.filters.convolve
L = scipy.ndimage.filters.convolve(L, H, mode='reflect', cval=1.0)

# compute the gradients
G_x = scipy.ndimage.filters.convolve(L, D_x, mode='reflect', cval=1.0)
G_y = scipy.ndimage.filters.convolve(L, D_x, mode='reflect', cval=1.0)
max = 0
# for each pixel
for x in range(h):
    for y in range(k):
        C = numpy.zeros((2, 2))
        if m - 1 < x < h - m - 1 and m - 1 < y < k - m - 1:

            for u in range(- m - 1, m):
                for v in range(- m - 1, m):
                    f_x = G_x[x + u, y + v]
                    f_y = G_y[x + u, y + v]

                    C[0, 0] += f_x ** 2
                    C[0, 1] += f_x * f_y
                    C[1, 0] += f_x * f_y
                    C[1, 1] += f_y ** 2

            C = numpy.multiply(C, (2 * m + 1) ** -1)

            e = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0] - 0.04 * ((C[0, 0] + C[1, 1]) ** 2)
            if abs(e) > max:
                max = abs(e)
            if abs(e) > THRESHOLD and m < x < h - m and m < y < k - m:
                corners.append(((x, y), abs(e)))

    print("\t\t%s / %s pixels processed" % (x * k, h * k))

print ("\n -- DONE constructing corner candidates")
print ("time elapsed: %s" % (time.time() - start))
print (max)

corners = sorted(corners, key=lambda eig: eig[1], reverse=True)
cor_copy = corners[:]

print ("\n -- DONE constructing corner candidates")
print ("time elapsed: %s" % (time.time() - start))

#
print ("---\n")
print(corners[1:50])
print(len(corners))

print ("---\n")

dist = 1
for i in range(len(cor_copy)):

    corner = cor_copy[i]
    px = corner[0][0]
    py = corner[0][1]

    for j in range(i, len(cor_copy)):

        other_corner = cor_copy[j]

        qx = other_corner[0][0]
        qy = other_corner[0][1]

        if px - dist <= qx <= px + dist and px != qx:
            if py - dist <= qy <= py + dist and px != qx:
                    if other_corner in corners:
                        corners.remove(other_corner)

#
print(corners[1:50])
print(len(corners))

print ("\n -- DONE thinning corner candidates")
print ("time elapsed: %s" % (time.time() - start))

L = numpy.multiply(L, 0.6)

for corner in corners:
    point = corner[0]

    for i in range(-10,10):
        if 1 < point[0] + i < h - 1:
            L[point[0] + i, point[1]] = 1.0

        if 1 < point[1] + i < k - 1:
            L[point[0], point[1] + i] = 1.0

print("\n\n----\nExecuted in %s seconds" % (time.time() - start) )

scipy.misc.imsave(OUTPATH + OUTFILENAME, L)







