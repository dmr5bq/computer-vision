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
THRESHOLD_LOW = 0.01
THRESHOLD_HIGH = 0.1

NO_EDGE = 0.0
WEAK_EDGE = 0.2
STRONG_EDGE = 1.0

RED = 0
GREEN = 1
BLUE = 2

P_RED = 0.30
P_GREEN = 0.59
P_BLUE = 0.11

OUTPATH = 'results/'
PATH = 'assets/'
FILENAME = 'flower.jpg'
OUTFILENAME = ('canny_result?hthreshold=%s&lthreshold=%s_' + FILENAME) % (THRESHOLD_HIGH, THRESHOLD_LOW)


start_time = time.time()

gc.enable()

# PART 1: computing smoothed gradients
# - creating G[x,y], edge strength array

#   load image
img = skimage.io.imread(PATH + FILENAME)

#   convert image to float format
img = skimage.img_as_float(img)

# get image h and k
h = img.shape[0]
k = img.shape[1]

# build arrays for data storage
L = numpy.zeros((h, k))
G = numpy.zeros((h, k))
I = numpy.zeros((h, k))
O = numpy.zeros((h, k))

# build the gaussian kernel for smoothing
H = numpy.multiply(numpy.array([
                        [2, 4, 5, 4, 2],
                        [4, 9, 12, 9, 4],
                        [5, 12, 15, 12, 5],
                        [4, 9, 12, 9, 4],
                        [2, 4, 5, 4, 2]
                    ]), 1/159)

# build the derivative arrays
D_x = numpy.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
D_y = numpy.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

step_0 = time.time() - start_time
print("step 0: %s" % step_0)

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

#   find the x- and y-component of the gradient

# smooth with a Gaussian kernel using a convolution
# use scipy.signal.convolve2d() or scipy.ndimage.filters.convolve
L = scipy.ndimage.filters.convolve(L, H, mode='reflect', cval=1.0)

# compute the gradients
G_x = scipy.ndimage.filters.convolve(L, D_x, mode='reflect', cval=1.0)
G_y = scipy.ndimage.filters.convolve(L, D_x, mode='reflect', cval=1.0)
D = scipy.arctan2(G_y, G_x)

L = None
gc.collect()

step_1 = time.time() - start_time - step_0
print("step 1: %s" % step_1)

#   at each pixel:
#     compute the edge-strength G

max = 0
for x in range(h):
    for y in range(k):

        dx = G_x[x, y]
        dy = G_y[x, y]

        G[x, y] = sqrt(dx ** 2 + dy ** 2)

        if G[x, y] > max:
            max = G[x, y]

G = scipy.hypot(G_y, G_x)

# PART 2: non-maximal suppression
# - creating I[x,y], the thinned-edge image

#   at each pixel:
#       find the direction ({0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4}) \
#       that is closest to the orientation of the edge found above

for x in range(h):
    for y in range(k):
        d = D[x, y]

        if (0 <= d < 22.5 and d >= 0) or \
           (d >= 157.5 and d < 202.5) or \
           (d >= 337.5 and d <= 360):
                D[x, y] = 0

        elif (d >= 22.5 and d < 67.5) or \
             (d >= 202.5 and d < 247.5):
                D[x, y] = 45

        elif (d >= 67.5 and d < 112.5)or \
             (d >= 247.5 and d < 292.5):
                D[x, y] = 90

        else:
            D[x, y] = 135

I = G.copy()

for x in range(1, h - 1):
    for y in range(1, k - 1):

        g = G[x, y]

        if D[x, y] == 0:

            if (g <= G[x, y + 1]) or \
               (g <= G[x, y - 1]):

                I[x, y] = 0

        elif D[x, y]==45:
            if (g <= G[x - 1, y + 1]) or \
               (g <= G[x + 1, y-1]):

                I[x, y] = 0

        elif D[x, y]==90:
            if (g <= G[x + 1, y]) or \
               (g <= G[x - 1, y]):
                I[x, y] = 0

        else:
            if (g <= G[x + 1, y + 1]) or \
               (g <= G[x - 1, y - 1]):

                I[x, y]=0


G = None
G_x = None
G_y = None
gc.collect()

step_2 = time.time() - start_time - step_1 - step_0
print("step 2: %s" % step_2)

# PART 3: thresholding
# - using two threshold values to connect edges that we find

#   for each pixel:
#       if I[x,y] < s, the low threshold, mark with NO_EDGE value
#       if I[x,y] > s and I[x,y] < t, the high threshold, mark with WEAK_EDGE value
#       if I[x,y] > t, mark with STRONG_EDGE value

for x in range(h):
    for y in range(k):
        s = THRESHOLD_LOW * max
        t = THRESHOLD_HIGH * max
        u = I[x, y]

        if u < s:
            I[x, y] = NO_EDGE

        if s < u < t:
            I[x, y] = WEAK_EDGE

        if u > t:
            I[x, y] = STRONG_EDGE


step_3 = time.time() - start_time - step_2 - step_1 - step_0
print("step 3: %s" % step_3)

#   at a STRONG_EDGE pixel:
#       visit the weak pixels in the 8 adjacent spaces around the pixel and mark them as STRONG_EDGE
#           use DFS to avoid not visiting pixels that have been visited previously

def check_pixel(x, y):
    O[x, y] = STRONG_EDGE
    try:
        if I[x, y] == WEAK_EDGE:

            if I[x + 1, y] == WEAK_EDGE:
                check_pixel( x + 1, y)

            if I[x + 1, y - 1] == WEAK_EDGE:
                check_pixel( x + 1, y - 1)

            if I[x, y - 1] == WEAK_EDGE:
                check_pixel( x, y - 1)

            if I[x - 1, y - 1] == WEAK_EDGE:
                check_pixel( x - 1, y - 1)

            if I[x - 1, y] == WEAK_EDGE:
                check_pixel( x - 1, y)

            if I[x - 1, y + 1] == WEAK_EDGE:
                check_pixel( x - 1, y + 1)

            if I[x + 1, y - 1] == WEAK_EDGE:
                check_pixel( x, y + 1)

            if I[x + 1, y + 1] == WEAK_EDGE:
                check_pixel(x + 1, y + 1)

    except Exception:
        pass

for x in range(h):
    for y in range(k):
        if I[x, y] == STRONG_EDGE:
            check_pixel(x, y)

step_4 = time.time() - step_3 - start_time - step_2 - step_1 - step_0
print("step 4: %s" % step_4)

print("%s seconds" % (time.time() - start_time))

scipy.misc.imsave(OUTPATH + OUTFILENAME, O)
