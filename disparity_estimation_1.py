#!/usr/bin/env python3

import cv2
import numpy as np

# Load left and right images as color images
left_image = cv2.imread('left.png')
right_image = cv2.imread('right.png')

# Convert images to grayscale
left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# Convert grayscale images to CV_8UC1 format
left_gray = cv2.convertScaleAbs(left_gray)
right_gray = cv2.convertScaleAbs(right_gray)

# Create StereoBM object
stereo = cv2.StereoBM_create()

# Set parameters
num_disparities = 256
block_size = 21
speckle_range = 1
speckle_window_size = 100
min_disparity = 6

# Set parameters in StereoBM object
stereo.setNumDisparities(num_disparities)
stereo.setBlockSize(block_size)
stereo.setSpeckleRange(speckle_range)
stereo.setSpeckleWindowSize(speckle_window_size)
stereo.setMinDisparity(min_disparity)

# Compute disparities & normalization
disparity_BM = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# Display the disparity map
cv2.imwrite('Disparity_Map_1.jpg', disparity_BM)
cv2.waitKey(0)
cv2.destroyAllWindows()
