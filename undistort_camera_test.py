
# This code loads the images to calibrate and generates calibration data.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

#
debug = True
dist_pickle = pickle.load( open( "camera_cal.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def undistort_test():
    files = 'camera_cal/*.jpg'
    images = glob.glob(files)
    for fname in images:
        if debug == True:
            print("Undistorting Cal Image:  {}".format(fname))
        img = cv2.imread(fname)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # uses BGR as loade
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite('output_images/undistorted_images-'+fname,undist)



#pipeline
undistort_test()
