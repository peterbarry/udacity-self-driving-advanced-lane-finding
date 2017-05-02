
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

def birdseye_image(img):

    img_size = (img.shape[1], img.shape[0])

    src = np.float32([[240,720],
                     [580,450],
                     [720,450],
                     [1165,720]])
    # Dest coords for perspective xform
    dst = np.float32([[300,720],
                     [300,0],
                     [900,0],
                     [900,720]])

    src = np.float32([[240,720],
                     [600,450],
                     [695,450],
                     [1165,720]])
    # Dest coords for perspective xform
    dst = np.float32([[300,720],
                     [300,0],
                     [900,0],
                     [900,720]])


    #src = np.float32([[240,719],
#                      [600,450],
#                      [690,450],
#                      [1165,719]])
#                         # Dest coords for perspective xform
#    dst = np.float32([[240,719],
#                      [240,0],
#                      [1165,0],
#                      [1165,719]])


    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped


def birdseye_test():
    files = 'test_images/*.jpg'
    images = glob.glob(files)
    for fname in images:
        if debug == True:
            print("birds eye view for  Cal Image:  {}".format(fname))
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # uses BGR as loade
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        img_size = (gray.shape[1], gray.shape[0])

        warped = birdseye_image(img)


        output_imagename = 'output_images/birdseye-'+fname
        print(output_imagename)
        cv2.imwrite(output_imagename,warped)



#pipeline
birdseye_test()
