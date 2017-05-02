
# This code loads the images to calibrate and generates calibration data.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# prepare object points
nx = 9#: enter the number of inside corners in x
ny = 6#: enter the number of inside corners in y


objp = np.zeros((nx*ny,3),np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # create a grid

debug = True

def calibrate_camera():
    objpoints= [] # 3d points in real sapce
    imgpoints= [] # 2d points in image space
    # Make a list of calibration images
    files = 'camera_cal/*.jpg'
    images = glob.glob(files)

    for fname in images:
        if debug == True:
            print("Processing Cal Image:  {}".format(fname))
        #   images = glob.glob(files)
        #img = cv2.imread(fname)
        # Convert to grayscale
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # uses BGR as loade

        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        cv2.imwrite('output_images/gray-'+fname,gray)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            cv2.imwrite('output_images/chessboard_corners-'+fname,img)
        else:
            print("Corners not found in {}".format(fname))


    shape = gray.shape[::-1] # img.shape[0:2] # assume last image is valid for corners.
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

# Save a dictionary into a pickle file.


    pickle_dict = { "rms": rms,
                    "mtx": mtx,
                    "dist":dist,
                    "rvecs":rvecs,
                    "tvecs":tvecs
                     }
                     
    print (pickle_dict)

    pickle.dump( pickle_dict, open( "camera_cal.p", "wb" ) )


    files = 'camera_cal/*.jpg'
    images = glob.glob(files)
    for fname in images:
        if debug == True:
            print("Undistorting Cal Image:  {}".format(fname))
        img = cv2.imread(fname)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # uses BGR as loade
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite('output_images/undistorted_images-'+fname,undist)


def image_undistort(img):
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # uses BGR as loade
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undis





#pipeline
calibrate_camera()
