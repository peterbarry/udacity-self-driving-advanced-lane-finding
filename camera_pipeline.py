
# This code loads the images to calibrate and generates calibration data.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

#
debug = True
print('Loading camera calibration data')
dist_pickle = pickle.load( open( "camera_cal.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def image_camera_undistort(img):
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # uses BGR as loade
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def undistort_test():
    files = 'camera_cal/*.jpg'
    images = glob.glob(files)
    for fname in images:
        if debug == True:
            print("Undistorting Cal Image:  {}".format(fname))
        img = cv2.imread(fname)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # uses BGR as loade
        undist = image_camera_undistort(img)
        cv2.imwrite('output_images/undistorted_images-'+fname,undist)

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

max_pixel_value=255


def line_detect(img, h_thresh=(50, 60), hx_thresh=(20,100),
            s_thresh=(200, 255), sx_thresh=(20, 50),
            red_thresh=(220,255),rx_thresh_r=(20,50),
            sobel_kernel=3):
    img = np.copy(img)

    R_channel = img[:,:,0]
    #G_channel = image[:,:,1]
    #B_channel = image[:,:,2]

    # Sobel x
    sobelx_r = cv2.Sobel(R_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx_r = np.absolute(sobelx_r) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_r = np.uint8(255*abs_sobelx_r/np.max(abs_sobelx_r))

    # Threshold x gradient
    rxbinary = np.zeros_like(scaled_sobel_r)
    rxbinary[(scaled_sobel_r >= rx_thresh_r[0]) &
        (scaled_sobel_r <= rx_thresh_r[1])] = max_pixel_value

    # Threshold color channel
    r_binary = np.zeros_like(R_channel)
    r_binary[(R_channel >= red_thresh[0]) & (R_channel <= red_thresh[1])] = max_pixel_value

    r_combined_binary = np.zeros_like(rxbinary)
    r_combined_binary[(r_binary == max_pixel_value) | (rxbinary == max_pixel_value)] = max_pixel_value

    #imgplot = plt.imshow(np.dstack(( r_combined_binary,
    #        np.zeros_like(r_combined_binary),
    #            np.zeros_like(r_combined_binary))))
    #plt.show(imgplot)


    # Convert to HLS color space and separate the  channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = max_pixel_value # yellow

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = max_pixel_value

    s_combined_binary = np.zeros_like(sxbinary)
    s_combined_binary[(s_binary == max_pixel_value) | (sxbinary == max_pixel_value)] = max_pixel_value

    # Sobel x
    #h_sobelx = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
    #h_abs_sobelx = np.absolute(h_sobelx) # Absolute x derivative to accentuate lines away from horizontal
    #h_scaled_sobel = np.uint8(255*h_abs_sobelx/np.max(h_abs_sobelx))

    # Threshold x gradient
    #hxbinary = np.zeros_like(h_scaled_sobel)
    #hxbinary[(h_scaled_sobel >= hx_thresh[0]) & (h_scaled_sobel <= hx_thresh[1])] = max_pixel_value

    # Threshold color channel
    #h_binary = np.zeros_like(h_channel)
    #h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = max_pixel_value

    #h_combined_binary = np.zeros_like(sxbinary)
    #h_combined_binary[(h_binary == 1) | (hxbinary == 1)] = max_pixel_value


    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( r_combined_binary, s_combined_binary, h_combined_binary))


    #color_binary = np.dstack(( r_binary_r, sxbinary, s_binary))

    #color_binary = np.dstack(( r_combined_binary, h_binary, s_binary))

    color_binary = np.dstack(( np.zeros_like(r_combined_binary),r_combined_binary, s_combined_binary, ))
    gray = cv2.cvtColor(color_binary, cv2.COLOR_RGB2GRAY)
    #gray_binary = np.zeros_like(h_channel)
    #gray_binary[(r_combined_binary != 0 ) & (s_combined_binary != 0)] = 1

    # Threshold color channel
    gray_binary = np.zeros_like(R_channel)
    gray_binary[(gray >= 1)] = max_pixel_value

    gray_binary_3 = np.dstack((gray_binary,gray_binary,gray_binary))


    return color_binary,gray_binary,gray_binary_3



def first_fit_and_polyfit(binary_img):

    out_img = np.copy(binary_img)
    #gray = img[:,:,0]
    #binary_warped = np.zeros_like(gray)
    #print(binary_warped.shape)
    #binary_warped[(gray > 0) ] = 255


    binary_warped=binary_img

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    #histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # update for python 3.0 // instead of / to get an int.
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    print('Midpoint')
    print(midpoint)
    print('Left Base')
    print (leftx_base)
    print('Right Base')
    print (rightx_base)

    print('histogram')
    print(histogram.size)

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image

        #cv2.rectangle crashing
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2)
        # Identify the nonzero pixels in x and y within the window

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #print('run polly on')
    #print(lefty)
    #print(leftx)
    #print(righty)
    #print(rightx)


    if len(leftx) != 0:
        found_left = True
        left_fit = np.polyfit(lefty, leftx, 2)
        print ('Left lane detected')
        print(left_fit)
    else:
        found_left = False
        left_fit=[0,0,0]
        print ('No Left lane detected')

    if len(rightx) != 0:
        found_right = True
        right_fit = np.polyfit(righty, rightx, 2)
        print ('Right lane detected')
        print(right_fit)
    else:
        found_right = False
        right_fit=[0,0,0]
        print ('No right lane detected')


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    if found_left == True:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    if found_right == True:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return found_left, left_fit,found_right, right_fit,out_img




def pipeline_test():
    files = 'test_images/*.jpg'
    images = glob.glob(files)
    for fname in images:
        if debug == True:
            print("pipeline test  Cal Image:  {}".format(fname))
        img = cv2.imread(fname)


        img  = image_camera_undistort(img)

        warped = birdseye_image(img)

        colour_binary,gray_binary,gray_binary_3 = line_detect(warped)

        output_imagename = 'output_images/line-detect-'+fname
        print(output_imagename)
        cv2.imwrite(output_imagename,colour_binary)


        found_left, left_fit,foundr_right, right_fit,out_img = first_fit_and_polyfit(gray_binary_3)


        output_imagename = 'output_images/pipeline-'+fname
        print(output_imagename)
        cv2.imwrite(output_imagename,out_img)



#pipeline
pipeline_test()
