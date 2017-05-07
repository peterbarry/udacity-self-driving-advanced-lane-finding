# This code loads the images to calibrate and generates calibration data.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


#
debug = True
print('Loading camera calibration data')
dist_pickle = pickle.load( open( "camera_cal.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Used to save some state history for the lane detection.
# Limited state at present used.
class LanesHistory:
    def __init__(self, debug_mode=False, show_plots=False):
        # Frame counter (used for finding new lanes)
        self.frame_number = 0
        self.left_fit=[0.0,0.0,0.0]
        self.right_fit=[0.0,0.0,0.0]

        src = np.float32([[240,720],
                             [600,450],
                             [695,450],
                             [1165,720]])
            # Dest coords for perspective xform
        dst = np.float32([[300,720],
                             [300,0],
                             [900,0],
                             [900,720]])
                                 # Given src and dst points, calculate the perspective transform matrix
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst,src )

    def inc_frame_counter(self):
        self.frame_number += 1

    def get_frame_counter(self):
        return self.frame_number

    def set_left_fit(self,fit):
        self.left_fit = fit
    def set_right_fit(self,fit):
        self.right_fit = fit

    def get_left_fit(self):
        return self.left_fit

    def get_right_fit(self):
        return self.right_fit

    def get_M(self):
            return self.M
    def get_Minv(self):
            return self.Minv

lane_history = LanesHistory()

# Use the image camera calibration to undistort camera image.
def image_camera_undistort(img):

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# Create a birs eye view of the road, using cv2 Perspective Transform.

def birdseye_image(img):

    img_size = (img.shape[1], img.shape[0])
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, lane_history.get_M(), img_size)
    return warped


max_pixel_value=255
def line_detect(img, h_thresh=(50, 60), hx_thresh=(20,100),
            s_thresh=(255, 255), sx_thresh=(20, 150),
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

    # Convert to HLS color space and separate the  channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    #h_channel = hsv[:,:,0]
    #l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Sobel x on the S channel
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = max_pixel_value # yellow

    # Threshold color channel
    #s_binary = np.zeros_like(s_channel)
    #s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = max_pixel_value

    s_combined_binary = np.zeros_like(sxbinary)
    # did not use the s-channel just used it for sobel.
    s_combined_binary[(sxbinary == max_pixel_value)] = max_pixel_value

    color_binary = np.dstack(( np.zeros_like(r_combined_binary),r_combined_binary, s_combined_binary, ))
    gray = cv2.cvtColor(color_binary, cv2.COLOR_RGB2GRAY)
    #gray_binary = np.zeros_like(h_channel)
    #gray_binary[(r_combined_binary != 0 ) & (s_combined_binary != 0)] = 1

    # Threshold color channel
    gray_binary = np.zeros_like(R_channel)
    gray_binary[(gray >= 1)] = max_pixel_value

    zeros_fill=np.zeros_like(R_channel)
    #gray_binary_3 = np.dstack((gray_binary,gray_binary,gray_binary))
    gray_binary_3 = np.dstack((gray_binary,zeros_fill,zeros_fill))

    # return a number of stacked results for debug and visualisation - gray_binary used for fitting.
    return color_binary,gray_binary,gray_binary_3



def first_fit_and_polyfit(binary_img):

# First time fitting - do exhaustive search.

    binary_warped=binary_img

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    #histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # update for python 3.0 // instead of / to get an int.
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255


    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

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

        #cv2.rectangle
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


    if len(leftx) != 0:
        found_left = True
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        found_left = False
        left_fit=[0,0,0]
        print ('*************No Left lane detected')

    if len(rightx) != 0:
        found_right = True
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        found_right = False
        right_fit=[0,0,0]
        print ('*************No right lane detected')


    # Calculate curvature
    y_eval = np.max(nonzeroy)
    #print(y_eval)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
            # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
            # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Generate x and y values for plotting
    ploty = np.linspace(200, binary_warped.shape[0]-1, binary_warped.shape[0] )
    if found_left == True:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    if found_right == True:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels

    # Generate a polygon
    # And recast the x and y points into usable format for cv2.fillPoly()
    margin = 10
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,0, 0))


    #fill in the road surface space
    infill_area_pts = np.hstack((left_line_window2, right_line_window1))
    cv2.fillPoly(window_img, np.int_([infill_area_pts]), (0,255, 0))


    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    #wraped  space mid point variation calculate
    lx = left_fitx[-1]
    rx = right_fitx[-1]
    mid = ((rx-lx)/2)+lx
    lane_width_pix = rx-lx
    image_mid = binary_warped.shape[1]/2
    lane_cms_per_pixel = 370.0 / lane_width_pix   # US  lane width = 3.7m
    dist_from_mid = ((mid - image_mid) * lane_cms_per_pixel)+46 #manual caibrated for warp offsets

    return  left_fit,right_fit,result,window_img,left_curverad, right_curverad,dist_from_mid

def subsequent_fit_and_polyfit(binary_img,left_fit,right_fit):

# optimsed search using previous polyfit.


    binary_warped=binary_img

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255


    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


        # Calculate curvature
    y_eval = np.max(nonzeroy)
        #print(y_eval)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

            # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

            # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
            # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Generate x and y values for plotting

    #Dont start plot at top of warped iamge.
    ploty = np.linspace(200, binary_warped.shape[0]-1, binary_warped.shape[0] )


    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels


    #
    # And recast the x and y points into usable format for cv2.fillPoly()
    margin = 10
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,0, 0))


    #fill in the road surface space
    infill_area_pts = np.hstack((left_line_window2, right_line_window1))
    cv2.fillPoly(window_img, np.int_([infill_area_pts]), (0,255, 0))


    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    #wapred space mid point variation calculate
    lx = left_fitx[-1]
    rx = right_fitx[-1]
    mid = ((rx-lx)/2)+lx
    lane_width_pix = rx-lx
    image_mid = binary_warped.shape[1]/2
    lane_cms_per_pixel = 370.0 / lane_width_pix   # US  lane width = 3.7m
    dist_from_mid = ((mid - image_mid) * lane_cms_per_pixel)+46 #manual caibrated for warp offsets

    return left_fit ,right_fit,result,window_img,left_curverad, right_curverad,dist_from_mid

def process_img(img):

            img_copy = np.copy(img)
            lane_history.inc_frame_counter()
            Minv = lane_history.get_Minv()

            img  = image_camera_undistort(img)

            warped = birdseye_image(img)

            colour_binary,gray_binary,gray_binary_3 = line_detect(warped)

            #add line detection overlay - debug
            #colour_binary_unwarped = cv2.warpPerspective(colour_binary, Minv, (colour_binary.shape[1], colour_binary.shape[0]))
            #img = cv2.addWeighted(img, 1, colour_binary_unwarped, 0.7, 0)

            if lane_history.get_frame_counter() == 1 :
                left_fit,right_fit,out_img,warped_lanes,left_curverad, right_curverad,dist_from_mid = first_fit_and_polyfit(gray_binary)
                lane_history.set_left_fit(left_fit)
                lane_history.set_right_fit(right_fit)
            else:
                left_fit = lane_history.get_left_fit()
                right_fit = lane_history.get_right_fit()
                left_fit,right_fit,out_img,warped_lanes,left_curverad, right_curverad,dist_from_mid = subsequent_fit_and_polyfit(gray_binary,left_fit,right_fit)
                lane_history.set_left_fit(left_fit)
                lane_history.set_right_fit(right_fit)

            unwarped_lanes = cv2.warpPerspective(warped_lanes, Minv, (warped_lanes.shape[1], warped_lanes.shape[0]))
            # Combine the result with the original campera processed image
            result = cv2.addWeighted(img, 1, unwarped_lanes, 0.3, 0)
            #result = img

            font = cv2.FONT_HERSHEY_SIMPLEX

            dist_str = "Distance from Center: {0:.2f} cms".format(dist_from_mid)
                #print(dist_str)
            cv2.putText(result, dist_str, (200,50), font, 1, (255,255,255), 2)

            left_roc_text = "Roc: {0:.2f} m".format(left_curverad)
            cv2.putText(result, left_roc_text, (20,650), font, 1, (255,255,255), 2)
            right_roc_text = "Roc: {0:.2f} m".format(right_curverad)
            cv2.putText(result, right_roc_text, (1000,650), font, 1, (255,255,255), 2)

            frame_counter_str = "Frame #: {0:2d}".format(lane_history.get_frame_counter())
            cv2.putText(result, frame_counter_str, (1000,50), font, 1, (255,0,0), 2)

            return result


def video_pipeline(input_filename,output_filename):

    clip1 = VideoFileClip(input_filename)
    out_clip = clip1.fl_image(process_img) #NOTE: this function expects color images!!
    out_clip.write_videofile(output_filename, audio=False)

def video_pipeline_test():
    video_pipeline('project_video.mp4','project_video-lines-output.mp4')
    video_pipeline('challenge_video.mp4','challenge_video-lines-output.mp4')
    video_pipeline('harder_challenge_video.mp4','harder_challenge_video-lines-output.mp4')

def images_pipeline_test():
    files = 'test_images/*.jpg'
    images = glob.glob(files)
    for fname in images:
        if debug == True:
            print("pipeline test  Cal Image:  {}".format(fname))
        img = cv2.imread(fname)
        img_copy = np.copy(img)


        img  = image_camera_undistort(img)

        warped = birdseye_image(img)
        Minv = lane_history.get_Minv()


        colour_binary,gray_binary,gray_binary_3 = line_detect(warped)

        output_imagename = 'output_images/line-detect-'+fname
        print(output_imagename)
        cv2.imwrite(output_imagename,colour_binary)


        left_fit,right_fit,out_img,warped_lanes,left_curverad, right_curverad,dist_from_mid  = first_fit_and_polyfit(gray_binary)



        output_imagename = 'output_images/histogram-first-pass-'+fname
        print(output_imagename)
        cv2.imwrite(output_imagename,out_img)

        output_imagename = 'output_images/warped-lanes-'+fname
        print(output_imagename)
        cv2.imwrite(output_imagename,warped_lanes)

        unwarped_lanes = cv2.warpPerspective(warped_lanes, Minv, (warped_lanes.shape[1], warped_lanes.shape[0]))
        # Combine the result with the original campera processed image
        result = cv2.addWeighted(img, 1, unwarped_lanes, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        dist_str = "Distance from Center: {0:.2f} cms".format(dist_from_mid)
        #print(dist_str)
        cv2.putText(result, dist_str, (200,50), font, 1, (255,255,255), 2)

        left_roc_text = "Roc: {0:.2f} m".format(left_curverad)
        cv2.putText(result, left_roc_text, (20,650), font, 1, (255,255,255), 2)
        right_roc_text = "Roc: {0:.2f} m".format(right_curverad)
        cv2.putText(result, right_roc_text, (1000,650), font, 1, (255,255,255), 2)

        output_imagename = 'output_images/unwarped-lanes-'+fname
        print(output_imagename)
        cv2.imwrite(output_imagename,unwarped_lanes)


        output_imagename = 'output_images/pipeline-'+fname
        print(output_imagename)
        cv2.imwrite(output_imagename,result)


run_images_pipeline = True
run_video_pipeline = True

if run_images_pipeline == True:
    images_pipeline_test()
if run_video_pipeline == True:
    video_pipeline_test()
