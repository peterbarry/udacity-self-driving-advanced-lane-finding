## Udacity Car-nd Term 1 Advanced Lane finding.


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)



[imageA]: ./camera_cal/calibration2.jpg "Calibration data"
[imageB]: ./output_images/chessboard_corners-camera_cal/calibration2.jpg "Corners data"
[imageC]: ./output_images/undistorted_images-camera_cal/calibration2.jpg "compensated image"

[image1]: ./test_images/test5.jpg "Road Image"
[image2]: ./output_images/birdseye-test_images/test5.jpg "Birds Eye Transformed"

[image3]: ./output_images/line-detect-test_images/test5.jpg "Soble Line detect"
[image4]: ./output_images/histogram-first-pass-test_images/test5.jpg "Histogram - First pass"
[image5]: ./output_images/warped-lanes-test_images/test5.jpg "Warped lanes"
[image6]: ./output_images/unwarped-lanes-test_images/test5.jpg "UnWarped lanes"
[image7]: ./output_images/pipeline-test_images/test5.jpg "Final overlay lanes"



[video1]: ./project_video-lines-output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The
The code for this step is located  file called `calibrate_camera.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
When searching for chessboard corners, the function requires knowledge of the number of inside corners.  For this calibration data x=9, and y = 6.  

Example Camera Calibration Image
![alt text][imageA]

Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Identification of chess board pattern.
![alt text][imageB]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:
```python
rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
```
Example of a corrected image.
![alt text][imageC]

The camera calibration data is then saved to a pickle file to be loaded by the camera pipeline.
```python
pickle_dict = { "rms": rms,
                "mtx": mtx,
                "dist":dist,
                "rvecs":rvecs,
                "tvecs":tvecs
                 }

print (pickle_dict)

pickle.dump( pickle_dict, open( "camera_cal.p", "wb" ) )


```


### Pipeline (single images)

#### 1. Example Input Image


![alt text][image1]

### Camera correction and birds eye view.

The code for my perspective transform I generate the perspective transform values m/Minv in the init code for LaneHistory, I chose the hardcode the source and destination points in the following manner:

```python
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

```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 240, 720      | 300, 7200        |
| 600, 450      | 300, 0      |
| 695, 450     | 900, 0      |
| 1165, 720      | 900, 720        |

![alt text][image2]

It should be noted this values took some time to establish through a lot of trial and error. I used the straight lines to visually verify that approx parallel lines are generated.

The code to transform an iamge using the calculated M matrix is
```python
def birdseye_image(img):

    img_size = (img.shape[1], img.shape[0])
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, lane_history.get_M(), img_size)
    return warped
```

#### 2. Thresholding, Sobel - X  transforms

I used a combination of color and gradient thresholds to generate a binary image.

I used to colour spaces RGB and HLS, using test image inspection I selected to selected the best channels for lighting,shadow and lane colour.

The code for the conversion is in the function line_detect() in video_camera_pipeline.py.

 The red channel of RGB colour space and the Saturation channel from the HLS space.

Sobel was run on both of these channels, focusing on X direction to extract lane lines. The sobel output was thresholded and each was merged into a colour image as shown here.
![alt text][image3]

This image was grayscaled and converter to a single Chanel binary output for line fitting.


#### 3. Historgram Searching for lines.
 The application uses a broad searching algorithm for the first image then uses a simplified search for all subsequent images in a video pipeline.
 ```python
 first_fit_and_polyfit(gray_binary)
OR
 subsequent_fit_and_polyfit(gray_binary,left_fit,
```
The fitting algorithms are reused from the classes.



I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Line  Fitting

The points within the historgram box search (left and right above) are provided to a line fit function.
``` python
        left_fit = np.polyfit(lefty, leftx, 2)
```      
This resulting lines are ploted into an image for  show here  (with a poly fill for the space in between)
![alt text][image5]

This image is in the warped transform space and we need to alter to have the same perspective as the driver sees. This is done using the Minv matrix.
``` python
unwarped_lanes = cv2.warpPerspective(warped_lanes, Minv, (warped_lanes.shape[1], warped_lanes.shape[0]))

```  

![alt text][image6]

We now have an image that can be merged into the original image to produce an overlay of the detected lanes.

![alt text][image7]
---

### Pipeline (video)

The pipeline is largely the same as those for images with the exception of using an optimized histogram search after the first image.
``` python

img_copy = np.copy(img)
lane_history.inc_frame_counter()
Minv = lane_history.get_Minv()

img  = image_camera_undistort(img)

warped = birdseye_image(img)

colour_binary,gray_binary,gray_binary_3 = line_detect(warped)


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

return result
```

Here's a [link to my video result](./project_video-lines-output.mp4)

---

### Discussion

#### 1. Colour space selection
 This took considerable time and is roust for the project video but wil lbe channelged for a lot of other examplesself.

 #### 1. Video Analysis
  The project pipeline does not use any sigificant time based checking, each image is treated in isolation. It performs well  here but not for the challenge videos. I would add significant frame to frame averaging and error checking.
