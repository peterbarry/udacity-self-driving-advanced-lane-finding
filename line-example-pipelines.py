import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


image = mpimg.imread('bridge_shadow.jpg')
# Edit this function to create your own pipeline.
def pipeline_2(img, s_thresh=(170, 255), sx_thresh=(20, 100), red_thresh=(200,255),rx_thresh_r=(50, 100)):
    img = np.copy(img)

    R_channel = img[:,:,0]
    #G_channel = image[:,:,1]
    #B_channel = image[:,:,2]

    # Sobel x
    sobelx_r = cv2.Sobel(R_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx_r = np.absolute(sobelx_r) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_r = np.uint8(255*abs_sobelx_r/np.max(abs_sobelx_r))

    # Threshold x gradient
    rxbinary_r = np.zeros_like(scaled_sobel_r)
    rxbinary_r[(scaled_sobel_r >= rx_thresh_r[0]) & (scaled_sobel_r <= rx_thresh_r[1])] = 1

    # Threshold color channel
    r_binary_r = np.zeros_like(R_channel)
    r_binary_r[(R_channel >= red_thresh[0]) & (R_channel <= red_thresh[1])] = 1



    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel

    # Stack each channel - got 'white on lones is stacking is good'
    color_binary = np.dstack(( r_binary_r, sxbinary, s_binary))

    # Stack each channel - gradoent on red did not help much.

    #color_binary = np.dstack(( rxbinary_r, sxbinary, s_binary))

    return color_binary



# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary

# Edit this function to create your own pipeline.
def pipeline_3(img, s_thresh=(170, 255), sx_thresh=(20, 100),red_thresh=(170,255),rx_thresh_r=(20, 100)):
    img = np.copy(img)

    R_channel = img[:,:,0]
    #G_channel = image[:,:,1]
    #B_channel = image[:,:,2]

    # Sobel x
    sobelx_r = cv2.Sobel(R_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx_r = np.absolute(sobelx_r) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_r = np.uint8(255*abs_sobelx_r/np.max(abs_sobelx_r))

    # Threshold x gradient
    rxbinary_r = np.zeros_like(scaled_sobel_r)
    rxbinary_r[(scaled_sobel_r >= rx_thresh_r[0]) & (scaled_sobel_r <= rx_thresh_r[1])] = 1

    # Threshold color channel
    r_binary_r = np.zeros_like(R_channel)
    r_binary_r[(R_channel >= red_thresh[0]) & (R_channel <= red_thresh[1])] = 1



    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( rxbinary_r, sxbinary, s_binary))

    return color_binary

# Edit this function to create your own pipeline.
def pipeline_4(img, s_thresh=(170, 255), sx_thresh=(20, 100),red_thresh=(170,255),rx_thresh_r=(20, 100)):
    img = np.copy(img)

    R_channel = img[:,:,0]
    #G_channel = image[:,:,1]
    #B_channel = image[:,:,2]

    # Sobel x
    sobelx_r = cv2.Sobel(R_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx_r = np.absolute(sobelx_r) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_r = np.uint8(255*abs_sobelx_r/np.max(abs_sobelx_r))

    # Threshold x gradient
    rxbinary_r = np.zeros_like(scaled_sobel_r)
    rxbinary_r[(scaled_sobel_r >= rx_thresh_r[0]) & (scaled_sobel_r <= rx_thresh_r[1])] = 1

    # Threshold color channel
    r_binary_r = np.zeros_like(R_channel)
    r_binary_r[(R_channel >= red_thresh[0]) & (R_channel <= red_thresh[1])] = 1

    r_combined_binary = np.zeros_like(rxbinary_r)
    r_combined_binary[(r_binary_r == 1) & (rxbinary_r == 1)] = 1


    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( r_combined_binary, sxbinary, s_binary))
    return color_binary

# Edit this function to create your own pipeline.
def pipeline_5(img, s_thresh=(170, 255), sx_thresh=(20, 50),red_thresh=(100,255),rx_thresh_r=(20,50)):
    img = np.copy(img)

    R_channel = img[:,:,0]
    #G_channel = image[:,:,1]
    #B_channel = image[:,:,2]

    # Sobel x
    sobelx_r = cv2.Sobel(R_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx_r = np.absolute(sobelx_r) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_r = np.uint8(255*abs_sobelx_r/np.max(abs_sobelx_r))

    # Threshold x gradient
    rxbinary_r = np.zeros_like(scaled_sobel_r)
    rxbinary_r[(scaled_sobel_r >= rx_thresh_r[0]) & (scaled_sobel_r <= rx_thresh_r[1])] = 1

    # Threshold color channel
    r_binary_r = np.zeros_like(R_channel)
    r_binary_r[(R_channel >= red_thresh[0]) & (R_channel <= red_thresh[1])] = 1

    r_combined_binary = np.zeros_like(rxbinary_r)
    r_combined_binary[(r_binary_r == 1) & (rxbinary_r == 1)] = 1


    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( r_combined_binary, sxbinary, s_binary))
    return color_binary

# Edit this function to create your own pipeline.
def pipeline_6(img, s_thresh=(170, 255), sx_thresh=(20, 50),red_thresh=(100,255),rx_thresh_r=(20,50)):
    img = np.copy(img)

    R_channel = img[:,:,0]
    #G_channel = image[:,:,1]
    #B_channel = image[:,:,2]

    # Sobel x
    sobelx_r = cv2.Sobel(R_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx_r = np.absolute(sobelx_r) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_r = np.uint8(255*abs_sobelx_r/np.max(abs_sobelx_r))

    # Threshold x gradient
    rxbinary_r = np.zeros_like(scaled_sobel_r)
    rxbinary_r[(scaled_sobel_r >= rx_thresh_r[0]) & (scaled_sobel_r <= rx_thresh_r[1])] = 1

    # Threshold color channel
    r_binary_r = np.zeros_like(R_channel)
    r_binary_r[(R_channel >= red_thresh[0]) & (R_channel <= red_thresh[1])] = 1

    r_combined_binary = np.zeros_like(rxbinary_r)
    r_combined_binary[(r_binary_r == 1) & (rxbinary_r == 1)] = 1


    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( r_combined_binary, sxbinary, s_binary))
    return color_binary

# Edit this function to create your own pipeline.
def pipeline_7(img, h_thresh=(170, 255), hx_thresh=(0,255),
            s_thresh=(170, 255), sx_thresh=(0, 255),
            red_thresh=(100,255),rx_thresh_r=(0,255)):
    img = np.copy(img)

    R_channel = img[:,:,0]
    #G_channel = image[:,:,1]
    #B_channel = image[:,:,2]

    # Sobel x
    sobelx_r = cv2.Sobel(R_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx_r = np.absolute(sobelx_r) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_r = np.uint8(255*abs_sobelx_r/np.max(abs_sobelx_r))

    # Threshold x gradient
    rxbinary_r = np.zeros_like(scaled_sobel_r)
    rxbinary_r[(scaled_sobel_r >= rx_thresh_r[0]) & (scaled_sobel_r <= rx_thresh_r[1])] = 1

    # Threshold color channel
    r_binary_r = np.zeros_like(R_channel)
    r_binary_r[(R_channel >= red_thresh[0]) | (R_channel <= red_thresh[1])] = 1

    r_combined_binary = np.zeros_like(rxbinary_r)
    r_combined_binary[(r_binary_r == 1) & (rxbinary_r == 1)] = 1


    # Convert to HLS color space and separate the  channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    s_combined_binary = np.zeros_like(sxbinary)
    s_combined_binary[(s_binary == 1) & (sxbinary == 1)] = 1

    # Sobel x
    h_sobelx = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    h_abs_sobelx = np.absolute(h_sobelx) # Absolute x derivative to accentuate lines away from horizontal
    h_scaled_sobel = np.uint8(255*h_abs_sobelx/np.max(h_abs_sobelx))

    # Threshold x gradient
    hxbinary = np.zeros_like(h_scaled_sobel)
    hxbinary[(h_scaled_sobel >= hx_thresh[0]) & (h_scaled_sobel <= hx_thresh[1])] = 1

    # Threshold color channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    h_combined_binary = np.zeros_like(sxbinary)
    h_combined_binary[(h_binary == 1) & (sxbinary == 1)] = 1


    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( r_combined_binary, s_combined_binary, h_combined_binary))

    color_binary = np.dstack(( r_combined_binary, sxbinary, s_binary))


    return color_binary

# Edit this function to create your own pipeline.



def pipeline_8(img, h_thresh=(170, 255), hx_thresh=(20,100),
            s_thresh=(170, 255), sx_thresh=(20, 100),
            red_thresh=(100,255),rx_thresh_r=(20,100),
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
    rxbinary_r = np.zeros_like(scaled_sobel_r)
    rxbinary_r[(scaled_sobel_r >= rx_thresh_r[0]) & (scaled_sobel_r <= rx_thresh_r[1])] = 1

    # Threshold color channel
    r_binary_r = np.zeros_like(R_channel)
    r_binary_r[(R_channel >= red_thresh[0]) | (R_channel <= red_thresh[1])] = 1

    r_combined_binary = np.zeros_like(rxbinary_r)
    r_combined_binary[(r_binary_r == 1) & (rxbinary_r == 1)] = 1


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
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    s_combined_binary = np.zeros_like(sxbinary)
    s_combined_binary[(s_binary == 1) & (sxbinary == 1)] = 1

    # Sobel x
    h_sobelx = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
    h_abs_sobelx = np.absolute(h_sobelx) # Absolute x derivative to accentuate lines away from horizontal
    h_scaled_sobel = np.uint8(255*h_abs_sobelx/np.max(h_abs_sobelx))

    # Threshold x gradient
    hxbinary = np.zeros_like(h_scaled_sobel)
    hxbinary[(h_scaled_sobel >= hx_thresh[0]) & (h_scaled_sobel <= hx_thresh[1])] = 1

    # Threshold color channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    h_combined_binary = np.zeros_like(sxbinary)
    h_combined_binary[(h_binary == 1) & (sxbinary == 1)] = 1


    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( r_combined_binary, s_combined_binary, h_combined_binary))


    color_binary = np.dstack(( rxbinary_r, sxbinary, s_binary))

    return color_binary

def pipeline_9(img, h_thresh=(170, 255), hx_thresh=(20,100),
            s_thresh=(170, 255), sx_thresh=(20, 100),
            red_thresh=(100,255),rx_thresh_r=(20,100),
            sobel_kernel=7):
    img = np.copy(img)

    R_channel = img[:,:,0]
    #G_channel = image[:,:,1]
    #B_channel = image[:,:,2]

    # Sobel x
    sobelx_r = cv2.Sobel(R_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx_r = np.absolute(sobelx_r) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_r = np.uint8(255*abs_sobelx_r/np.max(abs_sobelx_r))

    # Threshold x gradient
    rxbinary_r = np.zeros_like(scaled_sobel_r)
    rxbinary_r[(scaled_sobel_r >= rx_thresh_r[0]) & (scaled_sobel_r <= rx_thresh_r[1])] = 1

    # Threshold color channel
    r_binary_r = np.zeros_like(R_channel)
    r_binary_r[(R_channel >= red_thresh[0]) | (R_channel <= red_thresh[1])] = 1

    r_combined_binary = np.zeros_like(rxbinary_r)
    r_combined_binary[(r_binary_r == 1) & (rxbinary_r == 1)] = 1


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
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    s_combined_binary = np.zeros_like(sxbinary)
    s_combined_binary[(s_binary == 1) & (sxbinary == 1)] = 1

    # Sobel x
    h_sobelx = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
    h_abs_sobelx = np.absolute(h_sobelx) # Absolute x derivative to accentuate lines away from horizontal
    h_scaled_sobel = np.uint8(255*h_abs_sobelx/np.max(h_abs_sobelx))

    # Threshold x gradient
    hxbinary = np.zeros_like(h_scaled_sobel)
    hxbinary[(h_scaled_sobel >= hx_thresh[0]) & (h_scaled_sobel <= hx_thresh[1])] = 1

    # Threshold color channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    h_combined_binary = np.zeros_like(sxbinary)
    h_combined_binary[(h_binary == 1) & (sxbinary == 1)] = 1


    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( r_combined_binary, s_combined_binary, h_combined_binary))


    color_binary = np.dstack(( rxbinary_r, sxbinary, s_binary))

    return color_binary

def pipeline_10(img, h_thresh=(50, 60), hx_thresh=(20,100),
            s_thresh=(170, 255), sx_thresh=(20, 100),
            red_thresh=(230,255),rx_thresh_r=(20,100),
            sobel_kernel=9):
    img = np.copy(img)

    R_channel = img[:,:,0]
    #G_channel = image[:,:,1]
    #B_channel = image[:,:,2]

    # Sobel x
    sobelx_r = cv2.Sobel(R_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx_r = np.absolute(sobelx_r) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_r = np.uint8(255*abs_sobelx_r/np.max(abs_sobelx_r))

    # Threshold x gradient
    rxbinary_r = np.zeros_like(scaled_sobel_r)
    rxbinary_r[(scaled_sobel_r >= rx_thresh_r[0]) & (scaled_sobel_r <= rx_thresh_r[1])] = 1

    # Threshold color channel
    r_binary_r = np.zeros_like(R_channel)
    r_binary_r[(R_channel >= red_thresh[0]) | (R_channel <= red_thresh[1])] = 1

    r_combined_binary = np.zeros_like(rxbinary_r)
    r_combined_binary[(r_binary_r == 1) & (rxbinary_r == 1)] = 1


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
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    s_combined_binary = np.zeros_like(sxbinary)
    s_combined_binary[(s_binary == 1) & (sxbinary == 1)] = 1

    # Sobel x
    h_sobelx = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
    h_abs_sobelx = np.absolute(h_sobelx) # Absolute x derivative to accentuate lines away from horizontal
    h_scaled_sobel = np.uint8(255*h_abs_sobelx/np.max(h_abs_sobelx))

    # Threshold x gradient
    hxbinary = np.zeros_like(h_scaled_sobel)
    hxbinary[(h_scaled_sobel >= hx_thresh[0]) & (h_scaled_sobel <= hx_thresh[1])] = 1

    # Threshold color channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    h_combined_binary = np.zeros_like(sxbinary)
    h_combined_binary[(h_binary == 1) & (sxbinary == 1)] = 1


    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( r_combined_binary, s_combined_binary, h_combined_binary))


    #color_binary = np.dstack(( r_binary_r, sxbinary, s_binary))

    #color_binary = np.dstack(( r_combined_binary, h_binary, s_binary))
    color_binary = np.dstack(( r_combined_binary, h_binary, s_binary))

    return color_binary

def pipeline_11(img,
            s_thresh=(170, 255), sx_thresh=(20, 100),
            red_thresh=(0,255),rx_thresh_r=(20,100),
            sobel_kernel=3):
    img = np.copy(img)

    R_channel = img[:,:,0]
    #G_channel = image[:,:,1]
    #B_channel = image[:,:,2]

    # Sobel x
    # Threshold color channel
    r_binary_r = np.zeros_like(R_channel)
    r_binary_r[(R_channel >= red_thresh[0]) | (R_channel <= red_thresh[1])] = 1

    sobelx_r = cv2.Sobel(r_binary_r, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx_r = np.absolute(sobelx_r) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_r = np.uint8(255*abs_sobelx_r/np.max(abs_sobelx_r))

    # Threshold x gradient
    rxbinary = np.zeros_like(scaled_sobel_r)
    rxbinary[(scaled_sobel_r >= rx_thresh_r[0]) & (scaled_sobel_r <= rx_thresh_r[1])] = 1


    # Convert to HLS color space and separate the  channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Sobel x
    sobelx = cv2.Sobel(s_binary, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    # Sobel x
    h_sobelx = cv2.Sobel(h_binary, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
    h_abs_sobelx = np.absolute(h_sobelx) # Absolute x derivative to accentuate lines away from horizontal
    h_scaled_sobel = np.uint8(255*h_abs_sobelx/np.max(h_abs_sobelx))

    # Threshold x gradient
    hxbinary = np.zeros_like(h_scaled_sobel)
    hxbinary[(h_scaled_sobel >= hx_thresh[0]) & (h_scaled_sobel <= hx_thresh[1])] = 1


    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( r_combined_binary, s_combined_binary, h_combined_binary))


    #color_binary = np.dstack(( r_binary_r, sxbinary, s_binary))

    #color_binary = np.dstack(( r_combined_binary, h_binary, s_binary))
    color_binary = np.dstack(( rxbinary, hxbinary, sxbinary))

    return color_binary

max_pixel_value=255


def pipeline_12(img, h_thresh=(50, 60), hx_thresh=(20,100),
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

    imgplot = plt.imshow(np.dstack(( r_combined_binary,
            np.zeros_like(r_combined_binary),
            np.zeros_like(r_combined_binary))))
    plt.show(imgplot)


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
    gray_binary = np.zeros_like(h_channel)
    gray_binary[(r_combined_binary != 0 ) & (s_combined_binary != 0)] = 1
    return color_binary,gray_binary

result,gray_result = pipeline_12(image)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(48, 18))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

# ax2.imshow(result)
ax2.imshow(gray_result,cmap='gray')
ax2.set_title('Pipeline Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
