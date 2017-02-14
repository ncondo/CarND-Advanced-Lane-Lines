import numpy as np
import cv2
import pickle
import glob
from tracker import Tracker


# Read in the saved matrix and distortion coefficients
dist_pickle = pickle.load(open('./camera_cal/calibration_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Returns an array of the same size as the input image of ones where gradients
    were in the threshold range, and zeros everywhere else.
    :param img: input image in rgb format.
    :param orient: orientation in which to take the gradient (x or y).
    :param sobel_kernel: size of the sobel kernel to apply (must be odd number >= 3).
    :param thresh: threshold (0 to 255) for determining which gradients to include when creating binary output.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the absolute value of the derivative in the given x or y orientation
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Scale to 8-bit (0-255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude is within the given thresholds
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return binary output image
    return grad_binary


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    """
    Returns an array of the same size as the input image of ones where gradients
    were in the threshold range, and zeros everywhere else.
    :param img: input image in rgb format.
    :param sobel_kernel: size of the sobel kernel to apply (must be odd number >= 3).
    :param thresh: threshold (0 to 255) for determining which gradients to include when creating binary output.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude of the gradient
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Scale to 8-bit (0-255) then convert to type = np.uint8
    scaled_gradmag = np.uint8(255*gradmag/np.max(gradmag))
    # Create a mask of 1's where the scaled gradient magnitude is within the given thresholds
    mag_binary = np.zeros_like(scaled_gradmag)
    mag_binary[(scaled_gradmag >= thresh[0]) & (scaled_gradmag <= thresh[1])] = 1
    # Return binary output image
    return mag_binary


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Returns a binary image of the same size as the input image of ones where gradient directions
    were in the threshold range, and zeros everywhere else.
    :param img: input image in rgb format.
    :param sobel_kernel: size of the sobel kernel to apply (must be odd number >= 3).
    :param thresh: threshold (0 to pi/2) for determining which gradients to include when creating binary output.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate direction of gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a mask of 1's where the gradient direction is within the given thresholds
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return binary output image
    return dir_binary


def color_thresh(img, s_thresh=(0, 255)):
    """
    Returns a binary image of the same size as the input image of ones where pixel values
    were in the threshold range, and zeros everywhere else.
    :param img: input image in rgb format.
    :param s_thresh: threshold (0 to 255) for determining which pixels from s_channel to include when creating binary output.
    """
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Apply a threshold to the S channel
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    # Create a mask of 1's where pixel value is within the given thresholds
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Return binary output image
    return s_binary


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), max(0, int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output


# Make a list of test images
images = glob.glob('./test_images/test*.jpg')

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    # Undistort the image
    img = cv2.undistort(img, mtx, dist, None, mtx)
    # Process image and generate binary pixels of interest
    preprocess_image = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255))
    grady = abs_sobel_thresh(img, orient='y', thresh=(25,255))
    color_binary = color_thresh(img, s_thresh=(100,255))
    preprocess_image[((gradx == 1) & (grady ==1) | (color_binary == 1))] = 255

    # Define perspective transform area
    img_size = (img.shape[1], img.shape[0])
    bot_width = .76 # percent of bottom trapizoid width
    mid_width = .08 # percent of middle trapizoid width
    height_pct = .62 # percent of trapizoid height
    bottom_trim = .935 # percent from top to bottom to exclude hood of car
    src = np.float32([[img.shape[1]*(.5-mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+bot_width/2), img.shape[0]*bottom_trim], [img.shape[1]*(.5-bot_width/2), img.shape[0]*bottom_trim]])
    offset = img_size[0] * .25
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

    # Perform the transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preprocess_image, M, img_size, flags=cv2.INTER_LINEAR)

    # Define parameters for lane tracking
    window_width = 25
    window_height = 80

    # Set up class to do all the lane tracking
    curve_centers = Tracker(window_width=window_width, window_height=window_height, margin=25, y_m_per_pixel=10/720, x_m_per_pixel=4/384, smooth_factor=15)

    window_centroids = curve_centers.find_window_centroids(warped)

    # Points used to draw all the left and right windows
    left_points = np.zeros_like(warped)
    right_points = np.zeros_like(warped)

    # Points used to find the left and right lanes
    left_x = []
    right_x = []

    # Go through each level and draw the windows
    for level in range(0, len(window_centroids)):
        # window_mask is a function to draw window areas
        left_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        right_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        # Add center value found in frame to the list of lane points per left, right
        left_x.append(window_centroids[level][0])
        right_x.append(window_centroids[level][1])
        # Add graphic points from window mask here to total pixels found
        left_points[(left_points == 255) | ((left_mask == 1))] = 255
        right_points[(right_points == 255) | ((right_mask == 1))] = 255

    # Draw the results
    template = np.array(right_points+left_points, np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8) # making the original road pixels 3 color channels
    curve_boxes = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the original road image with window results

    # Fit the lane boundaries to the left, right center positions found
    y_vals = range(0, warped.shape[0])
    res_y_vals = np.arange(warped.shape[0]-(window_height/2), 0, -window_height)

    left_fit = np.polyfit(res_y_vals, left_x, 2)
    left_fit_x = left_fit[0]*y_vals*y_vals + left_fit[1]*y_vals + left_fit[2]
    left_fit_x = np.array(left_fit_x, np.int32)

    right_fit = np.polyfit(res_y_vals, right_x, 2)
    right_fit_x = right_fit[0]*y_vals*y_vals + right_fit[1]*y_vals + right_fit[2]
    right_fit_x = np.array(right_fit_x, np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fit_x-window_width/2, left_fit_x[::-1]+window_width/2), axis=0), np.concatenate((y_vals, y_vals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fit_x-window_width/2, right_fit_x[::-1]+window_width/2), axis=0), np.concatenate((y_vals, y_vals[::-1]), axis=0))), np.int32)
    middle_marker = np.array(list(zip(np.concatenate((right_fit_x-window_width/2, right_fit_x[::-1]+window_width/2), axis=0), np.concatenate((y_vals, y_vals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 1.0, 0.0)

    y_m_per_pixel = curve_centers.y_m_per_pixel # meters per pixel in y dimension
    x_m_per_pixel = curve_centers.x_m_per_pixel # meters per pixel in x dimension

    curve_fit_cr = np.polyfit(np.array(res_y_vals, np.float32)*y_m_per_pixel, np.array(left_x, np.float32)*x_m_per_pixel, 2)
    curverad = ((1 + (2*curve_fit_cr[0]*y_vals[-1]*y_m_per_pixel + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])

    # Calculate offset of the car between the lane
    camera_center = (left_fit_x[-1] + right_fit_x[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*x_m_per_pixel
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    # draw the text showing curvature, offset, and speed
    cv2.putText(result, 'Radius of curvature = ' + str(round(curverad, 3))+' (m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(result,'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    # Save undistorted files to disk
    write_name = './test_images/tracked'+str(idx+1)+'.jpg'
    cv2.imwrite(write_name, result)


