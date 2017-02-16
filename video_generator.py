import numpy as np
import cv2
import pickle
import glob
from moviepy.editor import VideoFileClip


# Read in the saved matrix and distortion coefficients
dist_pickle = pickle.load(open('./camera_cal/calibration_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    """
    Returns an array of the same size as the input image of ones where gradients
    were in the threshold range, and zeros everywhere else.
    :param img: input image in BGR format.
    :param sobel_kernel: size of the sobel kernel to apply (must be odd number >= 3).
    :param thresh: threshold (0 to 255) for determining which gradients to include when creating binary output.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    :param img: input image in BGR format.
    :param sobel_kernel: size of the sobel kernel to apply (must be odd number >= 3).
    :param thresh: threshold (0 to pi/2) for determining which gradients to include when creating binary output.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


def color_thresh(img, r_thresh=(2, 255), s_thresh=(0, 255)):
    """
    Returns a binary image of the same size as the input image of ones where pixel values
    were in the threshold range, and zeros everywhere else.
    :param img: input image in BGR format.
    :param r_thresh: threshold (0 to 255) for determining which pixels from r_channel to include in binary output.
    :param s_thresh: threshold (0 to 255) for determining which pixels from s_channel to include in binary output.
    """
    # Apply a threshold to the R channel
    r_channel = img[:,:,2]
    r_binary = np.zeros_like(img[:,:,0])
    # Create a mask of 1's where pixel value is within the given thresholds
    r_binary[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Apply a threshold to the S channel
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    # Create a mask of 1's where pixel value is within the given thresholds
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine two channels
    combined = np.zeros_like(img[:,:,0])
    combined[(s_binary == 1) | (r_binary == 1)] = 1
    # Return binary output image
    return combined


def process_image(img):

    # Undistort the image
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # Threshold gradient
    grad_binary = np.zeros_like(img[:,:,0])
    mag_binary = mag_thresh(img, sobel_kernel=9, thresh=(50, 255))
    dir_binary = dir_thresh(img, sobel_kernel=15, thresh=(0.7, 1.3))
    grad_binary[((mag_binary == 1) & (dir_binary == 1))] = 1

    # Threshold color
    color_binary = color_thresh(img, r_thresh=(220, 255), s_thresh=(150, 255))

    # Combine gradient and color thresholds
    combo_binary = np.zeros_like(img[:,:,0])
    combo_binary[(grad_binary == 1) | (color_binary == 1)] = 255

    # Define perspective transform area
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1]/2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1]/2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])

    # Perform the transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    warped_binary = cv2.warpPerspective(combo_binary, M, img_size, flags=cv2.INTER_LINEAR)

    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped_binary[int(warped_binary.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped_binary, warped_binary, warped_binary))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped_binary.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_binary.nonzero()
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
    left_lane_idx = []
    right_lane_idx = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_binary.shape[0] - (window+1)*window_height
        win_y_high = warped_binary.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2)
        cv2.rectangle(out_img, (win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2)
        # Identify the nonzero pixels in x and y within the window
        good_left_idx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_idx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_idx.append(good_left_idx)
        right_lane_idx.append(good_right_idx)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_idx) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_idx]))
        if len(good_right_idx) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_idx]))

    # Concatenate the arrays of indices
    left_lane_idx = np.concatenate(left_lane_idx)
    right_lane_idx = np.concatenate(right_lane_idx)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_idx]
    lefty = nonzeroy[left_lane_idx]
    rightx = nonzerox[right_lane_idx]
    righty = nonzeroy[right_lane_idx]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines
    warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

    # Warp the blank back to original image space using inverse perspective matrix
    unwarped = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original undistorted image
    result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)

    # Compute the raduis of curvature of lane
    # Measure radius of curvature at y-value closest to car
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    # Fit polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Sample output of radius of curvature
    #print(left_curverad, 'm', right_curverad, 'm')
    curverad = (left_curverad+right_curverad)/2

    # Calculate offset of the car between the lane
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-unwarped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    # draw the text showing curvature, offset, and speed
    cv2.putText(result, 'Radius of curvature = ' + str(round(curverad, 3))+' (m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(result,'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    return result


if __name__=='__main__':

    # Input and output videos to apply lane lines
    input_video = 'project_video.mp4'
    output_video = 'output1_tracked.mp4'

    # Apply lane lines to each frame of input video and save as new video file
    clip1 = VideoFileClip(input_video)
    video_clip = clip1.fl_image(process_image)
    video_clip.write_videofile(output_video, audio=False)



