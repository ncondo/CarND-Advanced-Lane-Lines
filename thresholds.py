import numpy as np
import cv2



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


def color_thresh(img, r_thresh=(0, 255), s_thresh=(0, 255)):
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


