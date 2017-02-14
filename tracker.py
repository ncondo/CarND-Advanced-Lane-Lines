import numpy as np
import cv2


class Tracker:

    def __init__(self, window_width, window_height, margin, y_m_per_pixel=1, x_m_per_pixel=1, smooth_factor=15):
        # list that stores all the past (left, right) center set values used for smoothing output
        self.recent_centers = []
        # the window pixel width of the center values, used to count pixels inside center windows to determine curve values
        self.window_width = window_width
        # the window pixel height of the center values, used to count pixels inside center windows to determine curve values
        # breaks the image into vertical levels
        self.window_height = window_height
        # The pixel distance in both directions to slide (left_window + right_window) template for searching
        self.margin = margin
        # Meters per pixel in vertical axis
        self.y_m_per_pixel = y_m_per_pixel
        # Meters per pixel in horizontal axis
        self.x_m_per_pixel = x_m_per_pixel
        # Number of frames to average over
        self.smooth_factor = smooth_factor


    def find_window_centroids(self, warped):

        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin

        # Store the (left,right) window centroid positions per level
        window_centroids = []
        window = np.ones(window_width)

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and the np.convolve the vertical image slice with the window template

        # Sum the quarter bottom of image to get slice, could use a different ratio
        left_sum = np.sum(warped[int(3*warped.shape[0]/4):, :int(warped.shape[1]/2)], axis=0)
        left_center = np.argmax(np.convolve(window, left_sum)) - window_width/2
        right_sum = np.sum(warped[int(3*warped.shape[0]/4):, int(warped.shape[1]/2):], axis=0)
        right_center = np.argmax(np.convolve(window, right_sum)) - window_width/2 + int(warped.shape[1]/2)

        # Add what we found for the first layer
        window_centroids.append((left_center, right_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(warped.shape[0]/window_height)):
            # Convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height), :], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            left_min_index = int(max(left_center+offset-margin, 0))
            left_max_index = int(min(left_center+offset+margin, warped.shape[1]))
            left_center = np.argmax(conv_signal[left_min_index:left_max_index])+left_min_index-offset
            # Find the best right centroid by using past right center as a reference
            right_min_index = int(max(right_center+offset-margin, 0))
            right_max_index = int(min(right_center+offset+margin, warped.shape[1]))
            right_center = np.argmax(conv_signal[right_min_index:right_max_index])+right_min_index-offset
            # Add what we found for that layer
            window_centroids.append((left_center, right_center))

        self.recent_centers.append(window_centroids)
        # Return averaged values of the line centers, helps to keep the markers from jumping around too much
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)




        