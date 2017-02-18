# Advanced Lane Finding for Self-Driving Cars

The goal of this project is to produce a robust pipeline for detecting lane lines given a raw image from a car's dashcam. The pipeline should output a visual display of the lane boundaries, numerical estimation of lane curvature, and vehicle position within the lane.

![Original Image](test_images/test_example1.jpeg)   ![Output Image](output_images/output_example1.jpeg)


## Files and Usage

1. camera_cal.py
    * Contains code for calibrating the camera to undistort images.
    * `python camera_cal.py` will return the camera matrix and distortion coefficients and save them in a pickle file calibration_pickle.p.
2. thresholds.py
    * Contains code for applying color and gradient thresholds to an image to better detect lane lines.
3. lane_tracker.py
    * Contains code for identifying lane lines and highlighting the lane boundaries.
4. video_generator.py
    * Contains code to generate a video with lane boundaries applied to an input video from a dashcam.
    * `python video_generator.py project_video.mp4` will save the output video as project_video_output.mp4 in the same directory.
5. image_generator.py
    * Contains code to generate images with lane boundaries applied to the test images from a dashcam.
    * `python image_generator.py` will save the output images in the output_images folder.

## Solution

### Overview

The steps taken to complete this project are as follows:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms and gradients to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Camera Calibration

The code for this step is contained in `camera_cal.py` and the sample images and outputs can be found in the camera_cal folder.

I started by preparing "object points", which will be the (x,y,z) coordinates of the chessboard corners in the world. The provided sample images of chessboards are fixed on the (x,y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all of the chessboard corners are successfully detected in a sample image. With each successful chessboard detection, `imgpoints` will be appended with the (x,y) pixel position of each of the corners. 
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the OpenCV `calibrateCamera()` function. The resulting camera matrix and distortion coefficients are then used to undistort images using the OpenCV `undistort()` function. Here an original image (left) and an undistorted image (right):

![Original](camera_cal/test_example2.jpeg)      ![Undistorted](camera_cal/output_example2.jpeg)


### Distortion Correction

Using the camera matrix and distortion coefficients produced in the previous step, I undistort all incoming raw images using the OpenCV `undistort()` function. I use this function in my `apply_lines()` function which can be seen on line 38 of the `lane_tracker.py` file. Notice the difference in position of the white car between the raw image (left) and undistorted image (right):

![Original Image](test_images/test_example1.jpeg)   ![Undistorted](output_images/example_undist1.jpeg)


### Thresholded Binary Images

The code for producing the thresholded binary images can be found in `thresholds.py` and some sample output images can be found in the output_images folder.

In order to accurately find the lane lines in an image, I applied a number of thresholding techniques to filter out potential noise (such as shadows, different color lanes, other cars, etc). I first applied a color threshold, where I save only the R (red) channel of the RGB image, and combine it with the S (saturation) channel of the image after converting it to HLS space. The reason I keep the red channel is because it does a good job of preserving the lanes in the image, but especially the yellow lane which other filters sometimes fail to detect. The binary image of the R channel (left) and the S channel (right) can be seen below:

![R Binary](output_images/example_rthresh1.jpeg)  ![S Binary](output_images/example_sthresh1.jpeg)

Next, I apply thresholds on the gradients using the OpenCV `Sobel()` function. I apply a threshold on the magnitude of the gradient to filter out weak signals using the `mag_thresh()` function which can be found on line 6 of the `thresholds.py` file. I apply a threshold on the direction of the gradient in order to filter out horizonal lines, as the lane lines should be relatively vertical. You can find this `dir_thresh()` function on line 30 of the `thresholds.py` file. The binary image using the magnitude threshold (left) and directional threshold (right) can be seen below:

![Mag Binary](output_images/example_magthresh1.jpeg)  ![Dir Binary](output_images/example_dirthresh1.jpeg)

The binary images produced when combining both color thresholds (left) and both gradient thresholds (right):

![Color Binary](output_images/example_colorthresh1.jpeg)  ![Grad Binary](output_images/example_gradthresh1.jpeg)

I then combine the color and gradient thresholded binary images to produce the final binary image used in the pipeline for detecting the lane lines. You can see the original undistorted image (left) compared with the thresholded binary image (right) below:

![Color Binary](output_images/example_undist2.jpeg)  ![Grad Binary](output_images/example_combothresh1.jpeg)


### Perspective Transform

