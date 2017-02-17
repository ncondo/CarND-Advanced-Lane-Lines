import pickle
import sys
from moviepy.editor import VideoFileClip
from lane_tracker import LaneTracker



def main(input_video):

    # Read in the saved matrix and distortion coefficients
    dist_pickle = pickle.load(open('./camera_cal/calibration_pickle.p', 'rb'))
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']

    # Create LaneTracker object with matrix and distortion coefficients
    lane_tracker = LaneTracker(mtx, dist)

    # Name of output video after applying lane lines
    output_video = input_video.split('.')[0]+'_output.mp4'

    # Apply lane lines to each frame of input video and save as new video file
    clip1 = VideoFileClip(input_video)
    video_clip = clip1.fl_image(lane_tracker.apply_lines)
    video_clip.write_videofile(output_video, audio=False)



if __name__=='__main__':

    # argv[1] should be name of input video to apply lane lines
    main(sys.argv[1])

    



