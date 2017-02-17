import pickle
import glob
import cv2
from lane_tracker import LaneTracker



def main():
    # Read in the saved matrix and distortion coefficients
    dist_pickle = pickle.load(open('./camera_cal/calibration_pickle.p', 'rb'))
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']

    # Create LaneTracker object with matrix and distortion coefficients
    lane_tracker = LaneTracker(mtx, dist)

    images = glob.glob('./test_images/test*.jpg')
    images.append('./test_images/straight_lines1.jpg')
    images.append('./test_images/straight_lines2.jpg')

    for idx, fname in enumerate(images):
        # Read in image
        img = cv2.imread(fname)

        result = lane_tracker.apply_lines(img)

        write_name = './output_images/final'+str(idx+1)+'.jpg'
        cv2.imwrite(write_name, result)



if __name__=='__main__':

    main()