
import os
import cv2
import argparse
import menpo.image

from Utils import GetDirectory
from UtilsImgProc import DisplayOpenCVImage

from AAM import AAM
from TrackingOpticalFlow import Tracking


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # default to lfpw dataset script directory 
    dataset = os.path.join( GetDirectory(__file__), 'lfpw' )
        
    parser.add_argument('--debug', dest='debug', help='Flag for debug messages', action='store_true')
    parser.add_argument('--dataset', dest='dataset', help='True to enable debug messages', default=dataset)
    parser.add_argument('--video', dest='video', help='0 for USB, path for file', default=0)
    parser.add_argument('--fps', dest='fps', help='Frame rate', default=15)
    parser.add_argument('--detection_rate', dest='detection_rate', help='Rate of detection with face detector (secs)', default=1)

    args = parser.parse_args();

    # Train classifier
    aam = AAM(args.dataset, args.debug);
    tracker = Tracking();

    # Boot camera into default configuration
    cap = cv2.VideoCapture(args.video)

    pts = None
    frame_count = 0

    while(cap.isOpened()):
        ret,frame = cap.read()
        frame_count += 1

        if ret:

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = menpo.image.Image(img)

            if pts is None and i % args.fps * args.detection_rate != 0:
                pts = aam.FitWildImageMenpo(img)
            elif pts is not None:
                pts = aam.FitWildImageMenpo(img, pts)

            DisplayOpenCVImage(frame, 5, pts)
        

