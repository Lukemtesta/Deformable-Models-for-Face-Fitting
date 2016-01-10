
import cv2
import numpy

class Tracking:

    def __init__(self, i_debug = False):

        self.debug = i_debug
        self.old_img = None

        self.config = dict(
            winSize  = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def TrackFeaturePoints(self, i_img, i_old_pts):

        new_pts = None
        i_old_pts = numpy.array(i_old_pts, dtype = numpy.uint8)
        print(i_old_pts)

        i_img = numpy.array(i_img, dtype = numpy.uint8)
        print(i_img.shape)

        if self.old_img is not None and i_old_pts is not None:
            print('Dumped details')
            print(self.old_img.shape)
            print(i_img.shape)
            new_pts, ttl, err = cv2.calcOpticalFlowPyrLK(self.old_img, i_img, i_old_pts, None)
            new_pts = new_pts[ttl >= 1]
            print('new_pts', new_pts)
            print('ttl',ttl)

        self.old_img = i_img

        return new_pts
