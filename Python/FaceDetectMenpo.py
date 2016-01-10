
import os
import cv2
import numpy

from menpodetect import load_dlib_frontal_face_detector

from UtilsImgProc import GreyscaleConversionMenpo

class FaceDetectMenpo:

    def __init__(self, i_debug = False):

        self.debug = i_debug
        
        self.Train()

    '''
    Instantiate cascade classifier with pre-trained openCV face cascades
    '''
    def Train(self):
        
        self.face_detector = load_dlib_frontal_face_detector()
        
    '''
    Detect faces in an image.

    \param[in]  i_img               Menpo image
    
    \return Array of detected faces as rects with features
    '''
    def Detect(self, i_img):

        i_img = GreyscaleConversionMenpo(i_img)
        self.face_detector(i_img)

        ret = None
        if 'dlib_0' in i_img.landmarks:
            ret = i_img.landmarks['dlib_0'].lms.bounding_box()

        return ret
