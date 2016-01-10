
import os
import cv2
import numpy

from Utils import GetDirectory
from UtilsImgProc import GreyscaleConversionOpenCV

class FaceDetectViola:

    def __init__(self, i_debug):

        self.debug = i_debug
        self.cascade_path = os.path.join(GetDirectory(__file__),'haarcascade_frontalface_default.xml')

        if not os.path.exists(self.cascade_path):
            raise RuntimeError('Could not locate cascade path ' + self.cascade_path)

        self.Train()

    '''
    Instantiate cascade classifier with pre-trained openCV face cascades
    '''
    def Train(self):
        
        self.classifier = cv2.CascadeClassifier(self.cascade_path)

    '''
    Detect faces in an image.

    \param[in]  i_img               OpenCV mat
    \param[in]  i_kernel_size       Gaussian smoothing kernel width
    \param[in]  i_downsample        Downsampling factor between image pyramid levels
    \param[in]  i_min_proportion    Percentage of image dimension to use for pyramids leaf image size
    \param[in]  i_max_proportion    Percentage of image dimension to use for pyramids root image size
    
    \return Array of detected faces as rects
    '''
    def Detect(self, i_img, i_kernel_size = 3, i_downsample = 1.1, i_min_proportion = 0.125, i_max_proportion = 1.0):

        i_img = GreyscaleConversionOpenCV(i_img)
        img = cv2.blur( i_img, (i_kernel_size,i_kernel_size) )
        img = numpy.array(img, dtype='uint8')

        # Increase image pyramid downsample factor and limit min/max
        # relative to image dimensions for computation gain (experience from Oxehealth)
        reject_levels = []
        level_weights = []

        size = numpy.array(i_img.shape)

        min_size = tuple( (size * i_min_proportion).astype(int) )
        max_size = tuple( (size * i_max_proportion).astype(int) )
        
        faces = self.classifier.detectMultiScale(
            image = img,
            rejectLevels = reject_levels,
            levelWeights = level_weights,
            scaleFactor = i_downsample,
            minNeighbors = 1,
            minSize = min_size,
            maxSize = max_size)

        if self.debug:
            print('reject_levels', reject_levels)
            print('level_weights', level_weights)

        return faces


        
