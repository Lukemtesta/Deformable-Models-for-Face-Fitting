
import unittest

import os
import numpy

from Utils import GetDirectory
from AAM import AAM
from FaceDetectViola import FaceDetectViola

import cv2
import menpo.io as menpoio

class TestMenpoSetup(unittest.TestCase):

    '''
    Test menpo core libraries are functioning correctly with a pre-defined model fit
    '''
    def test_menpo_core_dependencies(self):
        
        bb = menpoio.import_builtin_asset.breakingbad_jpg()
        bb = bb.crop_to_landmarks_proportion(0.5)

        # Annotations have 1 set of labels and 68 points for 724 x 679 3-channel image
        self.assertEquals( numpy.size(bb.pixels), 724 * 679 * 3)
        self.assertEquals( bb.landmarks['PTS'].n_labels, 1 )
        self.assertEquals( bb.landmarks['PTS'].n_landmarks, 68 )
        

class TestAAM(unittest.TestCase):

    '''
    Crops image landmarks (0.1 referenced from AAMs Basics) and convert to greyscale

    \param[in]  i_img   Menpo image to process
    \return processed menpo image
    '''
    def LoadImage(self, i_img, i_landmark_crop = 0.9):

        img = i_img.crop_to_landmarks_proportion(0.1)  
        if img.n_channels == 3:
            img = img.as_greyscale(mode='luminosity')
        return img

    '''
    Test the AAM is constructed correctly by fitting to a test image with annotated landmarks
    '''
    def test_trained_aam_default_dataset(self, i_image_count = 800):

        dataset = os.path.join( GetDirectory(__file__) , 'lfpw')

        Model = AAM(dataset, i_debug = True)

        testset = os.path.join(dataset, 'testset', '*')
        forward_backward_errors = [Model.FitAnnotatedImage(Model.LoadImage(img))
                                   for img in menpoio.import_images(testset, max_images=800, verbose = True) ]

        # Ensure mean error < 0.1 - Experimentally derived
        err = 0        
        for error in forward_backward_errors:
            err = err + error.final_error()

        self.assertTrue( err / len(forward_backward_errors) < 0.1 )


class TestViolaFaceDetect(unittest.TestCase):

    '''
    Test cascades load correctly and successfully detect single face in known image
    '''
    def test_detect_face(self):

        face_detector = FaceDetectViola(True);

        img_filepath = os.path.join( GetDirectory(__file__) , 'unit_test_viola_image.jpg')
        img = cv2.imread(img_filepath)
        
        faces = face_detector.Detect(img)
        self.assertEqual(len(faces), 1)

    '''
    Test cascade classifier configuration by loading cascade and not detecting face (face dims > maxSize)
    '''
    def test_default_detect_no_face(self):

        face_detector = FaceDetectViola(True);

        img_filepath = os.path.join( GetDirectory(__file__) , 'unit_test_viola_image.jpg')
        img = cv2.imread(img_filepath)
        
        faces = face_detector.Detect(img, 3, 1.3, 0.125, 0.5)
        self.assertEqual(len(faces), 0)
        

if __name__ == '__main__':
    unittest.main()
