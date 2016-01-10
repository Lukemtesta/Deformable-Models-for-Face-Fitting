
import os
import numpy

import menpo
import menpo.io as menpoio
from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter
from menpo.feature import fast_dsift, ndfeature

from FaceDetectMenpo import FaceDetectMenpo
from FaceDetectViola import FaceDetectViola
from UtilsImgProc import DisplayRects, GreyscaleConversionMenpo, ConvertRectToMenpoBoundingBox, GetLargestROI

'''
Dense SIFT feature detector
'''
@ndfeature
def float32_fast_dsift(i_img):
    return fast_dsift(i_img).astype(numpy.float32)


class AAM:

    '''
    Initialise members and train AAM

    \param[in]  i_dataset   Full filepath to training and test dataset
    \param[in]  i_debug     True to display debug info
    '''
    def __init__(self, i_dataset, i_debug = False):

        self.debug = i_debug
        self.dataset = i_dataset

        if not os.path.exists(self.dataset):
            raise RuntimeError('Database dir does not exist in ' + self.dataset)

        self.Train()

        self.viola_face_detector = FaceDetectViola(False)
        self.menpo_face_detector = FaceDetectMenpo()

        if self.debug:
            self.PrintDebug()

    '''
    Load training images and annotated landmarks from a training set in the file system
    '''
    def LoadDataset(self):

        trainset = os.path.join(self.dataset,'trainset','*')
        training_images = [self.LoadImage(img) for img in menpoio.import_images(trainset, verbose = True) ]

        return training_images

    '''
    Crops image landmarks (0.1 referenced from AAMs Basics) and convert to greyscale

    \param[in]  i_img   Menpo image to process
    \return processed menpo image
    '''
    def LoadImage(self, i_img, i_landmark_crop = 0.5):

        img = i_img.crop_to_landmarks_proportion( i_landmark_crop )  
        img = GreyscaleConversionMenpo(img)
        
        return img

    '''
    Train an Active Appearance Model and compute the

    \param[in]  i_diag                  Search gradient along model landmark
    \param[in]  i_scale                 Scale applied to search direction (search) || (initial, search)
    \param[in]  i_max_greyscale_dims    Dimensionality limit for PCA appearance model
    \param[in]  i_max_shape_dims        Dimensionality limit for PCA keypoint components
    '''
    def Train(self, i_diag = 150, i_scale = [0.5, 1.0], i_max_greyscale_dims = 200, i_max_shape_dims = 20):

        # laterals tuned for performance gain - Sacrifice mouth modes
        self.model = HolisticAAM(
            self.LoadDataset(),
            group='PTS',
            verbose=True,
            holistic_features=float32_fast_dsift,
            diagonal=i_diag,
            scales=i_scale,
            max_appearance_components = i_max_greyscale_dims,
            max_shape_components = i_max_shape_dims)
        
        self.fitter = LucasKanadeAAMFitter(
            self.model,
            n_shape = [5, 15],
            n_appearance= [50, 150]);            

    '''
    Fit an appearance model to an image with annotated landmarks

    \return Converged candidate fit
    '''
    def FitAnnotatedImage(self, i_img):

        gt = i_img.landmarks['PTS'].lms
        initial_shape = self.fitter.perturb_from_bb( gt, gt.bounding_box() )

        return self.fitter.fit_from_shape(i_img, initial_shape, gt_shape=gt)

    '''
    Fit an appearance model to an image without annotations using Menpo Face Detection

    \return Converged landmarks
    '''
    def FitWildImageMenpo(self, i_img, i_initial_guess = None, i_max_iters = 10):

        # Convert menpo image to expected openCV format
        i_img = GreyscaleConversionMenpo(i_img)

        ret = None
        if i_initial_guess is not None:
            pts = menpo.shape.PointCloud(i_initial_guess, False)
            ret = self.fitter.fit_from_shape(i_img, pts, i_max_iters).final_shape.points
        else:
            bb = self.menpo_face_detector.Detect(i_img)
            if bb is not None:
                ret = self.fitter.fit_from_bb(i_img, bb, i_max_iters).final_shape.points

        return ret

    '''
    Fit an appearance model to an image without annotations using Viola Face Detection

    \return Converged landmarks
    '''
    def FitWildImageViola(self, i_img, i_initial_guess = None, i_max_iters = 10):

        # Convert menpo image to expected openCV format
        i_img = GreyscaleConversionMenpo(i_img)
            
        img = i_img.pixels[0] * 255
        img = numpy.array(img, dtype=numpy.uint8)

        # Detect face with experiment tuning according to lfpw testset
        ret = None 
        
        if i_initial_guess is None:
            faces = self.viola_face_detector.Detect(img, 3, 1.1, 0.125, 1.0)
        
            # Fit candidate model
            if len(faces) > 1:
                faces = [GetLargestROI(faces)]

                faces = ConvertRectToMenpoBoundingBox(faces)
                fit = self.fitter.fit_from_bb(i_img, faces[0], i_max_iters)
                ret = fit.final_shape.points
            
        elif i_initial_guess is not None:

            pts = menpo.shape.PointCloud(i_initial_guess, False)
            ret = self.fitter.fit_from_shape(i_img, pts, i_max_iters).final_shape.points

        return ret

    '''
    Print debug information for the AAM class
    '''
    def PrintDebug(self):

        print('Dataset', self.dataset)
        print self.model

    
        




    
