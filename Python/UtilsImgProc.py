
import cv2
import numpy

from menpo.shape.pointcloud import bounding_box

'''
Display menpo image with optional keypoints

\param[in] i_img   3 or 1 channel image
\param[in] i_pts   Column of 2D keypoints
'''
def DisplayMenpoImage(i_img, i_wait = 0, i_pts = None):

    i_img = GreyscaleConversionOpenCV(i_img)

    img = i_img.pixels[0].copy()

    # If no keypoints provided, only display image
    if i_pts is not None:    
        for pt in i_pts:
            pt = pt.astype(int)
            cv2.circle(img, (pt[1], pt[0]), 2, 255, 0) 

    cv2.namedWindow('image')
    cv2.imshow('image',img)
    cv2.waitKey(i_wait)

'''
Display openCV image with optional keypoints

\param[in] i_img   3 or 1 channel image
\param[in] i_pts   Column of 2D keypoints
'''
def DisplayOpenCVImage(i_img, i_wait = 0, i_pts = None):

    # If no keypoints provided, only display image
    if i_pts is not None:    
        for pt in i_pts:
            pt = pt.astype(int)
            cv2.circle(i_img, (pt[1], pt[0]), 2, (255, 255, 255), 0) 

    cv2.namedWindow('image')
    cv2.imshow('image',i_img)
    cv2.waitKey(i_wait)

'''
Display image with roi overlay

\param[in] i_img     Input image
\param[in] i_faces   Array of rect rois [ [roi1] [roi2] ... [roi<N>]
'''
def DisplayRects(i_img, i_rois, i_wait = 0):

    img = i_img.copy()

    for roi in i_rois:

        top_left = (roi[0], roi[1])
        bottom_right = (roi[0] + roi[2], roi[1] + roi[3])
        cv2.rectangle(img, top_left, bottom_right, (255, 0,0), 4)

    cv2.namedWindow("ROIs")
    cv2.imshow("ROIs", img)
    cv2.waitKey(i_wait)


'''
Convert openCV Rect to Menpo bounding box

\param[in] i_rois    Array of rects

\return Array of menpo bounding boxes with same ordering as input array
'''
def ConvertRectToMenpoBoundingBox(i_rois):

    bbs = []
    for roi in i_rois:
        top_left = (roi[0], roi[1])
        bottom_right = (roi[0] + roi[2], roi[1] + roi[3])
        bbs.append( bounding_box( top_left, bottom_right ) )

    return bbs

'''
Return largest ROI in list of rois

\param[in] i_rois list of rois
\return           largest roi in list
'''
def GetLargestROI(i_rois):

    area = i_rois[:,2] * i_rois[:,3]
    return i_rois[area.argmax(axis=0)]

'''
Greyscale conversion for menpo image

\param[in] i_img Menpo image
\return Greyscale image
'''
def GreyscaleConversionMenpo(i_img):

    if i_img.n_channels == 3:
        i_img = i_img.as_greyscale(mode='luminosity')

    return i_img

'''
Greyscale conversion for openCV mat

\param[in] i_img OpenCV image
\return Greyscale image
'''
def GreyscaleConversionOpenCV(i_img):

    if len(i_img.shape) == 3:
        i_img = cv2.cvtColor(i_img, cv2.COLOR_BGR2GRAY)

    return i_img

