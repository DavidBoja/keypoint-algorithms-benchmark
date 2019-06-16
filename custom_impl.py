

import cv2
import numpy as np
import skimage.feature
from other_algorithms.asift import affine_detect
from other_algorithms.susan import susanCorner

# DETECTORS
class HarrisMataHarris():
    '''
    Wrapper for the OpenCV implementation of the Harris detector.
    '''
    def __init__(self, blockSize=2, ksize=3, k=0.04, borderType=cv2.BORDER_DEFAULT):
#         self.src = src
        self.blockSize = blockSize
        self.ksize = ksize
        self.k = k
        self.borderType = borderType

    def detect(self, img, mask=None):
        '''
        Keypoint detector method. The threshold is set to 0.01.
        The parameter mask is set to keep standardized inputs for the detect method in the openCv implementations.
        '''
        keypoints = []

        # already taking care of this in the implementation
        # if len(img.shape) == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        dst = cv2.cornerHarris(img,self.blockSize,self.ksize,self.k)
        indexes = np.where(dst>0.01*dst.max())

        for i in range(len(indexes[0])):
              keypoints.append(cv2.KeyPoint(x=int(indexes[1][i]), y=int(indexes[0][i]), _size=0))

        return keypoints


class ShiTomasi():
    '''
    other parameters are left to the default OpenCv parameters
    '''
    def __init__(self, maxCorners=1000000, qualityLevel=0.01,minDistance=10):
        self.maxCorners = maxCorners
        self.qualityLevel = qualityLevel
        self.minDistance = minDistance

    def detect(self, img, mask=None):
        keypoints = []

        # Already taken care of in the implementation
        # if len(img.shape) == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(img,
                                          self.maxCorners,
                                          self.qualityLevel,
                                          self.minDistance)
        corners = np.int0(corners)
        corners = corners[:,0,:]

        for i in range(corners.shape[0]):
              keypoints.append(cv2.KeyPoint(x=int(corners[i,0]),
                                            y=int(corners[i,1]),
                                            _size=0))

        return keypoints


class CensureClass():

    def detect(self, img, mask=None):
        keypoints = []

        censure = skimage.feature.CENSURE()
        censure.detect(img)

        for i in range(censure.keypoints.shape[0]):
              keypoints.append(cv2.KeyPoint(x=censure.keypoints[i,1],
                                            y=censure.keypoints[i,0],
                                            _size=0))

        return keypoints


class SusanClass():

    def detect(self, img, mask=None):
        keypoints = []

        kp = susanCorner(img)

        kp = np.where(kp>0)

        for i in range(len(kp[0])):
            keypoints.append(cv2.KeyPoint(x=kp[0][i],
                                          y=kp[1][i],
                                          _size=0))

        return keypoints


# DESCRIPTORS
class RootSIFT:
    def __init__(self, eps=1e-7):
        self.eps = eps


    def compute(self, image, kp):
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.compute(image, kp)

        if len(kp) == 0:
            return kp, np.array([])

        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        des /= (des.sum(axis=1, keepdims=True) + self.eps)
        des = np.sqrt(des)
        #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)

        return kp, des



# DETECTOR + DESCRIPTOR
class ASIFTClass():

    def detectAndCompute(self, img, mask=None):
        kp, des = affine_detect(cv2.xfeatures2d.SIFT_create(), img)

        return kp, des
