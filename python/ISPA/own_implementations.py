

import cv2


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
