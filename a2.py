#a2.py
import numpy as np
import math
import imageIO as io


#this file should only contain function definitions.
#It should not call the functions or perform any test.
#Do this in a separate file.

############## HELPER FUNCTIONS ###################
def imIter(im):
 for y in xrange(im.shape[0]):
    for x in xrange(im.shape[1]):
       yield y, x

def getBlackPadded(im, y, x):
 if (x<0) or (x>=im.shape[1]) or (y<0) or (y>= im.shape[0]): 
    return np.array([0, 0, 0])
 else:
    return im[y, x]

def clipX(im, x):
   return min(np.shape(im)[1]-1, max(x, 0))

def clipY(im, y):
   return min(np.shape(im)[0]-1, max(y, 0))
              
def getSafePix(im, y, x):
 return im[clipY(im, y), clipX(im, x)]

def point(x, y):
   return np.array([x, y])
              
################# END HELPER ######################

def check_my_module():
   ''' Fill your signature here. When upload your code, check if the signature is correct'''
   my_signature='Ryan Lacey'
   return my_signature 


def pix(im, y, x, repeatEdge=False):
    '''takes an image, y and x coordinates, and a bool
        returns a pixel.
        If y,x is outside the image and repeatEdge==True , you should return the nearest pixel in the image
        If y,x is outside the image and repeatEdge==False , you should return a black pixel
    '''
    if repeatEdge:
        return getSafePix(im, y, x)
    else:
        return getBlackPadded(im, y, x)


def scaleNN(im, k):
    '''Takes an image and a scale factor. Returns an image scaled using nearest neighbor interpolation.
    '''
    out = io.constantIm(im.shape[0]*k, im.shape[1]*k, 0.0)
    for y, x in imIter(out):
        out[y,x]=im[clipY(im, round(y/k)), clipX(im, round(x/k))]
    return out
    

def interpolateLin(im, y, x, repeatEdge=False):
    '''takes an image, y and x coordinates, and a bool
        returns the interpolated pixel value using bilinear interpolation
    '''
    # Get nearby neighbor coordinates
    leftX = int(math.floor(x))    
    rightX = int(math.ceil(x))
    bottomY = int(math.floor(y))
    topY = int(math.ceil(y))
    # Interpolate x
    if leftX == rightX:
        topDiff = pix(im, topY, x, repeatEdge)
        botDiff = pix(im, bottomY, x, repeatEdge)
    else:
        topDiff = pix(im, topY, rightX, repeatEdge) * (x - leftX) + pix(im, topY, leftX, repeatEdge) * (rightX - x)
        botDiff = pix(im, bottomY, rightX, repeatEdge) * (x - leftX) + pix(im, bottomY, leftX, repeatEdge) * (rightX - x)
    # interpolate temp x's
    if bottomY == topY:
        yDiff = topDiff
    else:
        yDiff = botDiff * (topY - y) + topDiff * (y - bottomY)
    return yDiff



def scaleLin(im, k):
    '''Takes an image and a scale factor. Returns an image scaled using bilinear interpolation.
    '''
    (height, width, depth) = np.shape(im)
    img = io.constantIm(im.shape[0]*k, im.shape[1]*k, 0.0)
    for y, x in imIter(img):
        # print img[y][x], interpolateLin(im, y, x, False), '\n'
        img[y][x] = interpolateLin(im, y/k, x/k, False)
    return img


def rotate(im, theta):
    '''takes an image and an angle in radians as input
        returns an image of the same size and rotated by theta
    '''
    img = im.copy()
    centerX = np.shape(im)[1] / 2
    centerY = np.shape(im)[0] / 2 
    for y, x in imIter(im):
        xp = centerX + (x - centerX) * math.cos(theta) - (y - centerY) * math.sin(theta)
        yp = centerY + (x - centerX) * math.sin(theta) + (y - centerY) * math.cos(theta)
        img[y][x] = interpolateLin(im, yp, xp, False) #im[clipY(img, round(yp))][clipX(img, round(xp))]
    return img


class segment:
    def __init__(self, x1, y1, x2, y2):
        #notice that the ui gives you x,y and we are storing as y,x
        self.P=np.array([y1, x1], dtype=np.float64)
        self.Q=np.array([y2, x2], dtype=np.float64)
        self.PQ = self.Q - self.P
        self.perpPQ = np.array([-self.PQ[1], self.PQ[0]])
    

    def uv(self, X):
        '''Take the (y,x) coord given by X and return u, v values
        '''
        PX = X - self.P
        u = np.dot(PX, self.PQ) / np.dot(self.PQ, self.PQ)
        v = np.dot(PX, self.perpPQ) / np.dot(self.PQ, self.PQ) ** 0.5
        return u, v

    def dist (self, X):
        '''returns distance from point X to the segment (pill shape dist)
        '''
        # return self.uv(X)[1]
        return min(abs(self.uv(X)[1]), np.dot(X - self.P, X - self.P), np.dot(X - self.Q, X - self.Q))
    
    def uvtox(self,u,v):
        '''take the u,v values and return the corresponding point (that is, the np.array([y, x]))
        '''
        return self.P + u * self.PQ + (v * self.perpPQ) / np.dot(self.PQ, self.PQ) ** 0.5



def warpBy1(im, segmentBefore, segmentAfter):
    '''Takes an image, one before segment, and one after segment. 
        Returns an image that has been warped according to the two segments.
    '''
    out = im.copy()
    for y, x in imIter(im):
        u, v = segmentAfter.uv(np.array([y,x]))
        yp, xp = segmentBefore.uvtox(u, v)
        out[y][x] = interpolateLin(im, yp, xp, True)
    return out


def weight(s, X,  a=10, b=1, p=1):
    '''Returns the weight of segment s on point X
    '''
    length = np.dot(s.PQ, s.PQ) ** 0.5
    return (length ** p / (a + s.dist(X))) ** b

def warp(im, segmentsBefore, segmentsAfter, a=10, b=1, p=1):
    '''Takes an image, a list of before segments, a list of after segments, and the parameters a,b,p (see Beier)
    '''
    out = im.copy()
    for y, x in imIter(out):
        DSUM = (0,0)
        weightsum = 0
        for i, segment in enumerate(segmentsAfter):
            u, v = segment.uv(np.array([y,x]))
            yp, xp = segmentsBefore[i].uvtox(u,v)
            displacement = np.array([yp-y, xp-x])
            w = weight(segment, np.array([y,x]))
            DSUM  += displacement * w
            weightsum += w
        out[y,x] =  interpolateLin(im, y+DSUM[0]/weightsum, x+DSUM[1]/weightsum, True) 
    return out


def morph(im1, im2, segmentsBefore, segmentsAfter, N=1, a=10, b=1, p=1):
    '''Takes two images, a list of before segments, a list of after segments, the number of morph images to create, and parameters a,b,p.
        Returns a list of images morphing between im1 and im2.
    '''
    sequence=list()
    sequence.append(im1.copy())
    seqFromStart = []
    seqFromEnd = []
    segmentDifferences = []
    stepSize = 1.0 / (N+1)
    for i in range(len(segmentsBefore)):
        (diffStartX, diffStartY) = segmentsAfter[i].P  - segmentsBefore[i].P
        (diffEndX, DiffEndY) = segmentsAfter[i].Q - segmentsBefore[i].Q
        segmentDifferences.append(segment(diffStartX, diffStartY, diffEndX, DiffEndY))
    for i in xrange(N):
        fromStart = []
        fromEnd = []
        for diff in segmentDifferences:
            print '\n\n================================================', segmentsBefore[i], diff.P, type(diff), type(diff.P), stepSize, i+1
            fromStart.append(segmentsBefore[i].P[1]+diff.P*stepSize*(i+1), segmentsBefore[i].P[0]+ diff.P*stepSize*(i+1), segmentsBefore[i].Q+ diff.Q*stepSize*(i+1), segmentsBefore+ diff.Q*stepSize*(i+1))
            fromEnd.append(segmentsAfter[i].P[1]-diff.P*stepSize*(i+1), segmentsAfter[i].P[0]+ diff.P*stepSize*(i+1), segmentsAfter[i].Q+ diff.Q*stepSize*(i+1), segmentsAfter+ diff.Q*stepSize*(i+1))
            # fromStart.append(segmentsBefore[i] + diff.P*stepSize*(i+1))
            # fromEnd.append(segmentAfter[i] - diff.Q*stepSize*i)
        seqFromStart.append(morph(im1.copy(), segmentsBefore, fromStart))
        seqFromEnd.append(morph(im2.copy(), segmentAfter, fromEnd))
    sequence.append(im2.copy())
    return sequence, seqFromStart, seqFromEnd
