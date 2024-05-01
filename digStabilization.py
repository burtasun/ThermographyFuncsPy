import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tifffile as tiff
import thermo_io
import thermo_procs
import thermo_phase_shift
import helpers_imgs
from scipy import ndimage

from globalVars import *
from typedefs import *

def getDimsOffsetTfsImgs(\
	transforms:list[np.ndarray],
	h:int, w:int,
) -> tuple[np.ndarray, np.ndarray]:
	#stitch directo
	#obtener dimensiones finales del frame completo roto-trasladando esquinas de frames
    corners = np.array(\
        [0, w, w, 0,\
		0, 0, h, h]).T.reshape(2,4)
    cornersTrans = np.zeros([2,4*len(transforms)])
	#roto-traslacion esquinas de cada frame
    for i in range(len(transforms)):
        rot = transforms[i][:2,:2]
        trans = transforms[i][:2,2:]
        cornersTrans[:,i*4:(i+1)*4] = rot @ corners + np.repeat(trans,4,axis=1)
        # print(f'cornersTrans[:,i*4:(i+1)*4]\n{cornersTrans[:,i*4:(i+1)*4]}')
    # print(f'cornersTrans\n{cornersTrans}')
    minMaxBB = np.percentile(cornersTrans,(0,100),axis=1)#mXmy;MXMY
    # print(f'minMaxBB\n{minMaxBB}')
    wBB, hBB = minMaxBB[1,:] - minMaxBB[0,:] #dimensiones del frame global
    upLeftCornerPts = cornersTrans[:,::4]
    offset = np.max(upLeftCornerPts, axis=1)#offset global respecto a origen (upleft)

    #Transformacion frames respecto a frame global
    transformsOut = list[np.ndarray]()
    for i in range(len(transforms)):
        t = np.identity(3,np.float32)
        t[:2,:3] = transforms[i][:2,:]#roto-traslacion
        d = np.identity(3,np.float32)
        d[:2, 2] = offset#traslacion
        t_global_local = d @ np.linalg.inv(t) #a grame global
        transformsOut.append(t_global_local)
        # print(f'{i} t_global_local\n{t_global_local}\n\n')
    return transformsOut, np.array([wBB,hBB]).astype(np.int32)

#helper delta2d -> Tf2d
def displacementToHomogMat(disp):
    ret = np.eye(3,3)
    ret[:2,2]=disp
    return ret

def stitchImgs(
    imgs:np.ndarray,
    deltas:np.ndarray | None
)->np.ndarray:
    if deltas is None:
        return imgs
    # stitching en unico frame superponiendo imagenes
    nImgs,h,w=imgs.shape
    nImgs = min(nImgs,deltas.shape[0])
    deltasTf = list[np.ndarray]()
    for i in range(nImgs):
        deltasTf.append(displacementToHomogMat(deltas[i,...].T))
        
    #global frame for stitching
    tfsGlob, whDims = getDimsOffsetTfsImgs(deltasTf,h,w)
    stitchedSeq = np.zeros((nImgs,whDims[1],whDims[0]),np.float32)
    fig, ax = plt.subplots()
    for i in range(nImgs):
        stitchedSeq[i,...] = cv.warpAffine(\
            imgs[i,...], tfsGlob[i][:2,:], (whDims[0],whDims[1]), None, borderMode=cv.BORDER_CONSTANT)
    return stitchedSeq
#stitchImgs

#Opencv feature match
class Matcher:
    def iniDescriptorAndMatcher(DescriptorType):
        #TODO parametrizar...
        if DescriptorType == 'AKAZE':
            kpDetectAndDescriptor = cv.AKAZE.create(\
                descriptor_type=cv.AKAZE_DESCRIPTOR_MLDB_UPRIGHT,\
                threshold=0.05,\
                max_points=-1)
            matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False) #descriptor binario / dist hamming 
            norm2Byte = False
        elif DescriptorType == 'SIFT':
            kpDetectAndDescriptor = cv.SIFT.create(nfeatures=0,contrastThreshold=0.04)
            matcher = cv.BFMatcher()
            norm2Byte = True
        else:
            raise(f'iniDescriptorAndMatcher {DescriptorType} no implementado')
        return kpDetectAndDescriptor, matcher, norm2Byte
    
            
    def __init__(self, DescriptorType:str) -> None:
        #detect kps and compute descriptor
        self.kpDetectAndDescriptor, self.matcher, self.norm2Byte = Matcher.iniDescriptorAndMatcher(DescriptorType)

    def setRefimg(self, img:np.ndarray):
        if self.norm2Byte:
            self.imgRef = Matcher.imgToByte(img)
        else:
            self.imgRef = img
        self.kpRef, self.desRef = self.kpDetectAndDescriptor.detectAndCompute(self.imgRef, None)

    def imgToByte(img):
        return (255.0*(img-np.percentile(img,0.01))/(np.percentile(img,99.99)-np.percentile(img,0.01))).astype(np.uint8)
            
    """
    distRatioMatch, ratio dist kNn1 Vs Knn2, proximidad relativa distrib
    estim 3d trans Ramsac, ransacThresh proj err
    """
    def matchImg(self, 
        img2Match:np.ndarray,\
        distRatioMatch = 0.75, minPtsMatch = 10,\
        ransacThresh = 3, confidence = 0.99,
        showMatches = False
    ):
        if self.norm2Byte:
            img2Match = Matcher.imgToByte(img2Match)
        kpsMatch, desMatch = self.kpDetectAndDescriptor.detectAndCompute(img2Match,None)
        #match descriptors
        matches = self.matcher.knnMatch(self.desRef,desMatch,k=2)
        #filter descriptors matched pairs by distance
        count = 0
        good,ptsRef,pts = [],[],[]
        for m,n in matches:      
            if m.distance < distRatioMatch*n.distance:
                good.append([m])
                ptsRef.append(self.kpRef[m.queryIdx].pt)
                pts.append(kpsMatch[m.trainIdx].pt)
                count += 1
        
        ptsRef = np.array(ptsRef)
        pts = np.array(pts)

        if count > minPtsMatch:
            #niap opencv no tiene estimacion traslacion 2D
            print(f'npts {ptsRef.shape[0]}')
            ptsRef_3d = np.zeros((ptsRef.shape[0],3),np.float32)
            ptsRef_3d[:,:2] = ptsRef
            pts_3d = np.zeros((pts.shape[0],3),np.float32)
            pts_3d[:,:2] = pts
            ret, trans3D, _ = cv.estimateTranslation3D(ptsRef_3d, pts_3d, ransacThreshold=ransacThresh, confidence=confidence)
            if ret == 0:
                print('match unsucsessfull')
                return None
        
        if showMatches:
            imgRefVis = Matcher.imgToByte(self.imgRef)
            img2MatchVis = Matcher.imgToByte(img2Match)
            imgMatches = cv.drawMatchesKnn(
                imgRefVis, self.kpRef,\
                img2MatchVis, kpsMatch, good, None, flags=2)
            plt.imshow(imgMatches,cmap='gray')
            plt.waitforbuttonpress()
        
        return trans3D.flatten()[:2].reshape((1,2))


class RegisterPars:
    subSampleReg = 1#1: no sub, 2: skips 1 frame / half, etc.
    maxFramesBackReg = 300# pair-wise registration of sequence-> (i,i-(i%maxFramesBackReg-1))

#rough approximation
def highPassFilter(img:np.ndarray, kSize = 11)->np.ndarray:
    if kSize<3:
        return img
    #low-pass substraction as mean filter
    imgLow = cv.blur(img,(kSize,kSize))
    return img-imgLow

'''
TODO impl subsample reg
TODO params ext
TODO post-proc traj
TODO others... bi-directional reg, etc. / sparse BA??
'''
def Register(
    imgs:list[np.ndarray],
    viewMatches = False,
    plotTrajectory = False,
    registerPars = RegisterPars(),
    smoothTrajectorySigma = 2
) -> np.ndarray:
    
    if registerPars is None:
        registerPars=RegisterPars()
        
    nimgs,h,w=imgs.shape
    deltas = np.zeros((nimgs,2))#incl 1st (null displacement)
    
    #SIFT or AKAZE
    # SIFT implemented only on 8bits!
    # AKAZE robust / relatively fast
    matcher = Matcher('SIFT')
    #save recurrent kp / descrip compute
    #   "highpass" mitigates information loss on 8 bit normalization
    matcher.setRefimg(highPassFilter(imgs[0,...]))

    iref = 0#idx ref img
    countFailedMatches = 0
    #TODO paralelize
    for i in range(1,imgs.shape[0]):
        if ((i%registerPars.maxFramesBackReg)==0):
            iref = i-1
            matcher.setRefimg(highPassFilter(imgs[iref,...]))#sucesivamente hereda errores locales, diverge mucho, en su lugar maximizamos distancia / solape 50% suficiente, OK
        trans_i_iref = matcher.matchImg(highPassFilter(imgs[i,...]), 0.5, 10, 1, showMatches=viewMatches)
        if trans_i_iref is None:
            print(f'image {i} regarding {iref}, does not have a match, using previous translation')
            deltas[i,...] = deltas[i-1,...]
            countFailedMatches+=1
        else:
            deltas[i,...] = trans_i_iref + deltas[iref,...] #regarding frame 0
    deltas = SmoothTrajectory(deltas, smoothTrajectorySigma, plotTrajectory)
    return deltas
#Register


def SmoothTrajectory(deltas:np.ndarray, sigma, plotTrajectory=True):
    if sigma<1:
        deltasFiltered = deltas
    else:
        deltasFiltered = ndimage.gaussian_filter1d(deltas, sigma=sigma, axis=0, mode='nearest')
    if plotTrajectory:
        plt.plot(deltas[:,0],deltas[:,1],label='Reg Trajectory')        
        plt.plot(deltasFiltered[:,0],deltasFiltered[:,1],label='Reg Trajectory Filt')
        plt.legend()
        plt.show()
    return deltasFiltered

if __name__=='__main__':
    accDeltas = np.stack(\
        (np.linspace(2,20-1,18),
         np.linspace(0,2-1,18))).astype(np.float32)
    SmoothTrajectory(accDeltas, 4)


#Temp filter 
#   promedia componente alta freq con frames adyacentes roto-trasladados
def TempFilter(\
    imgs:np.ndarray,
    deltas:list[np.ndarray] | None,
    averageRad = 2,
    kMeanBlur = 11
) -> np.ndarray:

    if averageRad <=0:
        return imgs
    
    nImgs,h,w=imgs.shape
    b = averageRad
    k = kMeanBlur
    imgsOut = np.zeros(imgs.shape,np.float32)

    if deltas is None:
        deltas=np.split(np.zeros((imgs.shape[0],2),np.float32),imgs.shape[0],axis=0) #TODO niapa
   
    for i in range(imgs.shape[0]):
        imgsHp_i = list[np.ndarray]()
        imgsHp_i.append(highPassFilter(imgs[i,...],k))
        imgsLp_i = cv.blur(imgs[i,...],(k, k))
        for j in range(max(0,i-b),min(i+b+1,len(deltas))):
            if j==i:
                continue
            tf_i_j = displacementToHomogMat(deltas[i]-deltas[j])
            imgsHp_j = highPassFilter(imgs[j,...])
            imgsHp_i_j = cv.warpAffine(imgsHp_j , tf_i_j[:2,...], (w,h), None, 
                borderValue=0, borderMode=cv.BORDER_CONSTANT,
                flags=cv.INTER_LINEAR)
            imgsHp_i.append(imgsHp_i_j)
        imgsHp_i = np.array(imgsHp_i)
        #Median excludes outlier/non conforming pixels from mean / and avoids overlap window masking
        imgsHp_i = np.median(imgsHp_i,axis=0)
        imgsTempFilt = imgsHp_i + imgsLp_i
        imgsOut[i,...]=imgsTempFilt
    #loop
    return imgsOut