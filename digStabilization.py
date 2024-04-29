import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2# as cv
import tifffile as tiff
import thermo_io
import thermo_procs
import thermo_phase_shift
import helpers_imgs
from globalVars import *
from typedefs import *

def getDimsOffsetTfsImgs(\
	transforms:list[np.ndarray],
	h:int, w:int,
) -> [np.ndarray, np.ndarray]:
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
    deltas:np.ndarray
)->np.ndarray:
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
    def __init__(self, DescriptorType:str) -> None:
        #detect kps and compute descriptor
        self.kpDetectAndDescriptor = cv2.AKAZE.create(\
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT,\
            threshold=0.05,\
            max_points=-1)
        #matcher, akaze, binary descriptor, NormHamming
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    def setRefimg(self, img:np.ndarray):
        self.imgRef = img
        self.kpRef, self.desRef = self.kpDetectAndDescriptor.detectAndCompute(self.imgRef, None)



    def matchImg(self, 
        img2Match:np.ndarray,\
        distRatioMatch = 0.75, minPtsMatch = 4,\
        ransacThresh = 3, confidence = 0.99,
        showMatches = False
    ):
        
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
            ret, trans3D, self.inliners = cv2.estimateTranslation3D(ptsRef_3d, pts_3d, ransacThreshold=3, confidence=0.99)
            if ret == 0:
                print('match unsucsessfull')
                return None
        
        if showMatches:
            def imgToByte(img):
                return (255 * (img-np.min(img))/(np.max(img)-np.min(img))).astype(np.uint8)
            imgRefVis = imgToByte(self.imgRef)
            img2MatchVis = imgToByte(img2Match)
            imgMatches = cv2.drawMatchesKnn(
                imgRefVis, self.kpRef,\
                img2MatchVis, kpsMatch, good, None, flags=2)
            plt.imshow(imgMatches,cmap='gray')
            plt.waitforbuttonpress()
        
        return trans3D.flatten()[:2].reshape((1,2))
    






class RegisterPars:
    subSampleReg = 1#1: no sub, 2: skips 1 frame / half, etc.
    maxFramesBackReg = 100# pair-wise registration of sequence-> (i,i-(i%maxFramesBackReg-1))

    
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
    registerPars = RegisterPars()
) -> np.ndarray:
    
    if registerPars is None:
        registerPars=RegisterPars()
        
    nimgs,h,w=imgs.shape
    deltas = np.zeros((nimgs,2))#incl 1st (null displacement)
    
    #robust / relatively fast
    matcher = Matcher('AKAZE')
    #save recurrent kp / descrip compute
    matcher.setRefimg(imgs[0,...])

    iref = 0#idx ref img
    countFailedMatches = 0
    #TODO paralelize
    for i in range(1,imgs.shape[0]):
        if ((i%registerPars.maxFramesBackReg)==0):
            iref = i-1
            matcher.setRefimg(imgs[iref,...])#sucesivamente hereda errores locales, diverge mucho
        trans_i_iref = matcher.matchImg(imgs[i,...], showMatches=viewMatches)
        if trans_i_iref is None:
            print(f'image {i} regarding {iref}, does not have a match, using previous translation')
            deltas[i,...] = deltas[i-1,...]
            countFailedMatches+=1
        else:
            deltas[i,...] = trans_i_iref + deltas[iref,...] #regarding frame 0
    if plotTrajectory:
        plt.close()
        plt.scatter(deltas[:,0],deltas[:,1])
        plt.plot(deltas[:,0],deltas[:,1],label="Registration Trajectory")
        plt.show()
        plt.close()
    return deltas
#Register