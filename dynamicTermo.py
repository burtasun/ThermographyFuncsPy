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

from typedefs import *
from globalVars import *
from digStabilization import *

"""
    agrupa segun direccion, devuelve list de lista de indices
"""
def ExtractSyncPulses (
    nImgs,
    nFramesPerPulse:int, #//espaciado entre frames de inicio de pulsado
	nDirections:int,#n switch dirs
	nPulsesPerDirection:int,#normalmente 1, previas multiples
	nPulsesPreserve:int,#normalmente el maximo posible
	nPulsesSkip:int,#salta n pulsos iniciales
	offsetStart:int #extension a preservar / corregir desviaciones sync
)->list[int]:
    print("ExtractSyncPulses")

    idxGoups=list[list[int]]()
    for i in range(nDirections): 
        idxGoups.append(list[int]())
    
    nFramesPerDirection = nPulsesPerDirection * nFramesPerPulse
    nFramesSkip = nPulsesSkip * nFramesPerPulse
    #Trocear secuencia en fragmentos correspondientes a pulsos equivalentes
    dir = 0
    while True:
        mOff = int(min(nFramesSkip + dir * nFramesPerDirection + offsetStart, nImgs))
        MOff = int(min(max((dir + 1) * nFramesPerDirection + offsetStart - nFramesSkip, mOff), nImgs))
        if mOff >= MOff: #fin
            break
        # std::cout << "mOff " << mOff << "   MOff " << MOff << "\n";
        d = dir % nDirections #rota
        #insert idx
        idxGoups[int(d)].extend(
            np.linspace(mOff,MOff-1,MOff-mOff).astype(np.int32).tolist()
        )        
        dir+=1
    
    for i,idx in enumerate(idxGoups):
        print(f'{i} {len(idx)}')

    return idxGoups
#ExtractSyncPulses

#alineamiento y sincronizacion superponiendo mascaras roto-trasladadas
def temporalAlignment (
    imgsIdx:list[int],
	imgs:np.ndarray, #i/o sobreescribe / frames locales sin roto-trasladar
	tfs:list[np.ndarray],#roto-traslacion del frame global
	whSzStitched,
	nFramesPerPulse:int, #espaciado entre frames de inicio de pulsado
	nFramesFramesWindow:int,
    offsetStart:int#extension a preservar
) -> np.ndarray:
    nImgs, h, w = imgs.shape
    out = np.zeros((nFramesFramesWindow,whSzStitched[1],whSzStitched[0]),np.float32)

    imgRt = np.zeros((whSzStitched[1],whSzStitched[0]), np.float32)
    imgPrev = np.zeros((whSzStitched[1],whSzStitched[0]), np.float32)
    
    def tfImg(tf:np.ndarray, imgin:np.ndarray, imgOut:np.ndarray):
        imgOut = cv.warpAffine(imgin, tf[:2,:], (imgOut.shape[1],imgOut.shape[0]), imgOut, cv.INTER_LINEAR, cv.BORDER_TRANSPARENT)

    i = offsetStart
    iter = 0
    maskCurr_b = np.zeros(imgRt.shape,dtype=bool)
    maskPrev_b = np.zeros(imgRt.shape,dtype=bool)
    
    while True:
        imgRt*=0
        idActGroup = imgsIdx[i]
        tfImg(tfs[idActGroup], imgs[idActGroup], imgRt)
        maskCurr_b = (imgRt>0).astype(np.uint8)
        if i != offsetStart:
            maskAux = maskCurr_b - maskPrev_b
        else:
            maskAux = np.copy(maskCurr_b)

        maxId = min(nFramesFramesWindow, len(imgsIdx))
        for idAct in range(i, maxId):
            idActGroup = imgsIdx[idAct]
            tfImg(tfs[idActGroup], imgs[idActGroup], imgRt)
            cv.copyTo(imgRt, maskAux, out[idAct-i,...])#copia con mascara
        plt.imshow(out[idAct-i,...])
        plt.pause(0.1)
        
        #swap manual...
        temp = maskCurr_b
        maskCurr_b = maskPrev_b
        maskPrev_b = temp

        i += nFramesPerPulse
        if i>=len(imgsIdx):
            break
    # //imgs.resize(nFramesFramesWindow)
    # out.swap(imgs);
    return out
#temporalAlignment


#extract oscillating component
def NormLockin (
    imgs: np.ndarray,
    npulses:int,
    nFramesPerPulse:int
) -> np.ndarray:
    
    out = np.zeros(imgs.shape,np.float32)

    print(f'NormLockin imgs.shape {imgs.shape}')
    print(f'NormLockin npulses {npulses}')
    print(f'NormLockin nFramesPerPulse {nFramesPerPulse}')

    #each pulse with average and linear component
    for i in range(npulses):
        #delta pulse
        delta_i =\
            imgs[(i+1) * nFramesPerPulse-1] - \
            imgs[( i ) * nFramesPerPulse]
        #average pulse
        aver_i = np.mean(imgs[i*nFramesPerPulse:(i+1)*nFramesPerPulse,...],axis=0)

        #normalizacion / oscillating = raw - average - linear
        for j in range(nFramesPerPulse):
            linear_ij = delta_i * (float(j) - float(nFramesPerPulse) / 2.0 - 0.5) / float(nFramesPerPulse)
            out[i * nFramesPerPulse + j] =\
                imgs[i * nFramesPerPulse + j] -\
                aver_i -\
                linear_ij
    return out

class DynamicTermoPars:
    nPulsesPerDirection = 1
    nPulsesPreserve = 1
    nPulseSkip=0
    nDirections = 4



def dynamicTermo(
    imgs:np.ndarray,
    deltasLocal:np.ndarray) -> list[np.ndarray]:
    
    nFramesPerPulse = int(float(Params.Input.frameRate) / float(Params.Input.excFreq))

    print("dynamicTermo")
    nImgs,h,w = imgs.shape
    nImgs = min(nImgs,deltasLocal.shape[0])

    #separate frames based on pulse direction
    idxGroups = ExtractSyncPulses(
        nImgs, nFramesPerPulse,
        DynamicTermoPars.nDirections,
        DynamicTermoPars.nPulsesPerDirection, 
        DynamicTermoPars.nPulsesPreserve,
        DynamicTermoPars.nPulseSkip, 0)
    
    #get bounding box stitched sequence and transform to global frame
    deltasTf = list[np.ndarray]()
    for i in range(nImgs):
        deltasTf.append(displacementToHomogMat(deltasLocal[i,...].T))
    deltasTf, whDims = getDimsOffsetTfsImgs(deltasTf, h, w)

    #align subset frames and sync
    nFramesPreserve = DynamicTermoPars.nPulsesPreserve * nFramesPerPulse
    imgsSynced = list[np.ndarray]()
    for i,group in enumerate(idxGroups):
        imSync = temporalAlignment(group, imgs, deltasTf, whDims, nFramesPerPulse, nFramesPreserve, 0)
        imgsSynced.append(imSync)
        cv.imwritemulti(f'{Params.outputDir}\\synced{i}.tiff',imSync)
    
    #Locking normalization / oscillating component
    imgsNorm = list[np.ndarray]()
    for i,imgs in enumerate(imgsSynced):
        imNorm = NormLockin(imgs, DynamicTermoPars.nPulsesPreserve, nFramesPerPulse)
        imgsNorm.append(imNorm)
        cv.imwritemulti(f'{Params.outputDir}\\norm{i}.tiff',imNorm)
    return imgsNorm

#dynamicTermo