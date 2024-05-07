import sys
import os
import numpy as np
import matplotlib as plt
import cv2 as cv
from numpy import fft

from globalVars import *
import helpers_imgs

from typedefs import ProcDict

#harmonicNum: 0 todos, o uno cualquiera
def FFT(termo:np.ndarray, harmonicNum=0) -> [np.ndarray,np.ndarray]:
    fftTermo = fft.fft(termo, axis=0)
    dims = fftTermo.shape[0]
    if harmonicNum!=0:
        fftTermo=fftTermo[harmonicNum]
        dims = 1
    mag = np.sqrt(np.power(fftTermo.real,2) + np.power(fftTermo.imag,2)).astype(np.float32)
    pha = np.arctan2(fftTermo.real, fftTermo.imag).astype(np.float32)
    pha = np.unwrap(np.unwrap(pha,axis=0),axis=1) #sensible a outliers en bordes! Unwrap-basura con paddings!
    return mag, pha

def FFT_Wrap(imgsList:np.ndarray, acquisitionPeriods:int)->tuple[np.ndarray,np.ndarray]:
    print('ProcTermo, Procs.FFT')
    phases, magnitudes = list[np.ndarray](), list[np.ndarray]()
    for imgs in imgsList:
        mag, pha = FFT(imgs, acquisitionPeriods)
        phases.append(pha)
        magnitudes.append(mag)
    outputDict = ProcDict()
    outputDict['FFT_Phases'] = phases
    outputDict['FFT_Mags'] = magnitudes
    return outputDict
#FFT_Wrap
    
    

#Higher order statistics 
#   Madruga 2010
def HOS(
    imgsIn:np.ndarray,
    dutyRatio:np.ndarray
) -> tuple(np.ndarray,np.ndarray,np.ndarray,np.ndarray):
    """
    Higher order statistics of temporal axis

    Parameters
    ----------
    * imgs[nImgs,Height,Width]
    * dutyRatio 0, no heating/all cooling, 0.5 common in pulsed/locking

    Return
    ------
    termMean, stdDev, kurtosis, skewness

    """
    #extraccion de momentos estadisticos del termograma
    #    media, desviacion estandar, 3 curtosis, 4 skewness

    #Only in cooling
    coolingFrameNum = int(float(imgsIn.shape[0])*dutyRatio)
    if coolingFrameNum > 0:
        if coolingFrameNum >= imgsIn.shape[0]:
            print(f'HOS_all coolingFrameNum >= imgs.shape[0]! \n\t {coolingFrameNum} >= {imgs.shape[0]}')
            print(f'HOS_all assuming middle frame!')
            coolingFrameNum = imgs.shape[0] // 2
        imgs = imgsIn[coolingFrameNum:,...]
    else:
        imgs = imgsIn

    n,h,w = imgs.shape

    termMean = np.mean(imgs,axis=0)

    #x-mu
    imgsNoMean = imgs - np.repeat(termMean,n).reshape((h,w,n)).transpose((2,0,1))#repeat on last dim, permute temporal axis to 1st dim as usual
    
    #(((sigma (x-mu)^2)/n)^.5)
    stdDev =\
        np.sqrt(
            np.mean(
                np.power(imgsNoMean,2), axis=0))

    #Higher orders // kurtorsis (j=3) skewness (j=4)
    #(((sigma (x-mu)^j)/n)/stdev^j)
    for j in range(3,5):
        hos_j=\
            np.mean(np.power(imgsNoMean,float(j)), axis=0) /  \
            np.power(stdDev,float(j))
        if j == 3: kurtosis = hos_j
        elif j == 4: skewness = hos_j

    return termMean, stdDev, kurtosis, skewness
#HOS
def HOS_Wrap(imgsList:np.ndarray, dutyRatio:int)->tuple[np.ndarray,np.ndarray]:
    print('ProcTermo, Procs.HOS')
    orderedNames = ["HOS_mean","HOS_stdDev","HOS_kurtosis","HOS_skewness"]
    dictRet = ProcDict()
    for name in orderedNames: dictRet[name]=list()
    for imgs in imgsList:
        retTuple = HOS(imgs, dutyRatio)
        for ret,key in zip(retTuple,orderedNames):
            dictRet[key].append(ret)
    return dictRet
#HOS_Wrap
    


'''ProcTermo / WRAP CALL FUNCS
TODO implement procs
TODO visualization / debug / log
TODO abstract params algs
'''
def ProcTermo(
        termos:list[np.ndarray]#nImgs,Rows,Cols = dataInputs[0].shape
) ->ProcDict:
    
    outputDictRets = list[ProcDict]()
    for ProcType in Params.Tasks.procTermo:
        if ProcType == Procs.FFT: #keys FFT_Phase, FFT_Mag
            outputDictRets.append(FFT_Wrap(termos, Params.Input.acquisitionPeriods))
        if ProcType == Procs.DeltaT:
            print('ProcTermo, Procs.DeltaT not implemented!')
        if ProcType == Procs.HOS:
            outputDictRets.append(HOS_Wrap(termos,Params.Input.dutyRatio))
        if ProcType == Procs.PCA:
            print('ProcTermo, Procs.PCA not implemented!')
        if ProcType == Procs.PCT:
            print('ProcTermo, Procs.PCT not implemented!')
        if ProcType == Procs.SenoidalFit:
            print('ProcTermo, Procs.SenoidalFit not implemented!')
        if ProcType == Procs.TSR:
            print('ProcTermo, Procs.TSR not implemented!')
            
    outputDict = ProcDict()#ej. {'FFT_Phase': list[np.ndarray], 'FFT_Mag': list[np.ndarray]}
    #flatten dict lists
    for dictRet in outputDictRets:
        for key in dictRet:
            outputDict[key]=dictRet[key]
    if Params.LogData.saveData:
        helpers_imgs.saveImgDict(outputDict,f'{Params.outputDir}','proc_')
            
    return outputDict
#ProcTermo