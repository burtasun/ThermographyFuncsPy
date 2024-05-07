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
    dutyRatio:float
) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
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
def HOS_Wrap(imgsList:np.ndarray, dutyRatio:int)->ProcDict:
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


#PCA convencional covarianza eje temporal
def PCA(
    imgs:np.ndarray,
    nComponents:int,
    ratioCovariance = 1.0
) -> list[np.ndarray]:
    """
    Principal component analysis

    Parameters
    ----------
    * imgs[nImgs,Height,Width]
    * nComponents, number of components, typical values for pulsed thermographies, 3-5
    * ratioVariance, alternatively deduce nComponents (nComponents <=0 )

    Return
    ------
    ret = list[np.ndarray] -> len(ret)=pca_nComponents
    """

    p,h,w = imgs.shape

    #media de columnas
    termMean = np.mean(imgs,axis=0)
    #termo centrado
    imgsNoMean = (imgs - np.repeat(termMean,p).reshape((h,w,p)).transpose((2,0,1))
                  ).transpose((1,2,0))#aplanamiento npx X t
    B = imgsNoMean.reshape(h*w,p)

    i, j = 10,20
    a = np.mean(imgsNoMean[i,j,:])
    b = np.mean(B[i*w+j,:])
    print(a, b)

    #covarianza
    C = B.transpose() @ B # p X p
    #svd
    try:
        U, S, VT = np.linalg.svd(C, False, True)
    except np.linalg.LinAlgError:
        print("Did not converge")
        exit(0)
    print(f'{U.shape}')
    print(f'{S.shape}')
    print(f'{VT.shape}')

    #Determinar numero de componentes
    l = 1; #nComponents
    ratioCovariance = max(0.0, min(ratioCovariance, 1.0))
    if nComponents > 0:
        l = min(nComponents, U.shape[1])
    elif ratioCovariance == 1.0:
        l = U.shape[1]
    else:
        #Ratio contribucion de cada componente
        #    Total de las componentes
        allComp = S.cumsum()
        for j in S.shape[0]:
            if (allComp[j] / allComp[-1]) >= ratioCovariance:
                l = j
                break
    # primeras l columnas
    W = U[:, :l];#p X l Mf W = U.topLeftCorner(U.rows(), l);//p X l
    #proyectar en base 
    T = B @ W;#p X l
    #Convertir a imagen
    T_3D = np.split(T.reshape((h,w,l)),l,axis=2)
    return T_3D
#PCA
def PCA_Wrap(imgsList:np.ndarray, nComponents:int)->ProcDict:
    print('ProcTermo, Procs.PCA')
    dictRet = ProcDict()
    for imgs in imgsList:
        retTuple = PCA(imgs, nComponents)
        for i,ret in enumerate(retTuple):
            if not (f'PCA_{i}' in dictRet):
                dictRet[f'PCA_{i}']=list()
            dictRet[f'PCA_{i}'].append(ret)
    return dictRet
#PCA_Wrap

    
#PCA convencional covarianza eje temporal
def deltaT(
    imgs:np.ndarray,
    dutyRatio = 0.5
) -> np.ndarray:
    """
    Delta T

    Parameters
    ----------
    * imgs[nImgs,Height,Width]
    * dutyRatio, pulsed thermography, dutyRatio

    Return
    ------
    ret = list[np.ndarray]
    """

    dutyRatio = min(1.0,max(0.0,dutyRatio))
    frameCool = 0
    frameMaxHeat = int(imgs.shape[0]*dutyRatio)
    return [imgs[frameCool,...]-imgs[frameMaxHeat,...]]
#deltaT
def deltaT_Wrap(imgsList:list[np.ndarray], dutyRatio:int)->ProcDict:
    print('ProcTermo, Procs.deltaT')
    dictRet = ProcDict()
    for imgs in imgsList:
        retTuple = deltaT(imgs, dutyRatio)
        for i,ret in enumerate(retTuple):
            if not (f'DeltaT_{i}' in dictRet):
                dictRet[f'DeltaT_{i}']=list()
            dictRet[f'DeltaT_{i}'].append(ret)
    return dictRet
#deltaT_Wrap


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
            outputDictRets.append(deltaT_Wrap(termos,Params.Input.dutyRatio))
        if ProcType == Procs.HOS:
            outputDictRets.append(HOS_Wrap(termos,Params.Input.dutyRatio))
        if ProcType == Procs.PCA:
            outputDictRets.append(PCA_Wrap(termos, 10))
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