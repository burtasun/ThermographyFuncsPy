import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math

from digStabilization import *
from typedefs import *

def ConvertToMaxContrastUchar(imgRAW:np.ndarray, minMaxV = None) -> np.ndarray:
    if len(imgRAW.shape)==0:
        return np.ndarray()
    #lin transf equal
    if minMaxV is None:
        percentiles = [0.1,99.9]
        percentilesVals = np.percentile(imgRAW,percentiles)
        if percentilesVals[0]==percentilesVals[1]:
            return imgRAW.astype(np.uint8)
        minRAW, maxRAW = tuple(percentilesVals)
    else:
        minRAW = minMaxV[0]
        maxRAW = minMaxV[1]
    linearRel = (255 / (maxRAW - minRAW)) if minRAW != maxRAW else 0
    imgRawConv = imgRAW.astype(np.float64)
    img_eq = (imgRawConv-float(minRAW))*linearRel#maximize to 8bits

    return np.clip(img_eq,0,255).astype(np.uint8)

def ConvertToMaxContrastUcharPercent(imgRAW:np.ndarray, percentile=100) -> np.ndarray:
    mv = (100-percentile)/2
    Mv = 100-mv
    percentilesVals = np.percentile(imgRAW,(mv,Mv))
    return ConvertToMaxContrastUchar(imgRAW,percentilesVals)

#overwrite input
def centerMeanSequence(imgs:list[np.ndarray]):
    #Centrar media
    means = list[float]()
    for im in imgs:
        means.append(np.mean(im))
    mean = np.mean(np.array(means))
    for im_i, mean_i in zip(imgs,means):
        im_i += -mean_i + mean

#helper para pseudo color
def gray2PseudoColor(im:np.ndarray, mM):
    imByte = ConvertToMaxContrastUchar(im, mM)
    convImVecColor = cv.applyColorMap(imByte, colormap=cv.COLORMAP_JET)
    return convImVecColor
def gray2PseudoColorPercent(im:np.ndarray, percentm:float, percentM:float):
    mM = np.percentile(im,[percentm,percentM])
    return gray2PseudoColor(im, mM)
# def gray2PseudoColor(im:np.ndarray):
#     return gray2PseudoColorPercent(im,0,100)

def saveImgsToFolder(imgs:list[np.ndarray], folder:str, prefix:str, ext:str):
    res = True
    for i,im in enumerate(imgs):
        res &= cv.imwrite(f'{folder}\\{prefix}_{i}.{ext}', im)
    return res

def saveImgDict(imageDict:ProcDict,logPath:str,logPrefixName:str):
    if os.path.exists(logPath)==False:
        print(f'saveImgDict logPath invalid!\n\t{logPath}')
    else:
        for procKey in imageDict:
            #raw
            cv.imwritemulti(f'{logPath}\\{logPrefixName}_{procKey}.tiff',imageDict[procKey])
            #visible
            imgsVis = helpers_imgs.ConvertToMaxContrastUchar(np.concatenate(\
                imageDict[procKey], axis=1))
            cv.imwrite(f'{logPath}\\{procKey}_byte.jpg',imgsVis)

def saveStitched(framesLists:np.ndarray,deltasReg:list[np.ndarray], absPath:str):
    for data,deltas in zip(framesLists,deltasReg):
        stitched = stitchImgs(data, deltas)
    cv.imwritemulti(absPath, stitched)
#saveStitched