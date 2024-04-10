import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math

def ConvertToMaxContrastUchar(imgRAW:np.ndarray) -> np.ndarray:
    if len(imgRAW.shape)==0:
        return np.ndarray()
    #lin transf equal
    percentiles = [0.1,99.9]
    percentilesVals = np.percentile(imgRAW,percentiles)
    if percentilesVals[0]==percentilesVals[1]:
        return imgRAW.astype(np.uint8)
    minRAW, maxRAW = tuple(percentilesVals)

    linearRel = (255 / (maxRAW - minRAW)) if minRAW != maxRAW else 0
    imgRawConv = imgRAW.astype(np.float64)
    img_eq = (imgRawConv-float(minRAW))*linearRel#maximize to 8bits
    
    return np.clip(img_eq,0,255).astype(np.uint8)