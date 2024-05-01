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
from globalVars import *
from typedefs import *

def PreProcTermo(imgsList:list[np.ndarray]):
    if not Params.PreprocTermoPars.averagePulses:
        return imgsList

    nPulsesFinal = Params.Input.acquisitionPeriods #no heredado
    if Params.Tasks.dynamicTermo:
        nPulsesFinal = Params.DynamicTermoPars.nPulsesPreserve

    for i,imgs in enumerate(imgsList):
        termoSplit = np.array(np.split(imgs, nPulsesFinal, axis=0))
        imgsList[i] = np.mean(termoSplit, axis=0)

    Params.Input.acquisitionPeriods  = 1

    return imgsList
#PreProcTermo

#TODO Specify preproc termo for each type, now, same for all
def PostProcTermo(procDict:ProcDict)->ProcDict:
    if not Params.Tasks.postprocTermo:
        return procDict

    print('[PostProcTermo]')
    for procKey in procDict:
        imgs = procDict[procKey]
        if Params.PostprocTermoPars.lowPassGaussKsize>3:
            k = Params.PostprocTermoPars.lowPassGaussKsize
            for i,img in enumerate(imgs):
                img = cv.GaussianBlur(img,(k,k), 0)
                imgs[i] = img
        if Params.PostprocTermoPars.centerMeanSet:
            helpers_imgs.centerMeanSequence(imgs)
        if Params.LogData.saveData:
            helpers_imgs.saveImgDict(procDict, f'{Params.outputDir}','postProc_')
    return procDict
#PostProcTermo


