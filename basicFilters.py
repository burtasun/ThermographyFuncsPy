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


#TODO Specify preproc termo for each type, now, same for all
def PostProcTermo(procDict:ProcDict)->ProcDict:
    print('[PostProcTermo]')
    for procKey in procDict:
        if Params.PostprocTermoPars.lowPassGaussKsize>3:
            k = Params.PostprocTermoPars.lowPassGaussKsize
            sigma = Params.PostprocTermoPars.lowPassGaussSigma
            for i,img in enumerate(procDict[procKey]):
                img = cv.GaussianBlur(img,(k,k), sigma)
                procDict[procKey][i] = img
        if Params.PostprocTermoPars.centerMeanSet:
            helpers_imgs.centerMeanSequence(procDict[procKey])
        if Params.LogData.saveData:
            helpers_imgs.saveImgDict(procDict, f'{Params.outputDir}','postProc_')
    return procDict
#PostProcTermo
