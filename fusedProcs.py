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

def PhaseShiftProc(imgs:list[np.ndarray])->np.ndarray:
    psRet = thermo_phase_shift.pixelWisePhaseShift(imgs, Params.Input.betasDeg)
    if psRet is None:
        return None
    if Params.LogData.saveData:
        sampledPhaseShift = thermo_phase_shift.samplePhaseShift(psRet)
    flowAcc = thermo_phase_shift.PhaseShiftedOpticalFlow(psRet, None, True)
    vorticityAndDivergence = thermo_phase_shift.IntegrateOpticalFlow(flowAcc, 21)
    return vorticityAndDivergence

#PhaseShiftProc
#TODO weighted sums / variance / etc.
def FusedProc(procDict:ProcDict):
    print('[FusedProc]')
    #for each list[np.ndarray] -> np.ndarray
    fusedProcDict = ProcDict()
    for procKey in procDict:
        imgs = procDict[procKey]
        for fuseProc in Params.Tasks.fusedProcs:
            if fuseProc == FusedProcs.PhaseShift:
                fusedProcImgs = PhaseShiftProc(imgs)
                if fusedProcImgs is None:
                    continue
                keySave = f'{procKey}_{str(fuseProc)}'
                fusedProcDict[keySave] = fusedProcImgs
            #FusedProcs.PhaseShift
        #Params.Tasks.fusedProcs
    #procDict
    if Params.LogData.saveData:
        helpers_imgs.saveImgDict(fusedProcDict,f'{Params.outputDir}','fused_')
    return fusedProcDict
#FusedProc