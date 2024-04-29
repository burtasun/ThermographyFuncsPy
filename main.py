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

def iniEnv():
    thermo_io.iniEnv()


def Register():
    print('[Register] not implemented')
    return None
#Register
def DynamicTermo():
    print('[DynamicTermo] not implemented')
    return None
#DynamicTermo
def PreProcTermo(procDict:ProcDict):
    print('[PreProcTermo] not implemented')
    return None
#PreProcTermo 


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



def PhaseShiftProc(imgs:list[np.ndarray])->np.ndarray:
    psRet = thermo_phase_shift.pixelWisePhaseShift(imgs, Params.Input.betasDeg)
    if psRet is None:
        return None
    if Params.LogData.saveData:
        sampledPhaseShift = thermo_phase_shift.samplePhaseShift(psRet)
    flowAcc = thermo_phase_shift.PhaseShiftedOpticalFlow(psRet, None, True)
    vorticity = thermo_phase_shift.IntegrateOpticalFlow(flowAcc, 21)
    # vorticityTruncate = np.copy(vorticity)
    # vorticityTruncate[vorticityTruncate<0] = 0
    return vorticity                

#PhaseShiftProc
def FusedProc(procDict:ProcDict):
    print('[FusedProc]')
    #for each list[np.ndarray] -> np.ndarray
    fusedProcDict = ProcDict()
    for procKey in procDict:
        imgs = procDict[procKey]
        for fuseProc in Params.Tasks.fusedProcs:
            if fuseProc == FusedProcs.PhaseShift:
                fusedProc = PhaseShiftProc(imgs)
                if fusedProc is None:
                    continue
                keySave = f'{procKey}_{str(fuseProc)}'
                fusedProcDict[keySave] = [fusedProc]
            #FusedProcs.PhaseShift
        #Params.Tasks.fusedProcs
    #procDict
    if Params.LogData.saveData:
        helpers_imgs.saveImgDict(fusedProcDict,f'{Params.outputDir}','fused_')
    return fusedProcDict
#FusedProc



if __name__=='__main__':
    print('''
--------------------------------------------
    Thermographic processing algorithms
--------------------------------------------
''')
    iniEnv()

    tasks = Params.Tasks

    #READ INPUT
    dataInputs = list[np.ndarray]()#nImgs,Rows,Cols = dataInputs[0].shape / full termo each list elem
    for fn in Params.Input.fns:
        termos, acquisitionPeriods = thermo_io.loadThermo(f'{Params.Input.dirFiles}\\{fn}',1,1)
        dataInputs.append(termos[0])
        Params.Input.acquisitionPeriods = acquisitionPeriods
    #REGISTER INPUT
    if tasks.register:
        for data in dataInputs:
            data = Register(data, Params.RegisterPars)
    #DYNAMIC TERMO
    if tasks.dynamicTermo:
        for data in dataInputs:
            data = DynamicTermo(data, Params.DynamicTermoPars)
    #PREPROC TERMO
    if tasks.dynamicTermo:
        for data in dataInputs:
            data = PreProcTermo(data, Params.DynamicTermoPars)
    #PROC TERMO
    procDataDict = ProcDict
    if len(tasks.procTermo)>0:
        procDataDict = thermo_procs.ProcTermo(dataInputs)
    #POSTPROC TERMO
    if tasks.postprocTermo:
        procDataDict = PostProcTermo(procDataDict)
    #FUSED PROC TERMO
    fusedProcData = ProcDict()
    if len(tasks.fusedProcs)>0:
        fusedProcData = FusedProc(procDataDict)