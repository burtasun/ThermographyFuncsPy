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
from postProcTermo import *
from fusedProcs import *
from digStabilization import *
def iniEnv():
    thermo_io.iniEnv()



def DynamicTermo():
    print('[DynamicTermo] not implemented')
    return None
#DynamicTermo
def PreProcTermo(procDict:ProcDict):
    print('[PreProcTermo] not implemented')
    return None
#PreProcTermo 



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
    deltasReg = None
    if tasks.register:
        deltasReg = list[np.ndarray]()
        for data in dataInputs:
            deltasReg.append(Register(data, False, True))
    #DYNAMIC TERMO
    if tasks.dynamicTermo:
        for data in dataInputs:
            data = DynamicTermo(data, Params.DynamicTermoPars)
    elif deltasReg is not None:
        for data,deltas in zip(dataInputs,deltasReg):
            data = stitchImgs(data, deltas)
            cv.imwritemulti(Params.outputDir + "\\stitchedSeq.tiff", data)

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