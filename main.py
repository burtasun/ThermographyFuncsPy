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
from basicFilters import *
from fusedProcs import *
from digStabilization import *
from dynamicTermo import *

def iniEnv():
    thermo_io.iniEnv()

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
    #TODO a params input
    offsetFrame = 60*60+3#Params.Input.offFrames
    dataInputs[0]=dataInputs[0][offsetFrame:,...][:240,...]

    #REGISTER INPUT
    deltasReg = None
    if tasks.register:
        deltasReg = list[np.ndarray]()
        for data in dataInputs:
            deltasReg.append(Register(data, False, False))
    
        if Params.LogData.saveStitched:
            for data,deltas in zip(dataInputs,deltasReg):
                stitched = stitchImgs(data, deltas)
                cv.imwritemulti(Params.outputDir + "\\stitchedSeq.tiff", stitched)
                
    for i, (data,deltas) in enumerate(zip(dataInputs,deltasReg)):
        dataInputs[i] = TempFilter(data,deltas,Params.tempFilterRadius)
        
    if Params.LogData.saveStitched:
        for data,deltas in zip(dataInputs,deltasReg):
            stitched = stitchImgs(data, deltas)
            cv.imwritemulti(Params.outputDir + "\\stitchedSeqFilt.tiff", stitched)
    
    #DYNAMIC TERMO
    if tasks.dynamicTermo:
        dataDynamic = list[np.ndarray]()
        for i, (data, deltas) in enumerate(zip(dataInputs, deltasReg)):
            dataDynamic.extend(dynamicTermo(data, deltas))
        dataInputs = dataDynamic
    
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