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
from globalVars import ParamsClass, Params, initGlobalVars

from basicFilters import *
from fusedProcs import *
from digStabilization import *
from dynamicTermo import *
from misc import *
from tempFilter_sharedMem import TempFilter

def iniEnv():
    initGlobalVars()
    thermo_io.iniEnv()
    if os.path.exists(Params.outputDir)==False:
        os.mkdir(Params.outputDir)
        if os.path.exists(Params.outputDir)==False:
            raise(f"Could not locate output directory\n\t{Params.outputDir}")

if __name__=='__main__':
    print('''
--------------------------------------------
    Thermographic processing algorithms
--------------------------------------------
''')
    iniEnv()

    prof = helperProfile()

    tasks = Params.Tasks

    #READ INPUT
    dataInputs = list[np.ndarray]()#nImgs,Rows,Cols = dataInputs[0].shape / full termo each list elem
    for fn in Params.Input.fns:
        termos, acquisitionPeriods = thermo_io.loadThermo(f'{Params.Input.dirFiles}\\{fn}',1,1)
        dataInputs.append(termos[0])
        Params.Input.acquisitionPeriods = acquisitionPeriods
    dataInputs[0]=dataInputs[0][Params.Input.offFrames:,...]
    prof.ticTocProfile("Read")

    #REGISTER INPUT
    deltasReg = None
    if tasks.register:
        deltasReg = list[np.ndarray]()
        for data in dataInputs:
            deltasReg.append(Register(data, False, True, smoothTrajectorySigma=0, descriptorType='AKAZE'))
    
        if Params.LogData.saveStitched:
            helpers_imgs.saveStitched(dataInputs, deltasReg, Params.outputDir + "\\stitchedSeq.tiff")
        prof.ticTocProfile("Register")

    for i, (data,deltas) in enumerate(zip(dataInputs,deltasReg)):
        dataInputs[i] = TempFilter(data,deltas,Params.tempFilterRadius)
    prof.ticTocProfile("TemporalFilter")

    if Params.LogData.saveStitched:
        helpers_imgs.saveStitched(dataInputs,deltasReg,Params.outputDir + "\\stitchedSeqFilt.tiff")
        prof.ticTocProfile("SaveStitched")

    #DYNAMIC TERMO / sync and thermal drift compensation
    if tasks.dynamicTermo:
        dataDynamic = list[np.ndarray]()
        for i, (data, deltas) in enumerate(zip(dataInputs, deltasReg)):
            dataDynamic.extend(dynamicTermo(data, deltas))
        dataInputs = dataDynamic
        prof.ticTocProfile("DynamicTermo")

    #PREPROC TERMO
    dataInputs = PreProcTermo(dataInputs)
    if Params.LogData.saveData:
        for i,imgs in enumerate(dataInputs):
            cv.imwritemulti(f'{Params.outputDir}\\NormAverage_{i}.tiff', imgs)
    prof.ticTocProfile("PreProcTermo")

    #PROC TERMO / fft, pca, etc.
    procDataDict = thermo_procs.ProcTermo(dataInputs)
    prof.ticTocProfile("TermoProc")

    #POSTPROC TERMO / filters
    procDataDict = PostProcTermo(procDataDict)
    prof.ticTocProfile("TermoPostProc")

    #FUSED PROC TERMO / phase-shift
    fusedProcData = ProcDict()
    if len(tasks.fusedProcs)>0:
        fusedProcData = FusedProc(procDataDict)
        prof.ticTocProfile("TermoFusedProc")

    prof.printProfile()