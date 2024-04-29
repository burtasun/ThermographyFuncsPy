import sys
import os
import numpy as np
import matplotlib as plt
import cv2 as cv
from numpy import fft

from globalVars import *
import helpers_imgs

#harmonicNum: 0 todos, o uno cualquiera
def FFT(termo:np.ndarray, harmonicNum=0) -> [np.ndarray,np.ndarray]:
    fftTermo = fft.fft(termo, axis=0)
    dims = fftTermo.shape[0]
    if harmonicNum!=0:
        fftTermo=fftTermo[harmonicNum]
        dims = 1
    mag = np.sqrt(np.power(fftTermo.real,2) + np.power(fftTermo.imag,2)).astype(np.float32)
    pha = np.arctan2(fftTermo.real, fftTermo.imag).astype(np.float32)
    return mag, pha




'''ProcTermo / WRAP CALL FUNCS
TODO implement procs
TODO visualization / debug / log
TODO abstract params algs
'''
def ProcTermo(
        termos:list[np.ndarray]#nImgs,Rows,Cols = dataInputs[0].shape
) ->dict[str,list[np.ndarray]]:
    
    outputDict = dict[str,list[np.ndarray]]()#ej. {'FFT_Phase': list[np.ndarray], 'FFT_Mag': list[np.ndarray]}
    for ProcType in Params.Tasks.procTermo:
        if ProcType == Procs.FFT: #keys FFT_Phase, FFT_Mag
            print('ProcTermo, Procs.FFT')
            phases, magnitudes = list[np.ndarray](), list[np.ndarray]()
            for t in termos:
                mag, pha = FFT(t, Params.Input.acquisitionPeriods)
                phases.append(pha)
                magnitudes.append(mag)
            outputDict['FFT_Phases'] = phases
            outputDict['FFT_Mags'] = magnitudes
        #Procs.FFT
                
        if ProcType == Procs.DeltaT:
            print('ProcTermo, Procs.DeltaT not implemented!')
        if ProcType == Procs.HOS:
            print('ProcTermo, Procs.HOS not implemented!')
        if ProcType == Procs.PCA:
            print('ProcTermo, Procs.PCA not implemented!')
        if ProcType == Procs.PCT:
            print('ProcTermo, Procs.PCT not implemented!')
        if ProcType == Procs.SenoidalFit:
            print('ProcTermo, Procs.SenoidalFit not implemented!')
        if ProcType == Procs.TSR:
            print('ProcTermo, Procs.TSR not implemented!')
            
        if Params.LogData.saveData:
            helpers_imgs.saveImgDict(outputDict,f'{Params.outputDir}','proc_')
            
    return outputDict
#ProcTermo