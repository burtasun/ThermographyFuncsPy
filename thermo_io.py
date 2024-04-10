import numpy as np
import matplotlib as plt
import cv2 as cv
import tifffile as tiff
import subprocess
import time
import os
import math

def iniEnv():
    pathEnv = os.environ['PATH']
    # print(f'{pathEnv}')
    append = r'C:\OPENCV\repo\build\install\x64\vc16\bin;C:\vcpkg\installed\x64-windows\bin'
    pathEnvNew = f'{pathEnv};{append}'
    os.environ['PATH'] = pathEnvNew
    # print(f'{pathEnv}')

#clientePy de ConvertEdevis2Tiff.exe
    #TODO test multiple directions
def loadThermo(
        abspath:str,
        nPulsesPerDirection=0,
        nDirections=0)-> [list[np.ndarray],int]:

    tini = time.time()
    errCodes = {
        0:"ok",
        1:"notFound",
        2:"numPulsesWrong",
        3:"errorWhileReading",
        4:"errorWhileWriting",
        4:"errorInFFT",
        5:"incoherenNframes"}
    
    exe = r'C:\Users\benat\source\repos\ThermographyFuncs\x64\Release\ConvertEdevis2Tiff.exe'

    inputPath = abspath
    outputPath = f'{inputPath[:inputPath.rfind(".")]}'

    argsDict={
        'input': inputPath,
        'output': outputPath,
        'mode': 'thermogram', #thermogram o fft
        'npulsesPerDirection': nPulsesPerDirection,
        'nDirections': nDirections
    }

    args=\
        argsDict['input'] + ' ' +\
        argsDict['output'] + ' ' +\
        argsDict['mode'] + ' ' + \
        str(argsDict['npulsesPerDirection']) + ' ' +\
        str(argsDict['nDirections'])
        
    exeArgs=exe + ' ' + args
    timeOut = 2000
    print(f'{exeArgs}')
    process = subprocess.Popen(exeArgs, stdout=subprocess.PIPE, creationflags=0x08000000)
    process.wait(timeOut) #internal ~= 0.3sg
    ret = process.returncode
    acquisitionPeriods = ret >> 16
    ret = ret & ((1<<8)-1)

    if not (ret in errCodes):
        print(f'codigo {ret} no reconocido')
        return None
    if errCodes[ret]!='ok':
        print(f'error al ejecutar el script {ret}: {errCodes[ret]}')
        return None
    
    termos=list[np.ndarray]()
    fullFFT=False
    for i in range(0, argsDict['nDirections']):
        if nDirections == 1:
            pathRead = argsDict['output'] + '.tiff'
        else:
            pathRead = argsDict['output']+'_thermogram_'+str(i)+'.tiff'
        termo = tiff.imread(pathRead)
        if termo is None:
            print(f'error al leer {pathRead}')
            return None
        os.remove(pathRead)
        termos.append(termo)
    tend = time.time()
    print(f'elapsed time {str(tend-tini)}')
    return termos, acquisitionPeriods

#test
if __name__=='__main__':
    iniEnv()
    fn=r'C:\Users\benat\source\repos\ThermographyFuncsPy\data\AHT020__Alpha0_H_Norm.ITvisLockin'
    termos, acquisitionPeriods = loadThermo(fn,1,1)
    for t in termos:
        print(f'{str(t.shape)}')