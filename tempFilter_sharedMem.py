import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tifffile as tiff
from scipy import ndimage

from globalVars import *
from typedefs import *
from digStabilization import highPassFilter, displacementToHomogMat

import psutil

import time
from multiprocessing import Pool, RawArray #shared_mem multiProc


#Temporal filter with multi-process and shared memory
#Global variables shared mem
var_dict = {}

#Ini shared mem for process pool
def init_sharedMem(imgsOut, imgsOutShape, \
                imgs, imgsShape, \
                deltas, deltasShape, \
                k,b,w,h   ):
    var_dict['imgsOut'] = imgsOut
    var_dict['imgsOutShape'] = imgsOutShape
    var_dict['imgs'] = imgs
    var_dict['imgsShape'] = imgsShape
    var_dict['deltas'] = deltas
    var_dict['deltasShape'] = deltasShape
    var_dict['k']=k
    var_dict['b']=b
    var_dict['w']=w
    var_dict['h']=h

def destruct_sharedMem():
    var_dict['imgsOut'] = []
    var_dict['imgsOutShape'] = []
    var_dict['imgs'] = []
    var_dict['imgsShape'] = []
    var_dict['deltas'] = []
    var_dict['deltasShape'] = []
    var_dict['k']=[]
    var_dict['b']=[]
    var_dict['w']=[]
    var_dict['h']=[]
    var_dict.clear()

#temp filter
#   TODO wrap this file for other algorithms
def workerProcess(i):

    # print(f'workerProcess {i}')
    # print(f'workerProcess {rangeProcWorker[0]} - {workerProcess[-1]}')
    now = time.time()

    imgsOut = np.frombuffer(var_dict['imgsOut'],dtype=np.float32).reshape(var_dict['imgsOutShape'])
    imgs = np.frombuffer(var_dict['imgs'],dtype=np.float32).reshape(var_dict['imgsShape'])
    deltas = np.frombuffer(var_dict['deltas'],dtype=np.float32).reshape(var_dict['deltasShape'])
    k = int(var_dict['k'])
    b = int(var_dict['b'])
    w = int(var_dict['w'])
    h = int(var_dict['h'])

    imgsHp_i = list[np.ndarray]()
    imgsHp_i.append(highPassFilter(imgs[i,...],k))
    imgsLp_i = cv.blur(imgs[i,...],(k, k))
    for j in range(max(0,i-b),min(i+b+1,len(deltas))):
        if j==i:
            continue
        tf_i_j = displacementToHomogMat(deltas[i]-deltas[j])
        imgsHp_j = highPassFilter(imgs[j,...])
        imgsHp_i_j = cv.warpAffine(imgsHp_j , tf_i_j[:2,...], (w,h), None, 
            borderValue=0, borderMode=cv.BORDER_CONSTANT,
            flags=cv.INTER_LINEAR)
        imgsHp_i.append(imgsHp_i_j)
    imgsHp_i = np.array(imgsHp_i)
    #Median excludes outlier/non conforming pixels from mean / and avoids overlap window masking
    imgsHp_i = np.median(imgsHp_i,axis=0)
    imgsTempFilt = imgsHp_i + imgsLp_i
    imgsOut[i,...]=imgsTempFilt
    #loop
    # print(f'workerProcess END {rangeProcWorker[0]} - {workerProcess[-1]}')
    # print(f'workerProcess END {i}')
    
    return f'{i}_{float(time.time()-now):.3f}s'
    



#Temp filter 
#   promedia componente alta freq con frames adyacentes roto-trasladados
#   dimensiones frames identicas a raw
def TempFilter(\
    imgsIn:np.ndarray,
    deltasIn:list[np.ndarray] | None,
    averageRad = 2,
    kMeanBlur = 11
) -> np.ndarray:

    if averageRad <=0:
        return imgsIn
    
    nImgs,h,w=imgsIn.shape
    b = averageRad
    k = kMeanBlur
    
    #flatten array
    deltas = np.array(deltasIn)
    print(f'deltasShape {deltas.shape}')
    # if deltasIn is None:
        # deltasIn=np.split(np.zeros((imgs.shape[0],2),np.float32),imgs.shape[0],axis=0) #TODO niapa
    
    #init shared mem buffers
    imgsShape = imgsIn.shape
    #   shared buffer
    imgsShared = RawArray('f', imgsShape[0]*imgsShape[1]*imgsShape[2]) 
    imgs_np = np.frombuffer(imgsShared,dtype=np.float32).reshape(imgsShape) #shallow copy
    #   deep copy to shared buffer
    np.copyto(imgs_np, imgsIn)
    
    imgsOutShape = imgsShape
    imgsOutShared = RawArray('f', imgsOutShape[0]*imgsOutShape[1]*imgsOutShape[2]) 
    
    deltasShape = deltas.shape
    deltasShared = RawArray('f', deltasShape[0]*deltasShape[1]) 
    deltas_np = np.frombuffer(deltasShared,dtype=np.float32).reshape(deltasShape) #shallow copy
    #   deep copy to shared buffer
    np.copyto(deltas_np, deltas)
    

    nProcesses = psutil.cpu_count(logical=False)#physical cores not logical ones

    nImgs = min(imgsIn.shape[0], len(deltasIn))

    before = time.time()
    initArgsShared = (imgsOutShared,imgsOutShape,imgsShared,imgsShape,deltasShared,deltasShape,k,b,w,h)
    with Pool(processes=nProcesses, initializer=init_sharedMem, initargs=initArgsShared) as pool:
        result = pool.map(workerProcess, range(nImgs))
        # print('Results (pool):\n', str(result))

    print(f'Total time {float(time.time()-before):.3f}s')
    imgsOut_np = np.frombuffer(imgsOutShared,dtype=np.float32).reshape(imgsOutShape) #shallow copy
    imgsOut_Local = np.copy(imgsOut_np)

    destruct_sharedMem()

    # listImgsDbg = list[np.ndarray]()
    # for i in range(0,imgsOut_Local.shape[0]):
    #     listImgsDbg.append(\
    #         np.concatenate((imgsIn[i],imgsOut_Local[i]),axis=1))
    # cv.imwritemulti(Params.outputDir+"\\tempFiltdbg.tiff",np.array(listImgsDbg,np.float32))

    return imgsOut_Local
