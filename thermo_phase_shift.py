import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math

class PhaseShiftRet:
    psmean=np.ndarray
    psampl=np.ndarray
    pspolar=np.ndarray
    
def pixelWisePhaseShift(phases:list[np.ndarray], betasDeg:list[float])->PhaseShiftRet:
    if len(phases)!=len(betasDeg):
        print(f'[pixelWisePhaseShift] error\n\tlen(phases)!=len(betasDeg)\n\t{len(phases)}!={len(betasDeg)}')
        return None
    if len(phases)==0:
        print('[pixelWisePhaseShift] error\n\tlen(phases)==0')
        return None
    if len(betasDeg)<3:
        print('[pixelWisePhaseShift] len(betasDeg)<3')
        return None

    betasInduct = np.array(betasDeg)*np.pi/180.
    print(f'{betasInduct}')
    betasPatter = betasInduct * 2.0 #polar frequency is 2Hz
    nBetas = len(betasDeg)
    A = np.zeros([nBetas,3])#nBetas X nPars (Pars = [mean,X,Y])
    for i in range(0,nBetas):
        A[i,0]=1
        A[i,1]=math.cos(betasPatter[i])
        A[i,2]=math.sin(betasPatter[i])
    A_inv = np.linalg.pinv(A) #3XnBetas
    assert(A_inv.shape[0]==A.shape[1] and A_inv.shape[1]==A.shape[0])
    print(f'A_inv\n{A_inv}')
    #truncate to zero epsilons
    A_inv = np.where(np.abs(A_inv) < 1e-10, 0, A_inv)
    print(f'A_inv\'\n{A_inv}')
    #weighed sum linear sys whole images
    nRows, nCols = phases[0].shape
    mu_x_y = np.zeros([nRows,nCols,3],dtype=np.float32)
    for i in range(0,nBetas):
        for j in range(0,3):
            if A_inv[j,i]==0:
                continue
            mu_x_y[:,:,j] += phases[i] * A_inv[j,i]
        #j
    #i
    retVal = PhaseShiftRet()
    retVal.psmean = mu_x_y[:,:,0]
    
    #x,y to polar / ample,polarPhase
    retVal.psampl = np.linalg.norm(mu_x_y[:,:,1:],axis=2)
    retVal.pspolar = np.arctan2(mu_x_y[:,:,2],mu_x_y[:,:,1])

    return retVal