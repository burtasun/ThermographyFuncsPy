import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math
import helpers_imgs
from globalVars import *

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


    # print(f'retVal.psmean\n{retVal.psmean.shape}\n')
    # print(f'retVal.psampl\n{retVal.psampl.shape}\n')
    # print(f'retVal.pspolar\n{retVal.pspolar.shape}\n')
    # dirOut = r'C:\Users\benat\source\repos\ThermographyFuncsPy\out'
    # imgs = list()
    # imgs.append(['mean',retVal.psmean])
    # imgs.append(['ampl',retVal.psampl])
    # imgs.append(['polar',retVal.pspolar])
    # for im in imgs:
    #     print(f'name: {im[0]}')
    #     plt.imshow(im[1])
    #     plt.waitforbuttonpress()
    #     imByte = helpers_imgs.ConvertToMaxContrastUchar(im[1])
    #     cv.imwrite(f'{dirOut}\\{im[0]}.tiff', im[1])
    #     cv.imwrite(f'{dirOut}\\{im[0]}_Byte.jpg', imByte)
        

    return retVal

def cosineImg(ampl:np.ndarray, polar:np.ndarray, offset:float) ->np.ndarray:
    return ampl * np.cos(polar-offset)

def samplePhaseShift(phaseShiftPars:PhaseShiftRet, nFrames = 36, noMean = True):
    #check input
    if \
        (phaseShiftPars.psampl.shape!=phaseShiftPars.psmean.shape) or \
        (phaseShiftPars.psampl.shape!=phaseShiftPars.pspolar.shape) or \
        phaseShiftPars.psampl.shape[0]==0 or nFrames < 1:

        print(f'[samplePhaseShift] inconsistent shapes')
        return None

    deltaRad = 2. * math.pi / float(nFrames)

    samples = np.zeros(phaseShiftPars.psampl.shape + (nFrames,), np.float32)
    for i in range(0,nFrames):
        val = cosineImg(phaseShiftPars.psampl,phaseShiftPars.pspolar, float(i)*deltaRad)
        if noMean == False:
            val+=phaseShiftPars.psmean
        # print(f'{val.shape}')
        # plt.imshow(val)
        # plt.waitforbuttonpress()
    return samples


#aux overlay flow
def logFlowIm(imIn:np.ndarray, flowIn:np.ndarray, scale = 4, scaleArrows = 16, thickness = 2, sub = 4):
    im = cv.resize(imIn, (imIn.shape[1]*scale, imIn.shape[0]*scale), interpolation=cv.INTER_LINEAR)
    im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    # im = imIn

    for i in range(0,flowIn.shape[0],sub):#row
        for j in range(0,flowIn.shape[1],sub):#col
            if (flowIn[i,j,0]==flowIn[i,j,1]) and flowIn[i,j,0]==0:
                continue
            #OpenCV points X and Y / numpy index row/y and col/x
            p1 = np.array([j*scale,i*scale]).astype(np.int32)
            p2 = np.array([
                          p1[0]+ flowIn[i,j,0] * scaleArrows,\
                          p1[1]+ flowIn[i,j,1] * scaleArrows]).astype(np.int32)
            cv.arrowedLine(im, p1, p2, (0, 0, 255), thickness, tipLength=0.5, line_type=cv.LINE_AA)
    return im


class OF_Pars:
    winSzConv = 21
    sigmConv = 0
    nFrames = 48
    #wrap pars OF_OpenCV
    pyr_scale = 0.5
    poly_sigma = 1.3
    levels = 1
    winSzOF = 7
    iterations = 3
    poly_n = 7
    flags = 0
#class OF_Pars
    

#accumulated optical flow
def PhaseShiftedOpticalFlow (\
    phaseShiftPars:PhaseShiftRet,\
    ofPars:OF_Pars, log=False) -> np.ndarray:
    
    if ofPars is None:
        ofPars = OF_Pars()

    nFrames = ofPars.nFrames
    winSz = ofPars.winSzConv
    sigmConv = ofPars.sigmConv

	#check inputs
    
    if \
        (phaseShiftPars.psampl.shape!=phaseShiftPars.psmean.shape) or \
        (phaseShiftPars.psampl.shape!=phaseShiftPars.pspolar.shape) or \
        phaseShiftPars.psampl.shape[0]==0 or nFrames < 2 or winSz < 3:

        print(f'[PhaseShiftedOpticalFlow] inconsistent shapes or params')
        return None

    deltaRad = 2 * math.pi / float(nFrames)
    
    ampl = phaseShiftPars.psampl
    polar = phaseShiftPars.pspolar
    mean = phaseShiftPars.psmean

    #frame 0
    #   optical flow in opencv as byte images
    f0 = helpers_imgs.ConvertToMaxContrastUchar(\
        cosineImg(ampl,polar,0))
    fPrev = f0#shallow copy?
    fCurr = np.zeros(f0.shape,f0.dtype)
    
    #current flow
    flowCurr = np.zeros(ampl.shape+(2,),ampl.dtype)
    #accumulated flow
    flowAcc = np.zeros(ampl.shape+(2,),ampl.dtype)
    
    for i in range(1,nFrames+1):
        radCurr = deltaRad * float(i%nFrames)
        fCurr = helpers_imgs.ConvertToMaxContrastUchar(\
			cosineImg(ampl,polar,radCurr))

        #componentes u,v // x,y optical flow
        cv.calcOpticalFlowFarneback(\
            fPrev,fCurr,flowCurr,\
            ofPars.pyr_scale, ofPars.levels, ofPars.winSzOF, ofPars.iterations,\
            ofPars.poly_n, ofPars.poly_sigma, ofPars.flags)
        print(f'flowCurr.shape {flowCurr.shape}')
        flowAcc += flowCurr
        fPrev = fCurr
        # if log:
        #     fPrevByte = helpers_imgs.ConvertToMaxContrastUchar(fPrev)
        #     flowOverlay = logFlowIm(fPrevByte,flowCurr)
        #     plt.imshow(flowOverlay)
        #     plt.waitforbuttonpress()
    #frames
    if log:
        fPrevByte = helpers_imgs.ConvertToMaxContrastUchar(fPrev)
        flowOverlay = logFlowIm(fPrevByte,flowAcc/float(nFrames))
        plt.imshow(flowOverlay)
        plt.waitforbuttonpress()
        cv.imwrite(f'{Params.outputDir}\\flowOverlayByte.jpg', helpers_imgs.ConvertToMaxContrastUchar(flowOverlay))
    return flowAcc

#PhaseShiftedOpticalFlow


class PhaseShiftOpticalFlowIntegrals:
    vorticity=np.ndarray
    divergence=np.ndarray
#PhaseShiftOpticalFlowIntegrals
    
def IntegrateOpticalFlow(flowAcc:np.ndarray, winSz = 11):
    #check input
    if winSz < 3 or len(flowAcc.shape)!=3 or flowAcc.shape[0]==0:
        print('[IntegrateOpticalFlow] invalid inputs')
        return None
    
    #convolve / augment borders
    b = int((winSz - 1) / 2) #border
    uv = cv.copyMakeBorder(flowAcc, b + 1, b + 1, b + 1, b + 1, cv.BORDER_REFLECT)


    #TODO ponderar radialmente
    # 	cv::Mat w;
    # 	if constexpr (gaussWeight) {
    # 		auto w1 = cv::getGaussianKernel(winSz, sigmWeight, CV_32F);
    # 		cv::Mat w2;
    # 		cv::transpose(w1, w2);
    # 		w = w1 * w2;
    # 		std::cout << "w\n" << w << "\n";
    # 	}


    #cache unitary vector window kernel
    rij = np.zeros([b*2+1,b*2+1,2],np.float32)
    for i in range(-b,b + 1): #row,y
        for j in range(-b,b + 1): #col,x
            if i==0 and j==0: continue
            rij[i+b,j+b,0] = float(j) / math.sqrt(float(i) * float(i) + float(j) * float(j))
            rij[i+b,j+b,1] = float(i) / math.sqrt(float(i) * float(i) + float(j) * float(j))

    #vorticity
    #   as 2D filter subtraction
    yxCompCross = cv.filter2D(flowAcc[:,:,1],cv.CV_32F, rij[:,:,0])
    xyCompCross = cv.filter2D(flowAcc[:,:,0],cv.CV_32F, rij[:,:,1])
    outputCross = xyCompCross - yxCompCross

    return outputCross