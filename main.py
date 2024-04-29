import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tifffile as tiff
import thermo_io
import thermo_procs
import thermo_phase_shift
import helpers_imgs
from globalVars import *
def iniEnv():
    thermo_io.iniEnv()

if __name__=='__main__':
    print('''
--------------------------------------------
    Thermographic processing algorithms
--------------------------------------------
''')
    iniEnv()
        
    phases = list[np.ndarray]()
    for fn in Params.Input.fns:
        termos, acquisitionPeriods = thermo_io.loadThermo(f'{Params.Input.dirFiles}\\{fn}',1,1)
        for t in termos:
            # print(f'{str(t.shape)}')
            mag, pha = thermo_procs.FFT(t, acquisitionPeriods)
            phases.append(pha)
            # plt.imshow(pha)
            # plt.waitforbuttonpress()
    # cv.imwritemulti(f'{Params.outputDir}\\phases.tiff',phases)
            

    #basic preproc
    cv.imwrite(f'{Params.outputDir}\\phasesConcat.tiff',np.hstack(phases))
    helpers_imgs.centerMeanSequence(phases)
    cv.imwrite(f'{Params.outputDir}\\phasesConcatCentered.tiff',np.hstack(phases))
    cv.imwrite(f'{Params.outputDir}\\phasesConcatCenteredByte.jpg',helpers_imgs.ConvertToMaxContrastUchar(np.hstack(phases)))

    imgs = list()
    for phase in phases:
        imgs.append(helpers_imgs.ConvertToMaxContrastUchar(phase))

    psRet = thermo_phase_shift.pixelWisePhaseShift(phases, Params.Input.betasDeg)
    if psRet is None:
        exit(0)
            
    sampledPhaseShift = thermo_phase_shift.samplePhaseShift(psRet)

    flowAcc = thermo_phase_shift.PhaseShiftedOpticalFlow(psRet, None, True)

    vorticity = thermo_phase_shift.IntegrateOpticalFlow(flowAcc, 21)

    tiff.imwrite(f'{Params.outputDir}\\vorticity.tiff',vorticity)
    tiff.imwrite(f'{Params.outputDir}\\vorticityByte.jpg',helpers_imgs.ConvertToMaxContrastUchar(vorticity))
    vorticityTruncate = np.copy(vorticity)
    vorticityTruncate[vorticityTruncate<0] = 0
    tiff.imwrite(f'{Params.outputDir}\\vorticityTruncateByte.jpg',\
        helpers_imgs.ConvertToMaxContrastUchar(vorticityTruncate))