import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tifffile as tiff
import thermo_io
import thermo_procs
import thermo_phase_shift

def iniEnv():
    thermo_io.iniEnv()

if __name__=='__main__':
    print('''
--------------------------------------------
    Thermographic processing algorithms
--------------------------------------------
''')
    iniEnv()

    #input
    #   TODO serializar
    outputDir = '.\\out'
    dirThermos =r'D:\Datasets\Termografias\Phase-Shifted_Induction_Thermography\POD\SetAuthentic\Cache'
    fns=[\
        'AHT_020_FlirX6541sc_220fps_3kW_B035_j000_30kHz_100PWM_MP_3P_5Hz_L_220601.ITvisLockin',\
        'AHT_020_FlirX6541sc_220fps_3kW_B035_j045_30kHz_90PWM_MP_3P_5Hz_L_220601.ITvisLockin',\
        'AHT_020_FlirX6541sc_220fps_3kW_B035_j090_30kHz_100PWM_MP_3P_5Hz_L_220601.ITvisLockin',\
        'AHT_020_FlirX6541sc_220fps_3kW_B035_j135_30kHz_90PWM_MP_3P_5Hz_L_220601.ITvisLockin'\
        ]
    betasDeg=[0,45,90,135]
        
        
        
    phases = list[np.ndarray]()
    for fn in fns:
        termos, acquisitionPeriods = thermo_io.loadThermo(f'{dirThermos}\\{fn}',1,1)
        for t in termos:
            # print(f'{str(t.shape)}')
            mag, pha = thermo_procs.FFT(t, acquisitionPeriods)
            phases.append(pha)
            # plt.imshow(pha)
            # plt.waitforbuttonpress()
    cv.imwritemulti(f'{outputDir}\\phases.tiff',phases)

    psRet = thermo_phase_shift.pixelWisePhaseShift(phases, betasDeg)
    if psRet is None:
        exit(0)
            
    sampledPhaseShift = thermo_phase_shift.samplePhaseShift(psRet)

    flowAcc = thermo_phase_shift.PhaseShiftedOpticalFlow(psRet, None, True)

    vorticity = thermo_phase_shift.IntegrateOpticalFlow(flowAcc, 21)
    plt.imshow(vorticity)
    plt.waitforbuttonpress()
    tiff.imwrite(f'{outputDir}\\vorticity.tiff',vorticity)