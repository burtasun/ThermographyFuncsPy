import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

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
    dirThermos =r'C:\Users\benat\source\repos\ThermographyFuncsPy\data'
    fns=[\
        'AHT020__Alpha0_H_Norm.ITvisLockin',\
        'AHT020__Alpha0_HV_Norm.ITvisLockin',\
        'AHT020__Alpha0_V_Norm.ITvisLockin',\
        'AHT020__Alpha0_HVn_Norm.ITvisLockin']
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
    
    psRet = thermo_phase_shift.pixelWisePhaseShift(phases, betasDeg)
    if psRet is None:
        exit(0)
    
    print(f'psRet.psmean\n{psRet.psmean.shape}\n')
    print(f'psRet.psampl\n{psRet.psampl.shape}\n')
    print(f'psRet.pspolar\n{psRet.pspolar.shape}\n')

    imgs = list()
    imgs.append(['mean',psRet.psmean])
    imgs.append(['ampl',psRet.psampl])
    imgs.append(['polar',psRet.pspolar])
    for im in imgs:
        print(f'name: {im[0]}')
        plt.imshow(im[1])
        plt.waitforbuttonpress()