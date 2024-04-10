import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import thermo_io
import thermo_procs

def iniEnv():
    thermo_io.iniEnv()

if __name__=='__main__':
    print('''
--------------------------------------------
    Thermographic processing algorithms
--------------------------------------------
''')
    iniEnv()
    fn=r'C:\Users\benat\source\repos\ThermographyFuncsPy\data\AHT020__Alpha0_H_Norm.ITvisLockin'
    termos, acquisitionPeriods = thermo_io.loadThermo(fn,1,1)
    for t in termos:
        print(f'{str(t.shape)}')
        mag, pha = thermo_procs.FFT(t, acquisitionPeriods)
        plt.imshow(pha)
        plt.waitforbuttonpress()