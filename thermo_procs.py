import numpy as np
import matplotlib as plt
import cv2 as cv
from numpy import fft

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

