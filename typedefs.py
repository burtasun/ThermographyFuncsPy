import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tifffile as tiff
import thermo_io
import thermo_procs
import thermo_phase_shift
import helpers_imgs
from globalVars import *

#type aliases
ProcDict = dict[str,list[np.ndarray]] #ej. {'FFT_Phase': list[np.ndarray], 'FFT_Mag': list[np.ndarray]}
