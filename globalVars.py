from enum import Enum

class Procs(Enum):
    FFT=0,#default
    DeltaT=1,
    HOS=2,
    PCA=3,
    PCT=4,
    SenoidalFit=5,
    TSR=6,

class FusedProcs(Enum):
    PhaseShift=0

#Singleton global
class Params:
    outputDir = '.\\out'
    threshOF = float
    
    class Input:
        dirFiles =r'D:\Datasets\Termografias\Phase-Shifted_Induction_Thermography\POD\SetAuthentic\Cache'
        fns=[\
            'AHT_020_FlirX6541sc_220fps_3kW_B035_j000_30kHz_100PWM_MP_3P_5Hz_L_220601.ITvisLockin',\
            'AHT_020_FlirX6541sc_220fps_3kW_B035_j045_30kHz_90PWM_MP_3P_5Hz_L_220601.ITvisLockin',\
            'AHT_020_FlirX6541sc_220fps_3kW_B035_j090_30kHz_100PWM_MP_3P_5Hz_L_220601.ITvisLockin',\
            'AHT_020_FlirX6541sc_220fps_3kW_B035_j135_30kHz_90PWM_MP_3P_5Hz_L_220601.ITvisLockin'\
            ]
        betasDeg=[0,45,90,135]
        acquisitionPeriods = 1#from itvis container
    
    class LogData:
        saveData = True

    class Tasks:#sequential proc tasks
        register = False #local stabilization
        dynamicTermo = False #sync multi-pulses
        preprocTermo = False
        postprocTermo = True
        procTermo = [Procs.FFT]
        fusedProcs = [FusedProcs.PhaseShift]#all procTermo are processed in groups
        # fusedProcs = []#all procTermo are processed in groups

    class RegisterPars:
        nada=0      
    class DynamicTermoPars:
        nada=0    
    class RegisterPars:
        nada=0
    class DynamicTermoPars:
        nada=0
    class PreprocTermoPars:
        nada=0
    class PostprocTermoPars:
        lowPassGaussKsize = 21
        lowPassGaussSigma = 1
        centerMeanSet = True
    class ProcTermoPars:
        nada=0
    class FusedProcsPars:
        nada=0