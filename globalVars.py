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
    #TODO como preproc
    tempFilterRadius = 12

    class Input:
        dirFiles = r'D:\Datasets\Termografias\DynamicMulti_Paper\Records\2023_11\Palanquillas\90x250'
        fns = ['Palanquilla_90x250_Induction_20231122_203841_v375_flock5_n1_nTot288_2.tiff']

        # dirFiles = r'D:\Datasets\Termografias\DynamicMulti_Paper\Records\2023_11\Tornillos\Tornillo1'
        # fns = ['Tornillo1_Induction_20231122_205516_v375_flock5_n1_nTot288_2.ItvisLockin']

        # dirFiles =r'D:\Datasets\Termografias\Phase-Shifted_Induction_Thermography\POD\SetAuthentic\Cache'
        # fns=[\
        #     'AHT_020_FlirX6541sc_220fps_3kW_B035_j000_30kHz_100PWM_MP_3P_5Hz_L_220601.ITvisLockin',\
        #     'AHT_020_FlirX6541sc_220fps_3kW_B035_j045_30kHz_90PWM_MP_3P_5Hz_L_220601.ITvisLockin',\
        #     'AHT_020_FlirX6541sc_220fps_3kW_B035_j090_30kHz_100PWM_MP_3P_5Hz_L_220601.ITvisLockin',\
        #     'AHT_020_FlirX6541sc_220fps_3kW_B035_j135_30kHz_90PWM_MP_3P_5Hz_L_220601.ITvisLockin'\
        #     ]
        # betasDeg=[0,45,90,135]
        betasDeg=[0]
        acquisitionPeriods = 1#from itvis container
        

        #TODO extraer de contenedor
        offFrames = 1#start pulse / offset no sync
        frameRate = 300
        acqPeriods = 144
        excFreq = 5.0

    class LogData:
        saveData = True
        saveStitched = True
    class Tasks:#sequential proc tasks
        preprocTermo = True
        register = True #local stabilization
        dynamicTermo = True #sync multi-pulses
        postprocTermo = False
        procTermo = [Procs.FFT]
        # fusedProcs = [FusedProcs.PhaseShift]#all procTermo are processed in groups
        fusedProcs = []#all procTermo are processed in groups

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