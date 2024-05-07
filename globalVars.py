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
class ParamsClass:
    outputDir = '.\\out\\Screw'
    outputDir = '.\\out\\Billet'
    threshOF = float
    #TODO como preproc
    tempFilterRadius = 6

    class InputPars:

        dirFiles = r'D:\Datasets\Termografias\DynamicMulti_Paper\Records\2023_11\Tornillos\Tornillo1'
        fns = ['Tornillo1_Induction_20231122_205516_v375_flock5_n1_nTot288_2.tiff']
        offFrames = 1#start pulse / offset no sync

        dirFiles = r'D:\Datasets\Termografias\DynamicMulti_Paper\Records\2023_11\Palanquillas\90x250'
        fns = ['Palanquilla_90x250_Induction_20231122_203841_v375_flock5_n1_nTot288_2.tiff']
        offFrames = 3#start pulse / offset no sync


        # dirFiles =r'D:\Datasets\Termografias\Phase-Shifted_Induction_Thermography\POD\SetAuthentic\Cache'
        # fns=[\
        #     'AHT_020_FlirX6541sc_220fps_3kW_B035_j000_30kHz_100PWM_MP_3P_5Hz_L_220601.ITvisLockin',\
        #     'AHT_020_FlirX6541sc_220fps_3kW_B035_j045_30kHz_90PWM_MP_3P_5Hz_L_220601.ITvisLockin',\
        #     'AHT_020_FlirX6541sc_220fps_3kW_B035_j090_30kHz_100PWM_MP_3P_5Hz_L_220601.ITvisLockin',\
        #     'AHT_020_FlirX6541sc_220fps_3kW_B035_j135_30kHz_90PWM_MP_3P_5Hz_L_220601.ITvisLockin'\
        #     ]
        betasDeg=[0,45,90,135]

        #from itvis container / TODO extraer de contenedor
        acquisitionPeriods = 3
        frameRate = 300
        excFreq = 5.0
        dutyRatio = 0.5

    Input = InputPars()
    
    class LogData:
        saveData = True
        saveStitched = False
    
    class TasksClass:#sequential proc tasks
        preprocTermo = True
        register = True #local stabilization
        dynamicTermo = True #sync multi-pulses
        postprocTermo = True
        procTermo = [Procs.FFT, Procs.HOS, Procs.PCA]
        fusedProcs = []#all procTermo are processed in groups
        fusedProcs = [FusedProcs.PhaseShift]#all procTermo are processed in groups
    Tasks = TasksClass()

    class DynamicTermo:
        nPulsesPerDirection = 1
        nPulsesPreserve = 3
        nPulseSkip=0
        nDirections = 4
        normAmplitudes = True
        RoiCrop = [36,-20,343,2717+343]
    DynamicTermoPars=DynamicTermo()

    class PreprocTermo:
        averagePulses = True
    PreprocTermoPars = PreprocTermo()
    
    class PostprocTermo:
        lowPassGaussKsize = 41
        centerMeanSet = True
    PostprocTermoPars = PostprocTermo()

#Params

global Params
Params = ParamsClass
def initGlobalVars():
    Params = ParamsClass()
    