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