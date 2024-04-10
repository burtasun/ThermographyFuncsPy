import commonImports
import thermo_io

def iniEnv():
    thermo_io.iniEnv()

if __name__=='__main__':
    print('''
--------------------------------------------
    Thermographic processing algorithms
--------------------------------------------
''')
    iniEnv()
    fn=r'C:\Users\benat\source\repos\ThermographyFuncsPy\AHT020__Alpha0_H_Norm.ITvisLockin'
    termos = thermo_io.loadThermo(fn,1,1)
    for t in termos:
        print(f'{str(t.shape)}')