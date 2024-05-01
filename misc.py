import time

class helperProfile:
    def __init__(self) -> None:
        self.t = time.time()
        self.profileDict = dict[str,float]()
        self.count = 0
    
    def ticTocProfile(self, name:str):
        before = self.t
        now = time.time()
        self.t = now
        self.profileDict[f'{self.count}_{name}']=now-before
        self.count+=1
    
    def printProfile(self):
        print("Profilling prog")
        for k in sorted(self.profileDict.keys()):
            print(f'{k} {self.profileDict[k]:.3f}s')