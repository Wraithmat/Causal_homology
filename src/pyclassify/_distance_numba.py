from numba.pycc import CC

cc = CC('_distance_numba_mod')

@cc.export('_distance_numba_func','f8(f8[:], f8[:])')
def _distance_numba(point1,point2):
    distance=0.0
    for i in range(len(point1)):
        distance+=(point1[i]-point2[i])**2
    return distance**(0.5)

if __name__=="__main__":
    cc.compile()

