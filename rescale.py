import numpy as np
import afqstensorutils as atu

dat = np.loadtxt("surf.csv",skiprows=1,delimiter=",")
with file("surf.csv") as f:
    header = f.readline()[:-1]
    
S = atu.Scaler(dat)

np.savetxt("surf_scaled.csv",S.scaled_data,delimiter=", ",header=header,comments="")
