import numpy as np
from scipy.fft import fftfreq, fft, ifft, irfft2, rfft2
import pickle
import os
import matplotlib.pyplot as plt

os.chdir("J:\\MTech Project\\DNS Code\\triadIntFull\\")

kmax = 86
N = 3*kmax-2
kx = fftfreq(N, 1./N)
kz = kx[:(N//2+1)].copy()
kz[-1] *= -1
K = np.array(np.meshgrid(kx, kx, kz, indexing='ij'), dtype=np.int16)
Kmag = np.sqrt(np.sum(K.astype(np.int32)**2, 0, dtype=np.int32))
# Kmag_flat = Kmag.flatten()

indList = [[]]        #indList[i] saves all the indices of K array whose magnitude falls in the vicinity of i (i+=.5).
for i in range(kmax):
    indList.append([])      #indList now has kmax+1 empty lists. Fill these with kmag = 0 to kmax

#Trim the k storage matrix by reducing the thickness of shell successively. The no. of wavenumbers increase at the rate of k^2. So, reduce shell thickness by that rate to get nearly constant no. of wavenumbers at each magnitude.
trimValue = 1000   #This is the approximate no. of wavenumbers that we want in large shells.
shellThick = np.empty(kmax+1)
for i in range(kmax+1): shellThick[i] = min(0.5,15*trimValue/2/np.pi/(i+int(i==0))**2)
shellThick[:] = 0.5         #For full matrix

N1 = N//2+1
N2 = N*N1
indexRange = np.append(np.arange(kmax+1),np.arange(-kmax,0))
for i in indexRange:
    for j in indexRange:
        for k in range(kmax+1):         #Only positive z-components in wavenumbers
            kmag = Kmag[i,j,k]
            if kmag<(kmax+0.5):
                temp = round(kmag)
                dist = abs(kmag-temp)
                if dist>shellThick[temp]: continue
                # temp2 = i*N2+j*N1+k
                # indList[temp].append((dist,temp2))
                indList[temp].append((np.array([i,j,k],dtype=np.int16)))         #For full matrix
                # Flatten also works.

# #Now, we have a trimmed store. But, we can further improve this by limiting the number of indexes stored at each magnitude by the trimValue. for this, we need to sort these by the distance from magnitude.
# for j in range(len(indList)):
#     list = indList[j]
#     if len(list)>trimValue:
#         temp = sorted(list)
#         temp2 = [i[1] for i in temp[:trimValue]]
#         i = trimValue
#         while (temp[i-1][0]==temp[i][0]):
#             temp2.append(temp[i][1])
#             i+=1
#             if i==len(list): break
#         indList[j]=temp2
#     else:
#         list = indList[j]
#         indList[j] = [i[1] for i in list]


filename = "kmag_file_"+str(N)+"_expandedIndex.bin"
# filename = "kmag_file_"+str(N)+"_trim"+str(trimValue)+".bin"
f = open(filename,"wb")
pickle.dump(indList,f)
f.close()

#Find max. magnitude of

##Open this file & test
f = open(filename,"rb")
indList2=pickle.load(f)
indCount = np.array([len(item) for item in indList2])
plt.figure()
plt.plot(indCount)
plt.show()