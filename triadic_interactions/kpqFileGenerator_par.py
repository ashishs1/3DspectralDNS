##Script to generate & save limited no. of randomly selected indices of k, p and q wavenumbers that form a triad in Fourier space.

import numpy as np
from scipy.fft import fftfreq, fft, ifft, irfft2, rfft2
import pickle
import os
from time import time
from tqdm import tqdm
import sys
import random

##Locate the kmag file and set some basic parameters.
os.chdir("G:\\MTech Project\\sim8\\triadic_interactions\\")
kmax = 86
N = 3*kmax-2
num_procs = 8           #Set this correctly to enable correct rank saving in kpq file.
fillLim = 200            #Limit on no. of wavenumber triads to be saved for each kpq triplet/considered for computation. This will have to be increased if standard deviation comes out to be too high.
filename = "kmag_file_256_expandedIndex.bin"    #Problem with using trim kmag file is that many kpq points are left unfilled
Np = N//num_procs
f = open(filename,"rb")
indList = pickle.load(f)
f.close()
# sys.stdout = open("out.log", 'w', buffering=1)
# sys.stdout = sys.__stdout__         #To bring output back to shell
##******---------------------------*******##

# kpqCount = np.zeros((kmax+1,kmax+1,kmax+1),dtype=np.int64)      #To find the total no. of triplets at each point. This needs to be multiplied by 2 to get actual count (in computational space, only half of the q's are found).
qfilmax = min(1,fillLim)+min(6,fillLim)+min(13,fillLim)+min(19,fillLim)+min(39,fillLim)+min(55,fillLim)+fillLim*(kmax-5) #This is for N=256 & fillLim<=72.

##Define functions
def KtoIndex(q):
    if q[0]<0: q[0]+=N
    if q[1]<0: q[1]+=N
    if q[2]<0:
        q[2]+=N      #This should not be required, as it will not be available
        raise ValueError("z-component should not be negative!")

N1 = N//2+1         #No. of wavenumber z-components/indices
N2 = N*N1           #Use these 2 nos. only to reduce the size of saved file. But beware, this will add computation time.
kind2 = np.empty(3,dtype=np.int16)
pind2 = np.empty(3,dtype=np.int16)
def addQ(kpqArray,k,p,k_,ki1,p_,srk,qfil,q_,qtype):
    qmag = np.sqrt(sum(q_**2))
    q = round(qmag)
    if qmag>(kmax+0.5): return None        #Only store q if it is smaller than kmax.
    # kpqCount[k,p,q] += 1            #Count the no. of k,p,q triplets
    if len(kpqArray[p][q])==fillLim: return None      #Can also use qfil[q] here.
    # KtoIndex(q_)            #Now q_ stores the indices of q wavenumber vector
    tmp = (p_[1] + (p_[1]<0)*N)
    srp = tmp//Np        #Source rank of p wavenumber
    pi1 = tmp%Np
    tmp = (q_[1] + (q_[1]<0)*N)
    srq = tmp//Np
    q_[1] = tmp%Np      #q wavenumber is not required. Only second index is required. So, effectively, only q indices are being stored, not the q wavenumber. IMPORTANT!!
    rankNumber1 = qtype*num_procs + srk
    rankNumber2 = srp*num_procs + srq
    kpqArray[p][q].append(np.array([rankNumber1,rankNumber2,k_[0],k_[1],k_[2],ki1,p_[0],p_[1],p_[2],pi1,q_[0],q_[1],q_[2]],dtype=np.int8))       #This is what will be saved for each kpq triad. Change dtype if N>256.
    qfil[q] += 1

def fillKPQ(k):
    kpqArray = []       #This array is supposed to store indices for all valid [p][q] pairs.
    for i in range(kmax+1):
        temp2 = []
        for j in range(kmax+1):
            temp2.append([])
        kpqArray.append(temp2)
    for p in range(k+1,kmax+1):
        qfil = np.zeros(kmax+1)         #Counts the no. of triplets filled for each qmag from 1 to kmax. Index filling to be stopped as soon as all numbers reach fillLim.
        #Max. qmag will be (p+0.5)+(k+0.5)=p+k+1, but this might not be found. Min. qmag will be (p-0.5)-(k+0.5)=p-k-1 (or 0), but this also might not be found. So, qmag will range from max(0,p-k-1) to p+k+1. Also, for qmag=0, only 1 q_ is available. If p>=k+1, this won't be filled. So qmag range should be counted as max(0,p-k-2)+1 to p+k+1. A similar problem will exist for k=0, but that is not handled here.
        qfilmax = fillLim*(p+k+1-max(0,p-k-2))       #Assumes p>k
        kinds = indList[k]
        pinds = indList[p]
        t1 = time()
        sampleSize = min(len(pinds),int(qfilmax*100/len(kinds)+1))
        kpinds = [(k_,p_) for k_ in kinds for p_ in random.sample(pinds,sampleSize)]
        random.shuffle(kpinds)
        for (k_,p_) in kpinds:
            tmp = (k_[1] + (k_[1]<0)*N)
            srk = tmp//Np        #Source rank of k wavenumber
            ki1 = tmp%Np            #Second index of K for k_ inside a rank. Other indices can be found from wavenumber value itself, but second index can't be, so it needs to be stored.
            kk = k_[2]
            pk = p_[2]
            addQ(kpqArray,k,p,k_,ki1,p_,srk,qfil,k_+p_,0)
            if pk>kk: addQ(kpqArray,k,p,k_,ki1,p_,srk,qfil,-k_+p_,1)
            else: addQ(kpqArray,k,p,k_,ki1,p_,srk,qfil,k_-p_,2)
            if sum(qfil)==qfilmax: break
    del kpinds,kinds,pinds
    #Unable to dump from within function. Joblib gives pickling error.
    return kpqArray

##Run the above function to generate the kpq array
# # for k in range(kmax+1):
# #     tstart = time()
# #     fillKPQ2(k)
# #     print(time()-tstart)
from joblib import Parallel, delayed
tstart = time()
res = Parallel(n_jobs=16)(delayed(fillKPQ)(ki) for ki in tqdm(np.arange(kmax+1)))
tend = time()
print(str(int((tend-tstart)//3600))+":"+str(int((tend-tstart)%3600//60))+":"+str(int((tend-tstart)%60)),"taken for complete computation.")

##Try to add missing kpq triplets from whatever is available from smaller k ones
#The above method leads to some unfilled triplets, where q~ is small (close to k-p-1). For this, we can utilise the already found triplets where k is small, by simply interchangeing k&q wavenumbers. The qtype changes though. qtype changes as follows: 0==>2, 1==>1, 2==>0. This fact will be utilised here:
for k in range(1,kmax+1):
    for p in range(k+1,kmax+1):
        for q in range(kmax+1):
            if q>=(max(0,p-k-2)+1) and q<=(k+p+1):
                if len(res[k][p][q])<fillLim:
                    if q>k: continue       #This means that filling might not be possible
                    #Now, we need to utilise res[q][p][k] to make new triplets
                    temp = res[q][p][k]
                    #temp=[rankNumber1,rankNumber2,k_[0],k_[1],k_[2],ki1,p_[0],p_[1],p_[2],pi1,q_[0],q_[1],q_[2]]
                    for i4 in temp:
                        #Check if this already exists in res[k][p][q]
                        exists = False
                        for i5 in res[k][p][q]:
                            if all(i5[2:5]==i4[10:]) and all(i5[6:10]==i4[6:10]) and all(i5[10:]==i4[2:5]): exists = True
                        if exists: continue
                        #Check if res[k][p][q] is completely filled
                        if len(res[k][p][q])==fillLim: break
                        #Find new quantities now.
                        [r1, r2] = i4[:2]
                        qtype = 2-(r1//num_procs)       #new
                        srq = r1%num_procs          #new
                        srp = r2//num_procs
                        srk = r2%num_procs          #new
                        [r1, r2] = [qtype*num_procs+srq, srp*num_procs+srk]     #new
                        ki1 = (i4[11] + (i4[11]<0)*N)%Np    #new
                        res[k][p][q].append(np.array([r1,r2,i4[10],i4[11],i4[12],ki1,i4[6],i4[7],i4[8],i4[9],i4[2],i4[3],i4[4]],dtype=np.int8))

##Dump the useful data in proper manner into a suitable file
#Each element in kpqData: (srk,srp,srq,np.array([qtype,k,p,q,k_[0],k_[1],k_[2],ki1,p_[0],p_[1],p_[2],pi1,q_[0],q_[1],q_[2]]))
dt = [('srk','i1'),('srp','i1'),('srq','i1'), ('data','(1,15)i1')]  #Dtype to store as above
kpqData = []
for i1 in range(kmax+1):
    for i2 in range(i1+1,kmax+1):
        for i3 in range(kmax+1):
            temp = res[i1][i2][i3]
            for i4 in range(len(temp)):
                [r1, r2] = temp[i4][:2]
                qtype = r1//num_procs
                srk = r1%num_procs
                srp = r2//num_procs
                srq = r2%num_procs
                kpqData.append((srk,srp,srq,np.append([qtype,i1,i2,i3],temp[i4][2:]).astype('i1')))

del temp
kpqData1 = np.array(kpqData,dtype=dt)
kpqData1 = np.sort(kpqData1,order=['srk', 'srp', 'srq'])    #Sort w.r.t. srk first, then w.r.t. srp for identical srk's, then w.r.t. srq for identical srp's.
#Testing np.sort:-
# test = []
# for i in range(100):
#     tmp = np.random.random((16))*64//8
#     test.append((tmp[0],tmp[1],tmp[2],tmp[3:]))
# test = np.array(test,dtype = dt)
# test = np.sort(test,order=['srp', 'srk', 'srq'])
kpqData = []
[srk, srp, srq] = [-1,-1,-1]
tmp = []
for element in kpqData1:
    [srk2,srp2,srq2] = [element[0],element[1],element[2]]
    if srk2==srk and srp2==srp and srq2==srq:
        tmp.append(element[3])
    else:
        kpqData.append((srk,srp,srq,tmp))
        [srk,srp,srq] = [srk2,srp2,srq2]
        tmp = [element[3]]
del kpqData1,tmp
# del kpqData[0]      #Because first element is garbage ([-1,-1,-1],[]).
saveDir = "G:\\MTech Project\\sim8\\triadic_interactions\\"
#Save different file for each processor
for i in range(num_procs):
    # kpqData_rank = kpqData.copy()
    kpqData_rank = [element for element in kpqData if (i==element[0]) or (i==element[1]) or (i==element[2])]
    filename = saveDir+"kpq_"+str(N)+"_"+str(num_procs)+"_fill"+str(fillLim)+"_r"+str(i)+".bin"
    f = open(filename,"wb")
    pickle.dump(kpqData_rank,f)
    f.close()

##Analyse kpqfile
import matplotlib.pyplot as plt
lengths = [len(kpqData[i][3]) for i in range(len(kpqData))]
plt.plot(lengths)
plt.show()
tmp = np.array([len(res[i1][i2][i3]) for i1 in range(kmax+1) for i2 in range(kmax+1) for i3 in range(kmax+1)])
n1 = np.sum((tmp==0) + (tmp==fillLim))       #No. of kpq triplets that were either completely filled or didn't form any possible triad
n2 = len(tmp)           #Total no. of kpq triplets
n3 = n2 - n1            #No. of kpq triplets that did start filling but weren't completed
if n3!=0:
    print("WARNING!! All triads not filled completely!")
    #Find where triads are missing
    for k in range(1,kmax+1):
        for p in range(k,kmax+1):
            for q in range(kmax+1):
                if q>=(max(0,p-k-2)+1) and q<=(k+p+1):
                    if len(res[k][p][q])<fillLim: print(str(k),str(p),str(q),len(res[k][p][q]))
                # if q>(max(0,p-k-2)+1) and q<(k+p+1):
                #     if len(res[k][p][q])<fillLim: print(str(k),str(p),str(q))
#The above shows that some triads are incompletely filled due to lack of enough wavenumbers, and some due to random choices. All of these are on the boundary of q limits.
##TO DO: Fill the incompletely filled points by rearranging the indices found for other completely filled points.
#Apart from the triplets 1-1-3 (having 12) & 2-3-6 (having 24), all triplets can have 25 sets of indices.


