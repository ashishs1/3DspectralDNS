###------Calculating triadic interactions on k-containing processes------###

from time import time
import numpy as np
from scipy.fft import fftfreq, fft, ifft, irfft2, rfft2
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_processes2 = comm.Get_size()
rank = comm.Get_rank()
import h5py
import os
import sys
import pickle

##Locate files to be read
simName = "G:\\MTech Project\\sim8\\"      #  <---------- LOCATION OF THE SIMULATION FOLDER
snapFile = simName+"snapshot_02000.hdf5"
triadDir = "G:\\MTech Project\\sim8\\triadic_interactions\\"
kpqFileName = triadDir+"kpq_256_8_fill200_r"+str(rank)+".bin"
os.chdir(triadDir)

fillLim = 200
kmax = 86
N = 3*kmax-2
num_processes = 8
Np = N // num_processes
sys.stderr = open("error.log", 'w', buffering=1 )
sys.stdout = open("out.log", 'w', buffering=1 )

##Parallel FFT functions, taken from spectralDNS code (author:Mikael Mortensen <mikaem@math.uio.no>)
Uc_hat = np.empty((N, Np, N//2+1), dtype=complex)
Uc_hatT = np.empty((Np, N, N//2+1), dtype=complex)
def fftn_mpi(u, fu):
    Uc_hatT[:] = rfft2(u, axes=(1, 2))
    fu[:] = np.rollaxis(Uc_hatT.reshape(Np, num_processes, Np, N//2+1), 1).reshape(fu.shape)
    comm.Alltoall(MPI.IN_PLACE, [fu, MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis=0)
    return fu

def ifftn_mpi(fu, u):
    Uc_hat[:] = ifft(fu, axis=0)
    comm.Alltoall(MPI.IN_PLACE, [Uc_hat, MPI.DOUBLE_COMPLEX])
    Uc_hatT[:] = np.rollaxis(Uc_hat.reshape((num_processes, Np, Np, N//2+1)), 1).reshape(Uc_hatT.shape)
    u[:] = irfft2(Uc_hatT, axes=(1, 2))
    return u
##----------------------------------------------------------------##

##Read snapshot file and kpq file & prepare blank output matrix
U_hat = np.zeros((3, N, Np, N//2+1), dtype=complex)
P_hat = np.zeros((N, Np, N//2+1), dtype=complex)
rho_hat = np.zeros((N, Np, N//2+1), dtype=complex)
U = np.empty((3, Np, N, N))        #Dimensional field is U*uc
P = np.empty((Np, N, N))           #Dimensional field is P*(gamma*pc)
rho = np.empty((Np, N, N))         #Dimensional field is rho*rhoc
f = h5py.File(snapFile, 'r')
if rank*Np<(kmax+1):
    temp3 = min(Np,kmax+1-rank*Np)
    U_hat[:,:kmax+1,:temp3,:kmax+1] = f['U_hat'][:,:kmax+1,rank*Np:rank*Np+temp3,:]
    U_hat[:,-kmax:,:temp3,:kmax+1] = f['U_hat'][:,-kmax:,rank*Np:rank*Np+temp3,:]
    rho_hat[:kmax+1,:temp3,:kmax+1] = f['rho_hat'][:kmax+1,rank*Np:rank*Np+temp3,:]
    rho_hat[-kmax:,:temp3,:kmax+1] = f['rho_hat'][-kmax:,rank*Np:rank*Np+temp3,:]
    P_hat[:kmax+1,:temp3,:kmax+1] = f['P_hat'][:kmax+1,rank*Np:rank*Np+temp3,:]
    P_hat[-kmax:,:temp3,:kmax+1] = f['P_hat'][-kmax:,rank*Np:rank*Np+temp3,:]
    del temp3
if (rank+1)*Np>(N-kmax):
    temp2 = 2*kmax + 1 + (rank+1)*Np - N
    temp3 = max(kmax+1,temp2-Np)
    temp4 = temp3+Np-temp2
    U_hat[:,:kmax+1,temp4:,:kmax+1] = f['U_hat'][:,:kmax+1,temp3:temp2,:]
    U_hat[:,-kmax:,temp4:,:kmax+1] = f['U_hat'][:,-kmax:,temp3:temp2,:]
    rho_hat[:kmax+1,temp4:,:kmax+1] = f['rho_hat'][:kmax+1,temp3:temp2,:]
    rho_hat[-kmax:,temp4:,:kmax+1] = f['rho_hat'][-kmax:,temp3:temp2,:]
    P_hat[:kmax+1,temp4:,:kmax+1] = f['P_hat'][:kmax+1,temp3:temp2,:]
    P_hat[-kmax:,temp4:,:kmax+1] = f['P_hat'][-kmax:,temp3:temp2,:]
    del temp2,temp3,temp4
f.close()

#Compute initial conditions in the real space
rho[:] = ifftn_mpi(rho_hat, rho)
P[:] = ifftn_mpi(P_hat, P)
for i in range(3): U[i] = ifftn_mpi(U_hat[i], U[i])

#Generate w_hat field
w_hat = np.zeros((3, N, Np, N//2+1), dtype=complex)
for i in range(3): w_hat[i] = fftn_mpi(U[i]*np.sqrt(rho)/1e10,w_hat[i])     #Output will be normalised by 1e-20. IMPORTANT!!

#Read kpq file
kpqData=pickle.load(open(kpqFileName,"rb"))
#Each element in kpqData: (srk,srp,srq,list of np.array([qtype,k,p,q,k_[0],k_[1],k_[2],ki1,p_[0],p_[1],p_[2],pi1,q_[0],q_[1],q_[2]]))
#We have chunks of p-data, which needs to be sent to different k's. These chunks of wp's can be sent from one process to another without tagging, as only 1 chunk exists for a source-destination pair. For each kp pair, we have chunks of q-data. These can be sent to the k-process, by tagging with srp.

#Prepare blank output matrix. Each triadic interaction of triplet kpq at triadicInt[k][p][q]
# triadicInt = []
# for i1 in range(kmax+1):
#     temp1 = []
#     for i2 in range(kmax+1):
#         temp2 = []
#         for i3 in range(kmax+1):
#             temp2.append([])
#         temp1.append(temp2)
#     triadicInt.append(temp1)
# del temp1,temp2
triadicInt = []
for k in range(kmax+1):
    temp2 = []
    for p in range(k+1,kmax+1):
        qN = p+k+1-max(0,p-k-2)
        temp2.append(np.zeros((qN,3)))  #triplet kpq values can be found at triadicInt[k][p-k-1][q-(p-k-1)-(p==k+1)]
    triadicInt.append(temp2)
del temp2
garbage = []

##Make send objects of p & q
for dataPacket in kpqData:
    #Nothing to send if rank is k-holder
    if rank==dataPacket[0]: continue
    #Make send objects if this process is a p-source
    if rank==dataPacket[1]:
        temp2 = dataPacket[3]
        if len(temp2)==1: temp2 = [temp2]      #To make it iterable
        data = np.array([w_hat[:,tmp[8],tmp[11],tmp[10]] for [tmp] in temp2])        #Bracket on tmp, because temp2 has 1 element, which contains the relevant np array.
        comm.Isend(data, dest=dataPacket[0], tag=dataPacket[2])     #Tag would have not been required if loop was continued for next few iterations, till p remains same. But this is fine for now.
    #Make send objects if this process is a q-source
    if rank==dataPacket[2]:
        data = []
        temp2 = dataPacket[3]
        if len(temp2)==1: temp2 = [temp2]      #To make it iterable
        for [tmp] in temp2:
            if tmp[0]==0: data.append(np.dot((tmp[8:11] - tmp[4:7]).astype(np.complex128), U_hat[:,tmp[12],tmp[13],tmp[14]].conj()))
            elif tmp[1]==1: data.append(np.dot((tmp[8:11] + tmp[4:7]).astype(np.complex128), U_hat[:,tmp[12],tmp[13],tmp[14]].conj()))
            else: data.append(np.dot((tmp[8:11] + tmp[4:7]).astype(np.complex128), U_hat[:,tmp[12],tmp[13],tmp[14]]))
        data = np.array(data)
        comm.Isend(data, dest=dataPacket[0], tag=dataPacket[1])

##Make receive objects and calculate
for dataPacket in kpqData:
    if rank==dataPacket[0]:
        twait=0         #Stores the time taken in waiting for communication
        temp2 = dataPacket[3]
        w_hatk = np.empty((len(temp2),3),dtype = complex)
        i=0
        if len(temp2)==1: temp2 = [temp2]      #To make it iterable
        for [tmp] in temp2:
            if tmp[0]==0: w_hatk[i] = w_hat[:,tmp[4],tmp[7],tmp[6]]
            else: w_hatk[i] = w_hat[:,tmp[4],tmp[7],tmp[6]].conj()
            i+=1
        #Receive p-data from other processes (or same process).
        if rank==dataPacket[1]:
            wpwk = np.array([np.dot(w_hatk[i],w_hat[:,temp2[i][0,8],temp2[i][0,11],temp2[i][0,10]]) for i in range(len(temp2))])
        else:
            w_hatp = np.empty(w_hatk.shape,dtype=complex)
            req = comm.Irecv(w_hatp,source=dataPacket[1], tag=dataPacket[2])
            tstart = time()
            req.Wait()
            twait = time()-tstart
            wpwk = np.array([np.dot(w_hatk[i],w_hatp[i]) for i in range(len(w_hatk))])
        #Receive q-data from other processes (or same process).
        if rank==dataPacket[2]:
            i=0
            for [tmp] in temp2:
                if tmp[0]==0: wpwk[i] *= np.dot((tmp[8:11] - tmp[4:7]).astype(np.complex128), U_hat[:,tmp[12],tmp[13],tmp[14]].conj())
                elif tmp[1]==1: wpwk[i] *= np.dot((tmp[8:11] + tmp[4:7]).astype(np.complex128), U_hat[:,tmp[12],tmp[13],tmp[14]].conj())
                else: wpwk[i] *= np.dot((tmp[8:11] + tmp[4:7]).astype(np.complex128), U_hat[:,tmp[12],tmp[13],tmp[14]])
                i+=1
        else:
            data = np.empty(len(wpwk),dtype=np.complex128)
            req = comm.Irecv(data, source=dataPacket[2], tag=dataPacket[1])
            tstart = time()
            req.Wait()
            twait = time()-tstart
            wpwk *= data
        #Fill the calculated quantity into triadicInt array
        # i = 0
        # for [tmp] in temp2:
        #     triadicInt[tmp[1]][tmp[2]][tmp[3]].append(np.imag(wpwk[i]))
        #     if rank==2 and np.imag(wpwk[i])==0: print("Im(wpwk) with zero at i=+"str(i)+":\n(k,p,q) = (",str(tmp[1]),str(tmp[2]),str(tmp[3])+") ",wpwk[i:min(10,len(wpwk)-1)])
        #     i+=1
        i = 0
        for [tmp] in temp2:
            temp3 = tmp[2]-tmp[1]-1     #triadicInt[k][p-k-1][q-(p-k-1)-(p==k+1)]
            temp4 = triadicInt[tmp[1]][temp3][tmp[3]-temp3-int(temp3==0)]
            temp5 = np.imag(wpwk[i])
            if rank==2 and temp5==0: print("Im(wpwk) with zero at i="+str(i)+":\n(k,p,q) = ("+str(tmp[1]),str(tmp[2]),str(tmp[3])+") ",wpwk[i])
            temp4[0] += temp5
            temp4[1] += temp5**2
            temp4[2] += 1
            i+=1

# ##Bring all triadic interaction quantities to one process
# triadicInt2 = np.zeros((kmax+1,kmax+1,kmax+1,3))    #3 values to be stored: sum(quantities), sum(squares), no. of quantites
# for i1 in range(kmax+1):
#     for i2 in range(kmax+1):
#         for i3 in range(kmax+1):
#             tmp = triadicInt[i1][i2][i3]
#             nTriad = len(tmp)
#             if nTriad==0: continue
#             [sumT,sqT] = [0,0]
#             for i4 in tmp:
#                 sumT += i4
#                 sqT += i4**2
#             triadicInt2[i1,i2,i3,0] = sumT
#             triadicInt2[i1,i2,i3,1] = sqT
#             # if sqT!=0: print(rank,i1,i2,i3,sqT,nTriad)
#             triadicInt2[i1,i2,i3,2] = nTriad
#             # if nTriad!=0: print("Triads found",rank,i1,i2,i3,nTriad)

def allreduce_mpi_numpy(senddata):
    sumdata = np.empty(senddata.shape, dtype=np.float64)
    comm.Allreduce([senddata, MPI.DOUBLE], [sumdata, MPI.DOUBLE], op=MPI.SUM)
    return sumdata

triadicInt2 = []
for k in range(len(triadicInt)):
    temp2 = []
    for p in range(kmax+1-k-1):
        temp2.append(allreduce_mpi_numpy(triadicInt[k][p]))
    triadicInt2.append(temp2)
del temp2
if rank==num_processes//2:
    triad_file = triadDir+"triadicIntSingleTimstp_N"+str(N)+"_fill"+str(fillLim)+".bin"
    os.makedirs(os.path.dirname(triad_file), exist_ok=True)
    triadfile = open(triad_file, "wb")
    pickle.dump(triadicInt2,triadfile)

