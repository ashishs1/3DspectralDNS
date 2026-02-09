##Restart to calculate triadic interactions##

from time import time
wt0 = time()
from numpy import *
from numpy import max as maxi
from numpy import min as mini
from scipy.fft import fftfreq, fft, ifft, irfft2, rfft2
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
import h5py
import os
import sys
import pickle

##Checklist before submitting:
#(1) Check simName folder path.
simName = "G:\\MTech Project\\sim8\\"      #  <---------- LOCATION OF THE SIMULATION FOLDER
#(2) Ensure that kpq array file(s) are in the directory
triadDir = "G:\\MTech Project\\sim8\\triadic_interactions\\"
kpqFileName = triadDir+"kpq_256_8_fill200_r"+str(rank)+".bin"
fillLim = 200
kpqData=pickle.load(open(kpqFileName,"rb"))         #Read kpq triplets file

#(3) Ensure that the timeSeries file is empty. (Back up old one, or change name below.)
if rank==0:
    time_series_file = simName+"timeSeries.bin"
    os.makedirs(os.path.dirname(time_series_file), exist_ok=True)
    tsfile = open(time_series_file, "wb")
    del time_series_file
##--------------------------##
sys.stderr = open("error.log", 'w', buffering=1 )
sys.stdout = open("out.log", 'w', buffering=1 )

kmax = 86
N = 3*kmax-2        #Try to keep this as a power of 2.
#The maximum wavenumber considered for calculating Fourier coeffs of non-linear terms is N//2. This should be >= 3/2*kmax-1. Hence, N>=3*kmax-2.

##Characteristic parameters
# [l, rhoc, pc, muc, kc, gamma, R] = [0.01, 1.2, 1e5, 1.81e-5, 2.4e-2, 1.4, 287]         #All values in SI units.
[l, rhoc, pc, muc, kc, gamma, R] = [0.0005, 0.1664, 1e5, 1.96e-5*1.1, 15.34e-2, 1.67, 2079]         #Indicative values of He@NTP (SI units: [m, kg/m3, Pa, kg/ms, W/mdegC, dimless, J/kgK])
urms = kmax**(4/3)*muc/rhoc/l   #Intended urms in SI units (Since n/L ~ 1/kmax)
cc = sqrt(gamma*pc/rhoc)        #Characteristic speed of sound, SI units
Mc = urms/cc                  #For TGV, intitially, urms~uc/2. So, for Mt~0.03, Mc~0.06.
[uc, Tc] = [Mc*cc, pc/rhoc/R]
Rec = rhoc*uc*l/muc
Prc = muc*gamma*R/(gamma-1)/kc
if rank==0:
    if Prc<0.95*.75 or Prc>1.05*.75: print("WARNING! Prandtl number is",round(Prc,3),"but should be close to 0.75!")
    Mt = urms/cc
    Rlambda = kmax**(2/3)
    print("Intended u' =",urms,"\t\tIntended Mt =",Mt,"\t\tIntended Rl =",Rlambda)  #\u03bb gives error
    del Mt,Rlambda

Np = N // num_processes
X = mgrid[rank*Np:(rank+1)*Np, :N, :N].astype(float)*2*pi/N
U = empty((3, Np, N, N))        #Dimensional field is U*uc
P = empty((Np, N, N))           #Dimensional field is P*(gamma*pc)
rho = empty((Np, N, N))         #Dimensional field is rho*rhoc
Uc_hat = empty((N, Np, N//2+1), dtype=complex)
Uc_hatT = empty((Np, N, N//2+1), dtype=complex)
kx = fftfreq(N, 1./N)
kz = kx[:(N//2+1)].copy()
kz[-1] *= -1
K = array(meshgrid(kx, kx[rank*Np:(rank+1)*Np], kz, indexing='ij'), dtype=int16)
K2 = sum(K.astype(int32)**2, 0, dtype=int32)
# K_over_K2 = K.astype(float) / where(K2 == 0, 1, K2).astype(float)
dealias = array((abs(K[0]) <= kmax)*(abs(K[1]) <= kmax)*(abs(K[2]) <= kmax), dtype=bool)

##Parallel FFT functions, taken from spectralDNS code (author:Mikael Mortensen <mikaem@math.uio.no>)
def fftn_mpi(u, fu):
    Uc_hatT[:] = rfft2(u, axes=(1, 2))
    fu[:] = rollaxis(Uc_hatT.reshape(Np, num_processes, Np, N//2+1), 1).reshape(fu.shape)
    comm.Alltoall(MPI.IN_PLACE, [fu, MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis=0)
    return fu

def ifftn_mpi(fu, u):
    Uc_hat[:] = ifft(fu, axis=0)
    comm.Alltoall(MPI.IN_PLACE, [Uc_hat, MPI.DOUBLE_COMPLEX])
    Uc_hatT[:] = rollaxis(Uc_hat.reshape((num_processes, Np, Np, N//2+1)), 1).reshape(Uc_hatT.shape)
    u[:] = irfft2(Uc_hatT, axes=(1, 2))
    return u
##----------------------------------------------------------------##

#########------------INITIAL CONDITIONS STARTED----------###########
v = empty((3, Np, N, N))        #Divergence free base flow (dimensionless)
vkvjk = empty((3, Np, N, N))
#NOTE: To set up any IC with first order solution, populate v, rho, P and vkvjk matrices only.

#(1) Zero initial conditions - No flow or gradients
# v[:] = 0
# rho[:] = rhoc
# P[:] = pc
# vkvjk[:] = 0

#(2) Travelling p-wave
# [pmax,aw] = [0.1,0.5]     #Max. pressure perturbation and Half-width of wave
# # pdash = pmax*cos((X[0]-pi)*pi/2/aw)**2*(abs(X[0]-pi)<=aw)
# pdash = pmax*exp(-9/2*((X[2]-pi)/aw)**2)
# udash = pdash/rhoc/cc
# P[:] = pc + pdash
# U[0] = 0
# U[1] = 0
# U[2] = udash
# rho[:] = rhoc*(1+pdash/gamma/pc)

#(3) Taylor Greene Vortex (A=B=uc, a=b=c=1. So, C=-2uc)
# urms = uc/2                 #Initial urms for TGV. This is slightly different from intended value.
# v[0] = uc/urms*cos(X[0])*sin(X[1])*sin(X[2])
# v[1] = uc/urms*sin(X[0])*cos(X[1])*sin(X[2])
# v[2] = -2*uc/urms*sin(X[0])*sin(X[1])*cos(X[2])
# # P[:] = pc + rhoc*uc**2/8*((cos(2*X[2])/2-1)*(cos(2*X[0])+cos(2*X[1])) + 2*cos(2*X[0])*cos(2*X[1]) - 4*cos(2*X[2]))
# # P[:] = pc
# vkvjk[0] = (uc/urms)**2/4*sin(2*X[0])*(-1 + 2*cos(2*X[1]) - cos(2*X[2]))
# vkvjk[1] = (uc/urms)**2/4*sin(2*X[1])*(2*cos(2*X[0]) - 1 - cos(2*X[2]))
# vkvjk[2] = (uc/urms)**2/2*sin(2*X[2])*(cos(2*X[0]) + cos(2*X[1]) - 2)

#(4) Multi-scale Taylor Greene Vortex (A=B=uc, a=b=c=1. So, C=-2uc)
# ilast = int(2*kmax/3)       #Initialise 2/3rd of the scales (for no reason)
# uc1 = uc/sqrt(sum(exp(array([-.386*(i-1) for i in range(1,ilast+1)]))))
# urms = uc/2                 #Initial urms for TGV. This is slightly different from intended value.
# v[0] = uc1/urms*cos(X[0])*sin(X[1])*sin(X[2])
# v[1] = uc1/urms*sin(X[0])*cos(X[1])*sin(X[2])
# v[2] = -2*uc1/urms*sin(X[0])*sin(X[1])*cos(X[2])
# for i in range(2,ilast+1):
#     v[0] += uc1*exp(-.386*(i-1))/urms*cos(i*X[0])*sin(i*X[1])*sin(i*X[2])
#     v[1] += uc1*exp(-.386*(i-1))/urms*sin(i*X[0])*cos(i*X[1])*sin(i*X[2])
#     v[2] -= 2*uc1*exp(-.386*(i-1))/urms*sin(i*X[0])*sin(i*X[1])*cos(i*X[2])
# # P[:] = pc + rhoc*uc**2/8*((cos(2*X[2])/2-1)*(cos(2*X[0])+cos(2*X[1])) + 2*cos(2*X[0])*cos(2*X[1]) - 4*cos(2*X[2]))
# # P[:] = pc
# vkvjk[0] = (uc/urms)**2/4*sin(2*X[0])*(-1 + 2*cos(2*X[1]) - cos(2*X[2]))
# vkvjk[1] = (uc/urms)**2/4*sin(2*X[1])*(2*cos(2*X[0]) - 1 - cos(2*X[2]))
# vkvjk[2] = (uc/urms)**2/2*sin(2*X[2])*(cos(2*X[0]) + cos(2*X[1]) - 2)

#(5) Multi-scale random velocity field
kp = 5          #Scale with peak energy
A = 16/kp**5/sqrt(pi/2)*(2*pi)**3
Umag = A*K2*exp(-2*K2/kp**2)/pi
phi = random.random((3, N, Np, N//2+1))*(2*pi)        #Random phases of U_hats
temp = where(K == 0, 0, sin(array([phi[1]-phi[2],phi[2]-phi[0],phi[0]-phi[1]])))/where(K == 0, 1, K).astype(float)
temp2 = sum(temp**2, 0)        #Denominator of U_hat magnitudes
temp2 = where(temp2==0, 0, 1.0/temp2)
temp = sqrt(Umag*temp2)*temp
v_hat = temp*(cos(phi)+1j*sin(phi))*N**3        #Multiplied by N^3 because of backward normalisation of scipy fft
del Umag, phi, temp, temp2
for i in range(3):
    v[i] = ifftn_mpi(v_hat[i]*dealias, v[i])
urms_actual = sum(v**2)/3/N**3 * array([1.0], dtype=float64)
sumdata = empty([1], dtype=float64)
comm.Allreduce([urms_actual, MPI.DOUBLE], [sumdata, MPI.DOUBLE], op=MPI.SUM)
urms_actual = sqrt(sumdata)
urms_actual = urms_actual[0]
vfactor = 1/urms_actual
if rank==0:
    print('Required to increase urms by a factor of',round(vfactor,4))
    print("urms required is",round(urms,2),". Actual dimless urms in IC is", urms_actual,".")
v *= vfactor
v_hat *= vfactor
vjk = empty((Np, N, N))
for j in range(3):
    vkvjk[j] = 0
    for k in range(3):
        vjk = ifftn_mpi(K[k]*v_hat[j], vjk)
        vkvjk[j] += v[k]*vjk

del v_hat, vjk, urms_actual, sumdata, vfactor

##--Solve ICs upto first order approximation as in Ristorcelli (1997)--##
del X
ep2 = gamma*(urms/cc)**2        #Epsilon square
p1_hat = zeros((N, Np, N//2+1), dtype=complex)
vi2_hat = empty((N, Np, N//2+1), dtype=complex)
vivj_hat = empty((N, Np, N//2+1), dtype=complex)
for i in range(3):
    vi2_hat = fftn_mpi(v[i]**2, vi2_hat)
    vivj_hat = fftn_mpi(v[i-2]*v[i-1], vivj_hat)
    p1_hat -= (K[i]**2*vi2_hat + 2*K[i-2]*K[i-1]*vivj_hat)
del vi2_hat, vivj_hat
p1_hat *= dealias/where(K2 == 0, 1, K2).astype(float)
p1 = empty((Np, N, N))
p1 = ifftn_mpi(p1_hat, p1)
rho[:] = (1+ep2*p1/gamma)
P[:] = (1+ep2*p1)/gamma             #Since P is non-dimensionalised w.r.t. gamma*pc
del p1      #Pressure and density terms are found till first order. Now, find velocity.
p1t_hat = zeros((N, Np, N//2+1), dtype=complex)
d_hat = zeros((N, Np, N//2+1), dtype=complex)
vjp1i_hat = empty((N, Np, N//2+1), dtype=complex)
p1i = empty((Np, N, N))
vivkvjk_hat = empty((N, Np, N//2+1), dtype=complex)
for i in range(3):
    p1i = ifftn_mpi(-1j*K[i]*p1_hat, p1i)
    for j in range(3):
        vivkvjk_hat = fftn_mpi(v[i]*vkvjk[j], vivkvjk_hat)
        vjp1i_hat = fftn_mpi(v[j]*p1i, vjp1i_hat)
        if i==j: d_hat -= vjp1i_hat
        p1t_hat += K[i]*K[j]*(vivkvjk_hat + vjp1i_hat)
p1t_hat *= 2*dealias/where(K2 == 0, 1, K2).astype(float)
d_hat -= p1t_hat
d_hat *= (dealias/gamma)
del p1_hat, p1t_hat, vjp1i_hat, p1i, vkvjk, vivkvjk_hat
w = empty((3, Np, N, N))
for i in range(3): w[i] = ifftn_mpi(-1j*K[i]*d_hat/where(K2 == 0, 1, K2).astype(float), w[i])
U[:] = (v[:] + ep2*w[:])*urms/uc        #Non-dimensionalised w.r.t. uc.
del d_hat, w, v

##--Transform initial conditions into the Fourier space - DO NOT COMMENT OUT--##
U_hat = empty((3, N, Np, N//2+1), dtype=complex)
M_hat = empty((3, N, Np, N//2+1), dtype=complex)
P_hat = empty((N, Np, N//2+1), dtype=complex)
rho_hat = empty((N, Np, N//2+1), dtype=complex)
T_hat = empty((N, Np, N//2+1), dtype=complex)       #T field is not computed. T=P/(rho*R)
for i in range(3):
    U_hat[i] = fftn_mpi(U[i], U_hat[i])
    M_hat[i] = fftn_mpi(U[i]*rho, M_hat[i])
rho_hat[:] = fftn_mpi(rho, rho_hat)
P_hat[:] = fftn_mpi(P, P_hat)
T_hat[:] = fftn_mpi((gamma*P)/rho, T_hat)
###-------------------INITIAL CONDITIONS DONE-------------------###

###-------------------DEFINE USEFUL FUNCTIONS-------------------###
temp_hat = empty((N, Np, N//2+1), dtype=complex)    #Temporarily stores rhouuij term in Mom eqns, and 2 pu terms & p diffusion term in energy eqn
temp = empty((Np, N, N))                #Stores pdiff term, abs(sqrt_rho_Ui), Ek
duidxi = empty((3, Np, N, N))           #Also temporarily stores duidxi-1 and dui-1dxi. 9 variables are now reduced to 3.
Sdudx = empty((3,3,3))      #Stores 2nd, 3rd & 4th order moments of du/dx in all directions.
w_hat = zeros((3, N, Np, N//2+1), dtype=complex)        #Additional array to store triadic interaction

#Other useful constants
[N2,N3] = [N**2,N**3]

def de_alias(arr):
    arr[kmax+1:-kmax,:,:] = 0
    arr[:,kmax+1-Np*rank:-kmax+Np*(num_processes-rank),:] = 0
    arr[:,:,kmax+1:] = 0

def allreduce_mpi_real(senddata):
    # -senddata must be a scalar
    # -the scalar values from all processes are sent to root and the sum is returned
    send_data = senddata * array(
        [1.0], dtype=float64
    )  # scalar needs to be in an array
    sumdata = empty([1], dtype=float64)
    comm.Allreduce([send_data, MPI.DOUBLE], [sumdata, MPI.DOUBLE], op=MPI.SUM)
    return sumdata[0]

def allreduce_mpi_numpy(senddata):
    sumdata = empty(senddata.shape, dtype=float64)
    comm.Allreduce([senddata, MPI.DOUBLE], [sumdata, MPI.DOUBLE], op=MPI.SUM)
    return sumdata

def saveParams(U, U_hat, rho, P, t, tstep):
    #Saves Mt, Rl, u'rms, lambda, epsilon, rho_bar, and vel derivative skewness & kurtosis.
    #urms, Mt & uiuj:
    temp2 = sum(U,axis=(1,2,3))/N3
    U_avg = allreduce_mpi_numpy(temp2)              #Spatial average of U. Should be close to zero.
    # temp2 = (sum((U[0]-U_avg[0])**2)+sum((U[1]-U_avg[1])**2)+sum((U[2]-U_avg[2])**2))/(3*N3)
    temp[:] = U[0,:,:,:]**2 + U[1,:,:,:]**2 + U[2,:,:,:]**2
    temp2 = sum(temp)/(3*N3)                        #Assumes U_avg is zero
    urms = sqrt(allreduce_mpi_real(temp2))          #Rms of fluctuating velocity
    temp2 = sum(temp*rho/P)/gamma/(3*N3)
    Mt = uc/cc*sqrt(allreduce_mpi_real(temp2))
    temp2 = array([sum(U[0]*U[1]),sum(U[1]*U[2]),sum(U[0]*U[2])])/N3
    uiuj = allreduce_mpi_numpy(temp2)           #Should be close to zero for isotropy
    #epsilon, and S & K:
    for i in range(3):
        duidxi[i] = ifftn_mpi(1j*K[i]*U_hat[i], duidxi[i])
        temp2 = sum(duidxi[i]**2)/N3
        Sdudx[0,i,i] = allreduce_mpi_real(temp2)
        temp2 = sum(duidxi[i]**3)/N3
        Sdudx[1,i,i] = allreduce_mpi_real(temp2)
        temp2 = sum(duidxi[i]**4)/N3
        Sdudx[2,i,i] = allreduce_mpi_real(temp2)
    temp2 = sum(sum(duidxi,axis=0)**2)/N3           #Spatial average of square of divergence of U
    del2 = allreduce_mpi_real(temp2)                #Mean of divU squared.
    temp3 = 0
    for i in range(3):
        duidxi[0] = ifftn_mpi(1j*K[i-1]*U_hat[i], duidxi[0])        #Read duidxi as duidxi-1
        duidxi[1] = ifftn_mpi(1j*K[i]*U_hat[i-1], duidxi[1])        #Read duidxi as dui-1dxi
        temp2 = sum(duidxi[0]**2)/N3
        temp3 += temp2
        Sdudx[0,i,i-1] = allreduce_mpi_real(temp2)
        temp2 = sum(duidxi[0]**3)/N3
        Sdudx[1,i,i-1] = allreduce_mpi_real(temp2)
        temp2 = sum(duidxi[0]**4)/N3
        Sdudx[2,i,i-1] = allreduce_mpi_real(temp2)
        temp2 = sum(duidxi[1]**2)/N3
        temp3 += temp2
        Sdudx[0,i-1,i] = allreduce_mpi_real(temp2)
        temp2 = sum(duidxi[1]**3)/N3
        Sdudx[1,i-1,i] = allreduce_mpi_real(temp2)
        temp2 = sum(duidxi[1]**4)/N3
        Sdudx[2,i-1,i] = allreduce_mpi_real(temp2)
        temp3 -= 2*sum(duidxi[0]*duidxi[1])/N3
    w2 = allreduce_mpi_real(temp3)              #Spatial average of square of vorticity magnitude
    epsilon = (4/3*del2 + w2)/Rec               #Dimensionless KE dissipation, Kida & Orszog, 1990
    #rho mean & rho_rms:
    temp2 = sum(rho)/N3
    rho_mean = allreduce_mpi_real(temp2)        #Dimensionless
    #Other parameters that can be computed post simulation also:
    # lamda = urms/Rec*sqrt(15/epsilon)
    # Rl = rho_mean*urms**2*sqrt(15/epsilon)
    # nu = muc/rho_mean
    #KE & IE (dimensional):
    temp2 = rhoc*uc**2*l**3*sum(rho*sum(power(U,2),axis=0))/2*(8*pi**3)/N3
    KE = allreduce_mpi_real(temp2)              #Dimensional KE
    temp2 = gamma*pc*l**3*sum(P)/(gamma-1)*(8*pi**3)/N3
    IE = allreduce_mpi_real(temp2)              #Dimensional IE
    del temp2,temp3
    #Print the above parameters to file:
    if rank==0:
        save(tsfile,concatenate(([tstep, t, urms, Mt, epsilon, rho_mean, KE, IE], U_avg.flatten(), Sdudx.flatten(), uiuj.flatten())))
        # tsfile.write(str(tstep)+" "+str(t)+" "+str(U_avg)+" "+str(urms)+" "+str(epsilon)+" "+str(rho_mean))
        tsfile.flush()

K1d = sqrt(K2).flatten()
Nk = int32(sqrt(3)*N/2) + 1
sections = linspace(0.0, sqrt(3)*N/2, Nk)
ixbin = []          #Pre-calculate list of indices in each section. Size of this will be slightly more than 1/4th of that of K1d as int16 is being used, and headers of python list will increase some size.
for i in range(Nk-1): ixbin.append(where((K1d <= sections[i+1]) & (K1d > sections[i]))[0].astype(int32))
spectra_k_plot = zeros((4, Nk - 1))         #Stores spectrums of KE in each direction & IE
k_plot = 0.5 * (sections[:-1] + sections[1:])
filename = simName + "k_for_plot.npz"
savez_compressed(filename, k_plot)
del k_plot,sections,K1d
def saveSpectra(U, rho, P_hat, tstep):
    #Saves KE spectrum & pressure spectrum
    # temp[:] = 0.0
    for j in range(3):
        temp_hat[:] = fftn_mpi(U[j]*sqrt(rho),temp_hat)      #Read temp_hat as sqrt_rho_Ui_hat
        # temp_hat[:,:,:N//2+1] /= N3       #Check if normalisation needed.
        for i in range(Nk-1): spectra_k_plot[j,i] = sum((abs(temp_hat)**2).flatten()[ixbin[i]])
        # temp[:,:,:N//2+1] += abs(temp_hat[:,:,:])**2      #Storing spectrum in different direction needed or not??
    for i in range(Nk-1): spectra_k_plot[3,i] = sum((abs(P_hat)**2).flatten()[ixbin[i]])
    spectra_k_plot_total = allreduce_mpi_numpy(spectra_k_plot)
    if rank == 0:
        filename = simName + "spectra_" + "%05d" % tstep + "_triadRun.npz"
        savez_compressed(filename, spectra_k_plot_total)

def saveSnapshot(U_hat, rho_hat, P_hat):
    filename = simName + "snapshot_%05d" % tstep + "_triadRun.hdf5"
    #Parallel write to hdf5 file
    f = h5py.File(filename, 'w', driver='mpio', comm=comm)
    dset = f.create_dataset("U_hat", (3,2*kmax+1,2*kmax+1,kmax+1), dtype=complex)
    dset2 = f.create_dataset("rho_hat", (2*kmax+1,2*kmax+1,kmax+1), dtype=complex)
    dset3 = f.create_dataset("P_hat", (2*kmax+1,2*kmax+1,kmax+1), dtype=complex)
    if rank*Np<(kmax+1):
        temp3 = min(Np,kmax+1-rank*Np)
        dset[:,:kmax+1,rank*Np:rank*Np+temp3,:] = U_hat[:,:kmax+1,:temp3,:kmax+1]
        dset[:,-kmax:,rank*Np:rank*Np+temp3,:] = U_hat[:,-kmax:,:temp3,:kmax+1]
        dset2[:kmax+1,rank*Np:rank*Np+temp3,:] = rho_hat[:kmax+1,:temp3,:kmax+1]
        dset2[-kmax:,rank*Np:rank*Np+temp3,:] = rho_hat[-kmax:,:temp3,:kmax+1]
        dset3[:kmax+1,rank*Np:rank*Np+temp3,:] = P_hat[:kmax+1,:temp3,:kmax+1]
        dset3[-kmax:,rank*Np:rank*Np+temp3,:] = P_hat[-kmax:,:temp3,:kmax+1]
        del temp3
    if (rank+1)*Np>(N-kmax):
        temp2 = 2*kmax + 1 + (rank+1)*Np - N
        temp3 = max(kmax+1,temp2-Np)
        temp4 = temp3+Np-temp2
        dset[:,:kmax+1,temp3:temp2,:] = U_hat[:,:kmax+1,temp4:,:kmax+1]
        dset[:,-kmax:,temp3:temp2,:] = U_hat[:,-kmax:,temp4:,:kmax+1]
        dset2[:kmax+1,temp3:temp2,:] = rho_hat[:kmax+1,temp4:,:kmax+1]
        dset2[-kmax:,temp3:temp2,:] = rho_hat[-kmax:,temp4:,:kmax+1]
        dset3[:kmax+1,temp3:temp2,:] = P_hat[:kmax+1,temp4:,:kmax+1]
        dset3[-kmax:,temp3:temp2,:] = P_hat[-kmax:,temp4:,:kmax+1]
        del temp2,temp3,temp4
    f.close()

triadicInt = []
for k in range(kmax+1):
    temp2 = []
    for p in range(i1+1,kmax+1):
        qN = p+k+1-max(0,p-k-2)
        temp2.append(np.zeros((3,qN)))  #triplet kpq values can be found at triadicInt[k][p-k-1][q-(p-k-1)-(p==k+1)]
    triadicInt.append(temp2)
del temp2
if rank==num_processes//2:
    triad_file = triadDir+"triadicIntRun_N"+str(N)+"_fill"+str(fillLim)+".bin"
    triadfile = open(triad_file, "ab")
def calcTriads(U_hat, U, rho, tstep):
    for j in range(3): w_hat[j] = fftn_mpi(U[j]*sqrt(rho)/1e6,w_hat[j])     #Output will be normalised by 1e-12. IMPORTANT!!
    ##Make send objects of p & q
    for dataPacket in kpqData:
        if rank==dataPacket[0]: continue    #Nothing to send if rank is k-holder
        if rank==dataPacket[1]:
            temp2 = dataPacket[3]        #Make send objects if this process is a p-source
            if len(temp2)==1: temp2 = [temp2]      #To make it iterable
            data = np.array([w_hat[:,tmp[8],tmp[11],tmp[10]] for [tmp] in temp2])        #Bracket on tmp, because temp2 has 1 element, which contains the relevant np array.
            comm.Isend(data, dest=dataPacket[0], tag=dataPacket[2])     #Tag would have not been required if loop was continued for next few iterations, till p remains same. But this is fine for now.
        if rank==dataPacket[2]:
            data = []                   #Make send objects if this process is a q-source
            temp2 = dataPacket[3]
            if len(temp2)==1: temp2 = [temp2]      #To make it iterable
            for [tmp] in temp2:
                if tmp[0]==0: data.append(np.dot((tmp[8:11] - tmp[4:7]), U_hat[:,tmp[12],tmp[13],tmp[14]].conj()))
                elif tmp[1]==1: data.append(np.dot((tmp[8:11] + tmp[4:7]), U_hat[:,tmp[12],tmp[13],tmp[14]].conj()))
                else: data.append(np.dot((tmp[8:11] + tmp[4:7]), U_hat[:,tmp[12],tmp[13],tmp[14]]))
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
                    if tmp[0]==0: wpwk[i] *= np.dot((tmp[8:11] - tmp[4:7]), U_hat[:,tmp[12],tmp[13],tmp[14]].conj())
                    elif tmp[1]==1: wpwk[i] *= np.dot((tmp[8:11] + tmp[4:7]), U_hat[:,tmp[12],tmp[13],tmp[14]].conj())
                    else: wpwk[i] *= np.dot((tmp[8:11] + tmp[4:7]), U_hat[:,tmp[12],tmp[13],tmp[14]])
                    i+=1
            else:
                data = np.empty(len(wpwk),dtype=np.complex128)
                req = comm.Irecv(data, source=dataPacket[2], tag=dataPacket[1])
                tstart = time()
                req.Wait()
                twait = time()-tstart
                wpwk *= data
            #Fill the calculated quantity into triadicInt array
            i = 0
            for [tmp] in temp2:
                temp3 = tmp[2]-tmp[1]-1     #triadicInt[k][p-k-1][q-(p-k-1)-(p==k+1)]
                temp4 = triadicInt[tmp[1]][temp3][tmp[3]-temp3-int(temp3==0)]
                temp5 = np.imag(wpwk[i])
                temp4[0] += temp5
                temp4[1] += temp5**2
                temp4[2] += 1
                i+=1
    ##Bring all triadic interaction quantities to one process
    if tstep%10==0:
        triadicInt2 = allreduce_mpi_numpy(triadicInt)   #Save if timestep is multiple of 10
        triadicInt2.append(tstep)
        if rank==num_processes//2:
            np.save(triadfile,triadicInt2)
            triadfile.flush()
        del triadicInt2

##--------------------Useful Functions Done--------------------##


##--Allocate space for remaining variables--##
M_hat0 = empty((3, N, Np, N//2+1), dtype=complex)
M_hat1 = empty((3, N, Np, N//2+1), dtype=complex)
P_hat0 = empty((N, Np, N//2+1), dtype=complex)
P_hat1 = empty((N, Np, N//2+1), dtype=complex)
rho_hat0 = empty((N, Np, N//2+1), dtype=complex)
rho_hat1 = empty((N, Np, N//2+1), dtype=complex)
dU = empty((N, Np, N//2+1), dtype=complex)          #Stores timestep changes for rho_hat, M_hat, and P_hat
##------------------------------------------##

#RK4 coefficients
a = [1./6., 1./3., 1./3., 1./6.]
b = [0.5, 0.5, 1.]
#Non-dimensional N-S equations' coefficients
[c1,c2,c3,c4] = [1/Mc**2, 1/Rec, (gamma-1)*Mc**2/Rec, 1/Prc/Rec]
#Dealias ICs
for i in range(3):
    U_hat[i] *= dealias
    M_hat[i] *= dealias
rho_hat[:] *= dealias
P_hat[:] *= dealias
# T_hat[:] *= dealias

######----------START SIMULATION RUNS---------######
if rank==0: print("Starting simulation!!")
t = 0.0
tstep = 0
dt = 2*pi/N/(maxi(abs(U[0])+abs(U[1])+abs(U[2]))+3*cc/uc)       #Dimless time. To get dimensional time, multiply with (l/uc)
ts_save1 = 10      #Timestep intervals to save simulation parameters at.
ts_save2 = 100      #Timestep intervals to save energy spectra at.
ts_save3 = 200      #Timestep intervals to save snapshots at.
triadStart = 1200
triadStop = 1800
saveParams(U, U_hat, rho, P, t, tstep)      #Save parameters at 0th timestep.
saveSpectra(U, rho, P_hat, tstep)           #Save spectra at 0th timestep.
while True:
    #Execute optional code during runtime. Use this to change dt, add forcing, update/add triadic interaction code, print useful variables, etc.
    try: exec(open(simName+"optionalCode.py",'r').read())
    except: None
    #Save everything and exit, if walltime is almost finished.
    # walltime = time() - wt0
    # if (walltime_avail-walltime)<300:
    #     saveParams(U, U_hat, rho, P, t, tstep)
    #     saveSpectra(U, rho, P_hat, tstep)
    #     saveSnapshot(U_hat, rho_hat, P_hat)
    #     break
    t += dt
    tstep += 1
    ##--Solve the 3D N-S equations for compressible flow--##
    rho_hat1[:] = rho_hat
    rho_hat0[:] = rho_hat
    M_hat1[:] = M_hat
    M_hat0[:] = M_hat
    P_hat1[:] = P_hat
    P_hat0[:] = P_hat
    for rk in range(4):
        if rk > 0:
            #Update U_hat, T_hat, P and U, which will be required for computing RHS.
            rho[:] = ifftn_mpi(rho_hat, rho)
            for i in range(3): U[i] = ifftn_mpi(M_hat[i], U[i])
            U[:] = U[:]/rho         #Adds some aliasing errors, but they're expected to be negligible
            P[:] = ifftn_mpi(P_hat, P)
            for i in range(3):
                U_hat[i] = fftn_mpi(U[i],U_hat[i])
                U_hat[i] *= dealias
            # T_hat[:] = fftn_mpi((gamma*P)/rho,T_hat)            #P and rho to be found using larger arrays?
        #---Continuity Eqn---#
        dU[:] = -1j*sum(K*M_hat,axis=0)
        rho_hat1[:] += a[rk]*dt*dU
        if rk < 3: rho_hat[:] = rho_hat0 + b[rk]*dt*dU      #Safe to update rho_hat as it is not needed in mom & energy eqns
        #---Momentum Eqns---#
        for i in range(3):
            dU[:] = -1j*c1*K[i]*P_hat - c2*(K2*U_hat[i] + K[i]/3*sum(K*U_hat, axis=0))
            #Calculate useful fields for energy equation also
            duidxi[i] = ifftn_mpi(1j*K[i]*U_hat[i], duidxi[i])
            for j in range(3):
                temp_hat[:] = fftn_mpi(rho*U[j]*U[i], temp_hat)
                temp_hat[:] *= dealias
                dU[:] -= 1j*K[j]*temp_hat                      #RHS of ith Momentum equation is done when this loop ends
            M_hat1[i] += a[rk]*dt*dU
            if rk < 3: M_hat[i] = M_hat0[i] + b[rk]*dt*dU   #Safe to update M_hat as it is not needed in energy eqn
        #---Energy Eqn---#
        temp_hat[:] = fftn_mpi((gamma*P)/rho,temp_hat)
        dU[:] = -c4*K2*temp_hat
        temp[:] = 0.0           #Read as pdiff (pressure diffusion term)
        for i in range(3):
            temp_hat[:] = fftn_mpi(P*U[i], temp_hat)        #Read temp_hat as puTerm1_hat
            dU[:] -= 1j*K[i]*temp_hat
            temp_hat[:] = fftn_mpi(P*duidxi[i], temp_hat)   #Read temp_hat as puTerm2_hat
            dU[:] -= (gamma-1)*temp_hat
            temp[:] += 4/3*duidxi[i]*(duidxi[i]-duidxi[i-1])
        #duidxi values no longer needed. Reuse it to store duidxi-1 and dui-1dxi for each i.
        for i in range(3):
            duidxi[0] = ifftn_mpi(1j*K[i-1]*U_hat[i], duidxi[0])        #Read duidxi as duidxi-1
            duidxi[1] = ifftn_mpi(1j*K[i]*U_hat[i-1], duidxi[1])        #Read duidxi as dui-1dxi
            temp[:] += (duidxi[1]+duidxi[0])**2
        temp_hat[:] = fftn_mpi(temp, temp_hat)             #Read temp_hat as pdiff_hat
        dU[:] += c3*temp_hat
        dU[:] *= dealias            #De-alias complete RHS, as all terms in third equation are non-linear
        P_hat1[:] += a[rk]*dt*dU
        if rk < 3: P_hat[:] = P_hat0 + b[rk]*dt*dU
    rho_hat[:] = rho_hat1[:]
    M_hat[:] = M_hat1[:]
    P_hat[:] = P_hat1[:]
    ##-----------------------------------------------##
    #Now find this timestep's solution in Cartesian space
    rho[:] = ifftn_mpi(rho_hat, rho)
    for i in range(3): U[i] = ifftn_mpi(M_hat[i], U[i])
    U[:] = U[:]/rho                         #Dealiasing error unaccounted for :(
    P[:] = ifftn_mpi(P_hat, P)
    #Complete the solution in Fourier Space
    for i in range(3):
        U_hat[i] = fftn_mpi(U[i],U_hat[i])
        U_hat[i] *= dealias
    # T_hat[:] = fftn_mpi((gamma*P)/rho,T_hat)    #No need to dealias as dU is dealiased after T_hat is used.
    #Save, if necessary
    if tstep%ts_save1==0: saveParams(U, U_hat, rho, P, t, tstep)
    if tstep%ts_save2==0: saveSpectra(U, rho, P_hat, tstep)
    if tstep%ts_save3==0: saveSnapshot(U_hat, rho_hat, P_hat)
    if tstep>=triadStart and tstep<triadStop: calcTriads(U_hat, U, rho, tstep)       #Calculate Triadic Interactions
    #Change deltaT if required after every 100 timesteps:
    # if tstep%100==0:
    #     dt_old = dt
    #     dt *= 2
    #     dt = min(dt,0.01 * (64 / kmax))
    #     if dt==dt_old*2: print("deltaT increased to "+str(dt)+" seconds.")
    #     maxU_old = maxi(abs(U[0])+abs(U[1])+abs(U[2]))
    # elif 0<tstep%100<=5:
    #     maxU = maxi(abs(U[0])+abs(U[1])+abs(U[2]))
    #     if abs(maxU-maxU_old)>0.05*maxU_old:
    #         dt = dt_old     #Change deltaT back to old one if fluctuations are high
    #         print("deltaT reduced back to "+str(dt)+" seconds.")

