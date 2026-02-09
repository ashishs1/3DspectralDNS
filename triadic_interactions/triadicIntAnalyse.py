##Script to analyse triadic interactions

from time import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import random

kmax = 86
N = 3*kmax-2
fillLim = 200
# triadDir = "E:\\MTech\\Project\\DNS Code\\triadIntFull\\"
triadDir = "J:\\MTech Project\\DNS Code\\triadIntFull\\"
triad_file = triadDir+"triadicIntRun_N"+str(N)+"_fill"+str(fillLim)+".bin"

##Read array(s) from file
triadFile = open(triad_file,'rb')
triadData = []
while True:
    try: triadData.append(pickle.load(triadFile))
    except EOFError: break
triadFile.close()
#Now, all the data has been loaded in triadData (unless there was a memory error while loading)
# triadData = triadData[-1]         #Pick the last list that was dumped
# print("Data from timestep",str(triadData[-1]))
# triadData = triadData[:-1]
nData = len(triadData)      #No. of timesteps for which data is available
triadicInt = np.zeros((nData,kmax+1,kmax+1,kmax+1,3))
#Get inputs for each triadicInt[k,p,q] from triadicData[k][p-k-1][q-(p-k-1)-(p==k+1)]
for i0 in range(nData):
    for i1 in range(1,kmax+1):
        for i2 in range(i1+1,kmax+1):
            i2dash = i2-i1-1
            for i3 in range(max(1,i2dash),min(i2+i1+2,kmax+1)):
                try: triadicInt[i0,i1,i2,i3,:] = triadData[i0][i1][i2dash][i3-i2dash-int(i2dash==0)]
                except:
                    print([i1,i2dash,i3-i2dash-int(i2dash==0)])
                    break

for i1 in range(1,kmax+1):
    for i2 in range(i1):
        triadicInt[:,i1,i2,:,0] = -triadicInt[:,i2,i1,:,0]     #S(p|k|q) = -S(k|p|q)
        triadicInt[:,i1,i2,:,1] = triadicInt[:,i2,i1,:,1]
        triadicInt[:,i2,i1,:,2]*=2        #Because 2 triadic interactions are summed at once during calculations.
        triadicInt[:,i1,i2,:,2] = triadicInt[:,i2,i1,:,2]   #S(p|k|q) for p<k has not been found.

triadicInt[:,:,:,:,0] *= 1e20  #Values were normalised by 1e20 during runtime
triadicInt[:,:,:,:,1] *= 1e40  #Values were normalised by 1e20 during runtime
triadicMean = np.zeros((nData,kmax+1,kmax+1,kmax+1))  #Saves the average value of k-p-q interaction
triadicStd = np.zeros((nData,kmax+1,kmax+1,kmax+1))  #Saves the std. deviation value of k-p-q interaction
triadicCoV = np.zeros((nData,kmax+1,kmax+1,kmax+1))  #Saves the coefficient of variation value of k-p-q interaction

triadicMean[:] = triadicInt[:,:,:,:,0]/np.where(triadicInt[:,:,:,:,2]==0,1,triadicInt[:,:,:,:,2])
triadicStd[:] = np.sqrt(triadicInt[:,:,:,:,1]/np.where(triadicInt[:,:,:,:,2]==0,1,triadicInt[:,:,:,:,2]) - triadicMean**2)
triadicCoV[:] = triadicStd/np.where(triadicMean==0,1,triadicMean)

#Fix p~20, and plot P(k|p) = sum(P(k|p|q)) over all q's v/s k.
##P(k|p) plot for shell thicknesses of (k,p) as (2,2)
ks = np.arange(2,65,2)
for i0 in range(12,18):
    print(i0)
    for pi in [20]:
        # pi = 20
        Pkps = []
        for k in ks:
            temp = np.array([],dtype=float)
            for q in range(kmax+1):
                temp2 = triadicMean[i0,k:k+2,pi:pi+2,q]
                temp2 = temp2[np.isfinite(temp2)]
                temp2 = temp2[(temp2!=0)]
                temp2 = temp2[(np.abs(temp2)<1e17)]
                temp = np.append(temp,temp2)
                # temp2 = triadicMean[i0,k:k+2,q,pi:pi+2]
                # temp2 = temp2[np.isfinite(temp2)]
                # temp2 = temp2[(temp2!=0)]
                # temp2 = temp2[(abs(temp2)<1e17)]
                # temp = np.append(temp,temp2)        #P(k|p) = 4*pi*k**2*{average over q[T(k|p,q)]}. Ideally, it should be average over k & p, and sum over q.
            if len(temp)!=0:
                if not np.isfinite(np.sum(temp)): print(temp)
                temp = np.average(temp)
            else: temp = 0
            Pkps.append(temp*4*np.pi*(k+0.5)**2)
        plt.figure()
        plt.plot(ks+0.5,Pkps)
        # plt.title("P(k|p) for p in wavenumber band "+str(pi-1)+"<p<"+str(pi+1))
        # plt.ylim(-1e9,1e9)
        plt.ylabel("P(k|p)")
        plt.xlabel("k")
        plt.axvline(pi-0.5,1.05*min(Pkps),1.05*max(Pkps),color="black",linestyle=(0,(1,5)))
        plt.axvline(pi+1.5,1.05*min(Pkps),1.05*max(Pkps),color="black",linestyle=(0,(1,5)))
        plt.show()

plt.show()
##P(k|p) plot for shell thicknesses of (k,p) as (1,1)
# # ks = np.arange(2,65)
# # for i0 in range(8,20):
# #     print(i0)
# #     for pi in range(14,18):
# #         # pi = 20
# #         Pkps = []
# #         for k in ks:
# #             temp = 0
# #             for q in range(kmax+1):
# #                 if str(triadicInt[i0,k,pi,q,0])!='nan':
# #                     tmp = triadicInt[i0,k,pi,q,2]-triadicInt[3,k,pi,q,2]
# #                     temp+=(triadicInt[i0,k,pi,q,0]-triadicInt[3,k,pi,q,0])/(np.where(tmp==0,1,tmp))       #P(k|p) = 4*pi*k**2*{sum over q[T(k|p,q)]}. Take average after peak energy dissipation only
# #                 # if str(triadicMean[i0,k,pi,q])!='nan':
# #                 #     temp+=triadicMean[i0,k,pi,q]       #P(k|p) = 4*pi*k**2*{sum over q[T(k|p,q)]}
# #                 # else: print("Found bad int value at ("+str(k)+","+str(pi)+").")
# #                 # if str(triadicMean[i0,k,pi+1,q])!='nan':
# #                 #     temp+=triadicMean[i0,k,pi+1,q]       #P(k|p) = 4*pi*k**2*{sum over q[T(k|p,q)]}
# #                 # else: print("Found bad int value at ("+str(k)+","+str(pi)+").")
# #             Pkps.append(temp*4*np.pi*k**2)
# #         plt.plot(ks,Pkps)
# #         plt.title("P(k|p) for p in wavenumber band "+str(pi-0.5)+"<p<"+str(pi+0.5))
# #         # plt.ylim(-1e9,1e9)
# #         plt.xlabel("k")
# #         plt.show()

##P(k|p) plot for shell thicknesses of (k,p) as (1,2)
# # ks = np.arange(2,65)
# # for i0 in range(5,nData):
# #     print(i0)
# #     for pi in range(20,25):
# #         Pkps = []
# #         for k in ks:
# #             temp = 0
# #             for q in range(kmax+1):
# #                 if str(triadicMean[i0,k,pi,q])!='nan':
# #                     temp+=triadicMean[i0,k,pi,q]       #P(k|p) = 4*pi*k**2*{sum over q[T(k|p,q)]}
# #                 # else: print("Found bad int value at ("+str(k)+","+str(pi)+").")
# #                 if str(triadicMean[i0,k,pi+1,q])!='nan':
# #                     temp+=triadicMean[i0,k,pi+1,q]       #P(k|p) = 4*pi*k**2*{sum over q[T(k|p,q)]}
# #                 # else: print("Found bad int value at ("+str(k)+","+str(pi)+").")
# #             Pkps.append(temp*4*np.pi*k**2)
# #         plt.plot(ks,Pkps)
# #         plt.title("P(k|p) at p="+str(pi))
# #         # plt.ylim(-1e9,1e9)
# #         plt.xlabel("k")
# #         plt.show()

##Contour plot of P(k|p) (shell thickness of 3)
from mpl_toolkits import mplot3d
ks, ps = np.meshgrid(np.arange((kmax+1)//3)*3+1.5,np.arange((kmax+1)//2)*2+1)
# if str(triadicMean[i1,i2,i3])!='nan' and str(triadicMean[i1,i2,i3])[-3:]!='inf':
Pkp_clean = np.zeros(((kmax+1)//2,(kmax+1)//3))
tstp = -1
for i1 in range((kmax+1)//2):
    for i2 in range((kmax+1)//3):
        for i3 in range(kmax+1):
            if str(triadicMean[tstp,2*i1,3*i2,i3])!='nan':
                Pkp_clean[i1,i2] += triadicMean[tstp,2*i1,3*i2,i3]
            if str(triadicMean[tstp,2*i1,3*i2+1,i3])!='nan':
                Pkp_clean[i1,i2] += triadicMean[tstp,2*i1,3*i2+1,i3]
            if str(triadicMean[tstp,2*i1,3*i2+2,i3])!='nan':
                Pkp_clean[i1,i2] += triadicMean[tstp,2*i1,3*i2+2,i3]
            if str(triadicMean[tstp,2*i1+1,3*i2,i3])!='nan':
                Pkp_clean[i1,i2] += triadicMean[tstp,2*i1+1,3*i2,i3]
            if str(triadicMean[tstp,2*i1+1,3*i2+1,i3])!='nan':
                Pkp_clean[i1,i2] += triadicMean[tstp,2*i1+1,3*i2+1,i3]
            if str(triadicMean[tstp,2*i1+1,3*i2+2,i3])!='nan':
                Pkp_clean[i1,i2] += triadicMean[tstp,2*i1+1,3*i2+2,i3]
linthresh = 10
for i1 in range((kmax+1)//2):
    for i2 in range((kmax+1)//3):
        if abs(Pkp_clean[i1,i2])>1e45 or str(Pkp_clean[i1,i2])=='nan': Pkp_clean[i1,i2]=1        #Set to zero, if outlier
        #Change Pkp_clean to symlog scaling:
        # if abs(Pkp_clean[i1,i2])<=linthresh: Pkp_clean[i1,i2]*=1/linthresh
        # elif Pkp_clean[i1,i2] < 0: Pkp_clean[i1,i2] = -np.log10(-Pkp_clean[i1,i2])
        # else: Pkp_clean[i1,i2] = np.log10(Pkp_clean[i1,i2])

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
# ax.set_zscale('symlog',linthresh=linthresh)
# ax.set_zticks([1e10,1e22,5e22])
# ax.contour3D(ks, ps, Pkp_clean, 50,cmap = 'binary',norm='symlog')
ax.plot_wireframe(ks, ps, Pkp_clean, rstride=1, cstride=1, cmap = 'binary')

ax.set_xlabel('k')
ax.set_ylabel('p')
ax.set_zlabel('P(K|p)')
plt.show()

##P(k|p) plot for shell thicknesses of (k,p) as (2,5)
[kband,pband,qband] = [2,2,5]
ks = np.arange(2,60,kband)
dtstp = 5

for i0 in range(12,13):
    print(i0)
    for pi in range(24,25):
        # pi = 20
        Pkps = []
        for k in ks:
            if np.any(triadicInt[i0,k:k+kband,pi:pi+pband,:,2]!=0):
                tmp = triadicInt[i0,k:k+kband,pi:pi+pband,:,2] - triadicInt[i0-dtstp,k:k+kband,pi:pi+pband,:,2]
                temp = (triadicInt[i0,k:k+kband,pi:pi+pband,:,0] - triadicInt[i0-dtstp,k:k+kband,pi:pi+pband,:,0])/np.where(tmp==0,1,tmp)
                temp = np.where(np.isfinite(temp),temp,0)
                # temp *= int(abs(temp)<1e19)
                temp2 = np.sum(temp)       #P(k|p) = 4*pi*k**2*{sum over q[T(k|p,q)]}
            Pkps.append(temp2*4*np.pi*(k+(kband-1)/2)**2)
        plt.figure()
        plt.plot(ks+(kband-1)/2,Pkps)
        # plt.title("P(k|p) for p in wavenumber band "+str(pi-1)+"<p<"+str(pi+1))
        # plt.ylim(-1e9,1e9)
        plt.ylabel("P(k|p)")
        plt.xlabel("k")
        plt.axvline(pi-0.5,1.05*min(Pkps),1.05*max(Pkps),color="black",linestyle=(0,(1,5)))
        plt.axvline(pi+(pband-1)/2,1.05*min(Pkps),1.05*max(Pkps),color="black",linestyle=(0,(1,5)))
        plt.show()

##4D Plots
from mpl_toolkits.mplot3d import Axes3D

triadicMean_clean = []
triadicStd_clean = []
triadicCoV_clean = []
tstp = 10
#Change to bands of 5:
[kband,pband,qband] = [5,5,5]
for i1 in range((kmax+1)//kband):
    for i2 in range((kmax+1)//pband):
        for i3 in range((kmax+1)//qband):
            [k1,p1,q1] = [kband*i1,pband*i2,qband*i3]
            if np.any(triadicInt[tstp,k1:k1+kband,p1:p1+pband,q1:q1+qband,2]!=0):
                temp = triadicMean[tstp,k1:k1+kband,p1:p1+pband,q1:q1+qband]
                temp = temp[np.isfinite(temp)]
                temp = temp[(temp!=0)]
                temp = temp[(np.abs(temp)<1e17)]
                if len(temp)!=0: triadicMean_clean.append([k1,p1,q1,np.average(temp)])
                else: triadicMean_clean.append([k1,p1,q1,0])
                temp = triadicStd[tstp,k1:k1+kband,p1:p1+pband,q1:q1+qband]
                temp = temp[np.isfinite(temp)]
                triadicStd_clean.append([k1,p1,q1,np.sum(temp)])
                temp = triadicCoV[tstp,k1:k1+kband,p1:p1+pband,q1:q1+qband]
                temp = temp[np.isfinite(temp)]
                triadicCoV_clean.append([k1,p1,q1,np.sum(temp)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
temp = random.sample(triadicMean_clean,min(2000,len(triadicMean_clean)))
# temp = [tmp for tmp in temp if abs(tmp[3])>1e10]    #Keep only points having value larger than a threshold
x = np.array([tmp[0] for tmp in temp])
y = np.array([tmp[1] for tmp in temp])
z = np.array([tmp[2] for tmp in temp])
c = np.array([tmp[3] for tmp in temp])
# img = ax.scatter(x, y, z, c=c, cmap='viridis',norm='symlog')
img = ax.scatter(x, y, z, c=c, cmap='viridis',norm=clrs.SymLogNorm(vmin=-1e14,vmax=1e14,linthresh=1e7))
ax.set_xlabel('k')
ax.set_ylabel('p')
ax.set_zlabel('q')
ax.set_title('Mean of Triadic Interactions')
fig.colorbar(img)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
temp = random.sample(triadicStd_clean,2000)
x = np.array([tmp[0] for tmp in temp])
y = np.array([tmp[1] for tmp in temp])
z = np.array([tmp[2] for tmp in temp])
c = np.array([tmp[3] for tmp in temp])
img = ax.scatter(x, y, z, c=c, cmap='winter',norm=clrs.LogNorm())
ax.set_xlabel('k')
ax.set_ylabel('p')
ax.set_zlabel('q')
ax.set_title('Std Deviation of Triadic Interactions')
fig.colorbar(img)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
temp = random.sample(triadicCoV_clean,2000)
x = np.array([tmp[0] for tmp in temp])
y = np.array([tmp[1] for tmp in temp])
z = np.array([tmp[2] for tmp in temp])
c = np.array([tmp[3] for tmp in temp])
img = ax.scatter(x, y, z, c=c, cmap='winter',vmin=-30,vmax=30)
ax.set_xlabel('k')
ax.set_ylabel('p')
ax.set_zlabel('q')
ax.set_title('Coefficient of Variation of Triadic Interactions')
fig.colorbar(img)
plt.show()

##Contour plot of S(k|p|q) over different fixed extremes of any one of the wavenumbers
import matplotlib.ticker as mticker

[kband,pband,qband] = [1,1,1]
tstp = 10
triadicMean_new = np.zeros(((kmax+1)//kband,(kmax+1)//pband,(kmax+1)//qband))
for i1 in range((kmax+1)//kband):
    for i2 in range((kmax+1)//pband):
        for i3 in range((kmax+1)//qband):
            [k1,p1,q1] = [i1*kband,i2*pband,i3*qband]
            if np.any(triadicInt[tstp,k1:k1+kband,p1:p1+pband,q1:q1+qband,2]!=0):
                tmp = triadicInt[tstp,k1:k1+kband,p1:p1+pband,q1:q1+qband,2]-triadicInt[tstp-10,k1:k1+kband,p1:p1+pband,q1:q1+qband,2]
                tmp = np.where(tmp==0,1,tmp)
                temp = (triadicInt[tstp,k1:k1+kband,p1:p1+pband,q1:q1+qband,0]-triadicInt[tstp-10,k1:k1+kband,p1:p1+pband,q1:q1+qband,0])/tmp
                temp = temp[np.isfinite(temp)]
                temp = temp[(temp!=0)]
                temp = temp[(np.abs(temp)<1e14)]
                triadicMean_new[i1,i2,i3] = np.average(temp)
[qi1,qi2] = [1,-3]      #Indices of extremes to plot for
[pi1,pi2] = [1,-2]
[ki1,ki2] = [1,-2]

def sym_log(data,linthresh):
    tmp = data
    tmp2 = (np.abs(tmp)<=linthresh)*tmp/linthresh + (tmp>linthresh)*np.log10((tmp*(tmp>0)+(tmp<1e-16))*10/linthresh) - (tmp<-linthresh)*np.log10((-tmp*(tmp<0)+(tmp>-1e-16))*10/linthresh)
    return tmp2

def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"

ps, ks = np.meshgrid((np.arange((kmax+1)//kband)+.5)*kband,(np.arange((kmax+1)//pband)+.5)*pband)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
# ax.contour3D(ks, ps, sym_log(triadicMean_new[:,:,qi2],10), 50,cmap = 'binary',norm='symlog')
# ax.plot_wireframe(ks, ps, sym_log(triadicMean_new[:,:,qi2],1e15), rstride=1, cstride=1, cmap = 'binary')
# ax.plot_wireframe(ks, ps, np.log10(np.abs(triadicMean_new[:,:,qi2])+(triadicMean_new[:,:,qi2]==0)), rstride=2, cstride=2)
surf = ax.plot_surface(ks, ps, np.log10(np.abs(triadicMean_new[:,:,qi2])), edgecolor='royalblue', lw=0.5, rstride=2, cstride=2, alpha=0.3)
ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.contourf(ks, ps, triadicMean_new[:,:,qi2]/1e14, zdir='z', offset=-8.5, cmap='coolwarm')
ax.contourf(ks, ps, triadicMean_new[:,:,qi2]/1e14, zdir='x', offset=0, cmap='coolwarm')
ax.contourf(ks, ps, triadicMean_new[:,:,qi2]/1e14, zdir='y', offset=0, cmap='coolwarm')
ax.set(xlabel='k', ylabel='p', zlabel='$S^{ww}(k|p|q)$', title = 'Surface plot of S(k|p|q) at q='+str(ks[qi2,0]))
ax.set_xlim(0,kmax)
ax.set_ylim(0,kmax)
# fig.colorbar(surf)
plt.show()