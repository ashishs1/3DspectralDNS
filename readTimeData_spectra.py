# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:43:52 2022

@author: singh
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftfreq
import os
from PIL import Image

os.chdir("G:\\MTech Project\\sim8\\")
kmax = 86
[l, rhoc, pc, muc, kc, gamma, R] = [0.0005, 0.1664, 1e5, 1.96e-5*1.1, 15.34e-2, 1.67, 2079]         #Indicative values of He@NTP (SI units: [m, kg/m3, Pa, kg/ms, W/mdegC, dimless, J/kgK])
urms = kmax**(4/3)*muc/rhoc/l   #Intended urms in SI units (Since n/L ~ 1/kmax)
cc = np.sqrt(gamma*pc/rhoc)        #Characteristic speed of sound, SI units
Mc = urms/cc                  #For TGV, intitially, urms~uc/2. So, for Mt~0.03, Mc~0.06.
[uc, Tc] = [Mc*cc, pc/rhoc/R]
Rec = rhoc*uc*l/muc
N = 3*kmax - 2
# kx = fftfreq(N, 1./N)
# kz = kx[:(N//2+1)].copy()
# kz[-1] *= -1
# K = np.array(np.meshgrid(kx, kx, kz, indexing='ij'), dtype=np.int16)
# K2 = np.sum(K.astype(np.int32)**2, 0, dtype=np.int32)
# del K
# K1d = np.sqrt(K2).flatten()
# del K2
# Nk = np.int32(np.sqrt(3)*N/2) + 1
# sections = np.linspace(0.0, np.sqrt(3)*N/2, Nk)
# ixbin = []
# for i in range(Nk-1): ixbin.append(np.where((K1d <= sections[i+1]) & (K1d > sections[i]))[0].astype(np.int16))
# ixbinN = []
# for i in range(len(ixbin)): ixbinN.append(len(ixbin[i]))
# np.save("ixbinN",ixbinN)
ixbinN2 = np.load("ixbinN.npy")

# file = "spectra_0145908.npz"
# # file = "spectra_0312310.npz"
# loaded = np.load(file)
# k_plot = loaded['arr_0'][:plotRange]
# spectarray = loaded['arr_1'][1][:plotRange]
# time = loaded['arr_2']
# loaded.close()
# fig = plt.figure(figsize=(18,12))
# plt.loglog(k_plot, spectarray, 'b')
# plt.loglog(k_plot[1:kmax+3], spectarray[2]*(k_plot[1:kmax+3]/k_plot[2])**(-7/3), 'b--')
# plt.title('Time = '+np.format_float_scientific(time,precision=2))
# plt.xlabel('|K|')
# plt.ylabel("$E_K$")
# # plt.yscale("log")
# # plt.xscale("log")
# tick_pos = [1, 10, 16]
# labels = ['1/L', '10', '$k_{max}$']
# plt.xticks(tick_pos, labels)
# # plt.legend()
# plt.ylim([1e3,1e16])
# plt.show()


######-----Read the Simulation Output Data-----######
pressure = True
counter=0
plt.rcParams.update({'font.size': 20})
list = os.listdir()
for file in list[::-1]:
    if file[-4:]!='.npz' or file[0:7]!='spectra': continue
    loaded = np.load(file)
    ampIndex = 8            #Index for deciding position of slope
    if pressure: amplitude = loaded['arr_0'][3][ampIndex]
    else: amplitude = loaded['arr_0'][1][ampIndex]
    loaded.close()
    break

plotRange = kmax*3//2 - 3

for file in os.listdir():
    if file=='k_for_plot.npz':
        loaded = np.load(file)
        k_plot = loaded['arr_0'][:plotRange]
        loaded.close()
        slopeArray = amplitude/N**3/ixbinN2[ampIndex]*k_plot[ampIndex]**2*(k_plot[1:kmax+3]/k_plot[ampIndex])**(-2)
        break

pngsDone = 0
images = []
if pressure: path = os.getcwd()+'\\pressure\\'
else: path = os.getcwd()
for file in sorted(os.listdir(path)):
    if file[-4:]=='.png':
        try: number = int(file[:-4])
        except: continue
        pngsDone+=1
        if len(images)==number: images.append(Image.open(file))
        elif len(images)>number: images[number] = Image.open(file)
        else:
            while len(images)<number: images.append(None)
            images.append(Image.open(file))


fig = plt.figure(figsize=(18,12))
for file in sorted(os.listdir()):
    if file[-4:]!='.npz' or file[0:7]!='spectra': continue
    counter+=1
    if counter<=pngsDone: continue
    loaded = np.load(file)
    if pressure: spectarray = loaded['arr_0'][3,:plotRange]/ixbinN2[:plotRange]*k_plot**2/N**3
    else: spectarray = np.sum(loaded['arr_0'][:3,:plotRange],axis=0)/ixbinN2[:plotRange]*k_plot**2/N**3
    # time = loaded['arr_2']
    loaded.close()
    tstep = file[8:13]
    plt.loglog(k_plot, spectarray, 'b')
    plt.loglog(k_plot[1:kmax+3], slopeArray, 'b--')
    plt.title('Timestep = '+tstep)
    plt.xlabel('|K|')
    if pressure: plt.ylabel("$P_K$")
    else: plt.ylabel("$E_K$")
    tick_pos = [1, 10, kmax]
    labels = ['1', '10', '$k_{max}$']
    plt.xticks(tick_pos, labels)
    if not pressure: plt.ylim([1e-9,1e7])
    else: plt.ylim([1e-6,1e3])
    # plt.legend()
    # plt.show()
    fig.canvas.draw()
    images.append(Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb()))
    plt.savefig(path+str(counter-1)+".png")
    # counter+=1
    print(counter-1)
    plt.clf()

os.chdir(path)
images[0].save('EnergySpectrum.gif', save_all=True, append_images=images[1:], duration=300, loop=0)
del fig
# plt.show()

##To estimate how much time it will take for energy spectrum to reach dissipation scales
# ks = []
# ts = []
# for file in os.listdir():
#     if file[-4:]!='.npz' or file[0:7]!='spectra': continue
#     loaded = np.load(file)
#     spectarray = np.sum(loaded['arr_0'][:3,:plotRange],axis=0)/ixbinN2[:plotRange]*k_plot**2
#     loaded.close()
#     temp = np.abs(spectarray - 1e-6)
#     idx = temp.argmin()
#     if spectarray[idx]>1e-6:idx2 = idx+1
#     else: idx2 = idx-1
#     kind = (temp[idx]*k_plot[idx2]+temp[idx2]*k_plot[idx])/(temp[idx]+temp[idx2])       #k indicative of 1e-6 value of spectrum
#     ks.append(kind)
#     ts.append(int(file[8:13]))
#
# ks = np.array(ks)
# ts = np.array(ts)
# coeffs = np.polyfit(np.log(ks),ts,1)
# l_by_n = 235.5              #Max k after which spectrum should fall sharply
# t_exp = 0
# for i in range(len(coeffs)): t_exp = t_exp*np.log(l_by_n) + coeffs[i]
# print("The spectrum is expected to mature after",round(t_exp),"timesteps.")
# plt.figure()
# plt.plot(np.log(ks),ts)
# ts_fit = ks*0
# for i in range(len(coeffs)):
#     ts_fit = ts_fit*np.log(ks) + coeffs[i]
# plt.plot(np.log(ks),ts_fit)
# plt.show()
