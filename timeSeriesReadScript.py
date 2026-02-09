import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir("G:\\MTech Project\\sim8\\")

#Time series is stored as:
# tstep    -->   0
# t        -->   1
# urms     -->   2
# Mt       -->   3
# epsilon  -->   4
# rho_mean -->   5
# KE       -->   6
# IE       -->   7
# U_avg    -->   8,9,10
# Sdudx    -->   11-37      #dudx moments. i,j value of k-th dudx moment is on index 11+9*(k-2)+3*i+j
# uiuj     -->   38,39,40

#Setting basic/initial simulation parameters:
kmax = 86
N = 3*kmax-2
[l, rhoc, pc, muc, kc, gamma, R] = [0.0005, 0.1664, 1e5, 1.96e-5*1.1, 15.34e-2, 1.67, 2079]         #Indicative values of He@NTP (SI units: [m, kg/m3, Pa, kg/ms, W/mdegC, dimless, J/kgK])
urms = kmax**(4/3)*muc/rhoc/l   #Intended urms in SI units (Since n/L ~ 1/kmax)
cc = np.sqrt(gamma*pc/rhoc)        #Characteristic speed of sound, SI units
Mc = urms/cc                  #For TGV, intitially, urms~uc/2. So, for Mt~0.03, Mc~0.06.
[uc, Tc] = [Mc*cc, pc/rhoc/R]
Rec = rhoc*uc*l/muc

snaps = [1200]          #tsteps where the simulation was restarted
counter=0
out = []
# time = 0
for filename in os.listdir():
    if filename[:10]!='timeSeries' or filename[-4:]!='.bin': continue
    tmp = len(out)
    f = open(filename,"rb")
    fsz = os.fstat(f.fileno()).st_size
    if tmp==0: out = np.load(f)
    # if time!=0:
    while f.tell() < fsz:
        out = np.vstack((out, np.load(f)))
    f.close()
    # dtstep = np.average(out[tmp+1:,1])
    # print(len(out))
    if counter<len(snaps): out = out[:snaps[counter]//10]    #Store values only for those timesteps which are not repeated in the next simulation
    counter+=1

for i in range(len(snaps)-1):
    lastCorrectTime = out[snaps[i]//10-1,1]
    dt_new = out[snaps[i]//10,1]/((snaps[i]+3290)//10+1)           #Actually this is 10*dt.
    print(dt_new)
    out[snaps[i]//10:,1]+=(lastCorrectTime+dt_new-out[snaps[i]//10,1])
tmp = out[1:,1] - out[:-1,1]

lamda = out[:,2]/Rec*np.sqrt(15/out[:,4])
Rlambda = out[:,5]*out[:,2]**2*np.sqrt(15/out[:,4])
# Mt = uc/cc*out[:,2]
epsilon = out[:,4]*uc**3/out[:,5]/l
nu = muc/rhoc/out[:,5]
eeta = (nu**3/epsilon)**0.25
epsilon0 = 15*nu*uc**2/l**2*out[:,11]

fig = plt.figure()
plt.plot(out[:,1],out[:,2]*uc)
plt.xlabel('Dimensionless time $(t/t_c)$')
plt.title('urms (m/s)')
plt.show()

fig = plt.figure()
plt.plot(out[:,1],out[:,5]*rhoc)
plt.title('$\u03c1_{mean} (kg/m^3)$')
plt.show()

fig = plt.figure()
plt.plot(out[:,1],out[:,8]*uc, label='$u\u0305_x$')
plt.plot(out[:,1],out[:,9]*uc, label='$u\u0305_y$')
plt.plot(out[:,1],out[:,10]*uc, label='$u\u0305_z$')
plt.xlabel('Dimensionless time $(t/t_c)$')
plt.title('$u\u0305$ (m/s)')
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(out[:,1],epsilon/1e3)
plt.xlabel('Dimensionless time $(t/t_c)$')
# plt.xlabel('No. of timesteps')
plt.title('Energy diffusion rate, \u03b5 (kJ/kg/s)')
# plt.title('epsilon (m\u00b2/s\u00b3)')
plt.show()

fig = plt.figure()
plt.plot(out[:,1],out[:,6]*1e6/(8*np.pi**3))
plt.title('KE (\u03bcJ)')
plt.xlabel('Dimensionless time $(t/t_c)$')
plt.show()

fig = plt.figure()
plt.plot(out[:,1],out[:,7]*1e6/(8*np.pi**3))
plt.title('IE (\u03bcJ)')
plt.xlabel('Dimensionless time $(t/t_c)$')
plt.show()

fig = plt.figure()
plt.plot(out[:,0],(out[:,7]+out[:,6])*1e6/(8*np.pi**3))
plt.title('TE (\u03bcJ)')
plt.xlabel('No. of timesteps')
plt.show()

fig = plt.figure()
plt.scatter(out[:,0],out[:,20]/out[:,11]**1.5, label = 'Sdudx')
# plt.scatter(out[:,0],out[:,20]/out[:,11]**1.5, label = 'Sdudy')
# plt.scatter(out[:,0],out[:,21]/out[:,12]**1.5, label = 'Sdudz')
# plt.scatter(out[:,0],out[:,22]/out[:,13]**1.5, label = 'Sdvdx')
# plt.scatter(out[:,0],out[:,23]/out[:,14]**1.5, label = 'Sdvdy')
# plt.scatter(out[:,0],out[:,24]/out[:,15]**1.5, label = 'Sdvdz')
# plt.scatter(out[:,0],out[:,25]/out[:,16]**1.5, label = 'Sdwdx')
# plt.scatter(out[:,0],out[:,26]/out[:,17]**1.5, label = 'Sdwdy')
# plt.scatter(out[:,0],out[:,27]/out[:,18]**1.5, label = 'Sdwdz')
plt.title('Vel. Derivative Skewness')
plt.legend()
plt.show()

fig = plt.figure()
# plt.scatter(out[:,1],out[:,29]/out[:,11]**2)
plt.scatter(out[:,3]**2*Rlambda,out[:,29]/out[:,11]**2)
plt.title('Kdudx')
plt.xlabel('$M_t^2R_\u03BB$')
plt.show()

fig = plt.figure()
plt.plot(out[:,1],Rlambda)
plt.title('Rlambda')
plt.xlabel('Dimensionless time $(t/t_c)$')
plt.show()

fig = plt.figure()
plt.plot(out[:,1],out[:,3]*Rlambda)
plt.title('$M_tR_\u03BB$')
plt.xlabel('Dimensionless time $(t/t_c)$')
plt.show()

fig = plt.figure()
plt.plot(out[:,1],l/eeta)
plt.title('L/\u03b7')
plt.xlabel('Dimensionless time $(t/t_c)$')
plt.show()

fig = plt.figure()
plt.plot(out[:,1],out[:,38], label='<u\u2081.u\u2082 >')        #*uc**2
plt.plot(out[:,1],out[:,39], label='<u\u2082.u\u2083 >')
plt.plot(out[:,1],out[:,40], label='<u\u2081.u\u2083 >')
plt.title('Velocity correlations (dimless)')
# plt.title('Velocity correlations $(m^2/s^2)$')
plt.xlabel('Dimensionless time $(t/t_c)$')
plt.legend()
plt.show()
