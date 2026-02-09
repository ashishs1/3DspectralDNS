cd("G:\\MTech Project\\Julia\\test")

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
kmax = 11
N = 3*kmax-1
l, rhoc, pc, muc, kc, gamma, R = 0.0005, 0.1664, 1e5, 1.96e-5*1.1, 15.34e-2, 1.67, 2079         #Indicative values of He@NTP (SI units: [m, kg/m3, Pa, kg/ms, W/mdegC, dimless, J/kgK])
urms = kmax^(4/3)*muc/rhoc/l   #Intended urms in SI units (Since n/L ~ 1/kmax)
cc = sqrt(gamma*pc/rhoc)        #Characteristic speed of sound, SI units
Mc = urms/cc                  #For TGV, intitially, urms~uc/2. So, for Mt~0.03, Mc~0.06.
uc, Tc = Mc*cc, pc/rhoc/R
Rec = rhoc*uc*l/muc

snaps = []          #tsteps where the simulation was restarted
counter=0
out = Matrix(undef,41,0)
line = Vector(undef,41)
# time = 0
for filename in readdir()
    if length(filename)<20 || filename[1:10]!="timeSeries" || filename[end-9:end]!="_julia.bin" continue end
    global out, counter
    tmp = size(out)[2]
    f = open(filename,"r")
    try
        while true
            for i in 1:41 line[i] = deserialize(f) end
            out = hcat(out,line)
        end
    catch EOFError
        print("File read completely!")
    end
    close(f)
    # dtstep = average(out[tmp+1:,1])
    # print(len(out))
    if counter<length(snaps) && snaps[counter+1]÷10>size(out)[2]
        out = out[:,1:snaps[counter+1]÷10]    #Store values only for those timesteps which are not repeated in the next simulation
    end
    counter+=1
end

for i in 1:length(snaps)-1
    lastCorrectTime = out[2,snaps[i]÷10]
    dt_new = out[2,snaps[i]÷10+1]/((snaps[i]+3290)÷10+1)           #Actually this is 10*dt.
    print(dt_new)
    out[2,snaps[i]÷10+1:end]+=(lastCorrectTime+dt_new-out[2,snaps[i]÷10+1])
end
tmp2 = out[2,2:end] - out[2,1:end-1]

lamda = out[3,:]./Rec.*sqrt.(15 ./out[5,:])
Rlambda = out[6,:].*out[3,:].^2 .*sqrt.(15 ./out[5,:])
# Mt = uc/cc*out[3,:]
epsilon = @. out[5,:]*uc^3/out[6,:]/l
nu = @. muc/rhoc/out[6,:]
eeta = @. (nu^3/epsilon)^0.25
epsilon0 = @. 15*nu*uc^2/l^2*out[12,:]

using Plots, LaTeXStrings
# gr()
# plotly()
# using PlotlyJS
p1 = plot(out[2,:],out[3,:]*uc)
xlabel!(p1,L"Dimensionless\:time\:(t/t_c)")
title!(p1,"urms (m/s)")
display((p1))

p1 = plot(out[2,:],out[6,:]*rhoc)
title!(p1,L"\mathrm{\rho_{mean}\:(kg/m^3)}")
display((p1))

p1 = plot(out[2,:],out[9,:]*uc, label=L"\mathrm{\overline{u}_x}")     #"u\u0305_x"
plot!(out[2,:],out[10,:]*uc, label=L"\mathrm{\overline{u}_y}")
plot!(out[2,:],out[11,:]*uc, label=L"\mathrm{\overline{u}_z}")
xlabel!(p1,L"\textrm{Dimensionless\:time\:(t/t_c)}")
title!(p1,"u\u0305 (m/s)")
# plt.legend()
display((p1))

p1 = plot(out[2,:],epsilon/1e3)
xlabel!(p1,L"\textrm{Dimensionless\:time\:(t/t_c)}")
# xlabel!(p1,"No. of timesteps")
title!(p1,"Energy diffusion rate, \u03b5 (kJ/kg/s)")
# title!(p1,"epsilon (m\u00b2/s\u00b3)")
display((p1))

p1 = plot(out[2,:],out[7,:]*1e6/(8*pi^3))
title!(p1,"KE (\u03bcJ)")
xlabel!(p1,L"\textrm{Dimensionless\:time\:(t/t_c)}")
display((p1))

p1 = plot(out[2,:],out[8,:]*1e6/(8*pi^3))
title!(p1,"IE (\u03bcJ)")
xlabel!(p1,L"\textrm{Dimensionless\:time\:(t/t_c)}")
display((p1))

p1 = plot(out[1,:],(out[8,:]+out[7,:])*1e6/(8*pi^3))
title!(p1,"TE (\u03bcJ)")
xlabel!(p1,"No. of timesteps")
display((p1))

p1 = @. scatter(out[1,:],out[21,:]/out[12,:]^1.5, label = "Sdudx")
# plt.scatter(out[1,:],out[22,:]/out[13,:]^1.5, label = "Sdudy")
# plt.scatter(out[1,:],out[23,:]/out[14,:]^1.5, label = "Sdudz")
# plt.scatter(out[1,:],out[24,:]/out[15,:]^1.5, label = "Sdvdx")
# plt.scatter(out[1,:],out[25,:]/out[16,:]^1.5, label = "Sdvdy")
# plt.scatter(out[1,:],out[26,:]/out[17,:]^1.5, label = "Sdvdz")
# plt.scatter(out[1,:],out[27,:]/out[18,:]^1.5, label = "Sdwdx")
# plt.scatter(out[1,:],out[28,:]/out[19,:]^1.5, label = "Sdwdy")
# plt.scatter(out[1,:],out[29,:]/out[20,:]^1.5, label = "Sdwdz")
title!(p1,"Vel. Derivative Skewness")
# plt.legend()
display((p1))

# plt.scatter(out[3,:],out[30,:]/out[12,:]^2)
p1 = @. scatter(out[4,:]^2*Rlambda,out[30,:]/out[12,:]^2)
title!(p1,L"\textrm{K_{dudx}}")
xlabel!(p1,L"\mathrm{M_t^2R_\lambda}")
display((p1))

p1 = plot(out[2,:],Rlambda)
title!(p1,L"\textrm{R_\lambda}")
xlabel!(p1,L"\textrm{Dimensionless\:time\:(t/t_c)}")
display((p1))

p1 = plot(out[2,:],out[4,:]*Rlambda)
title!(p1,L"\mathrm{M_tR_\lambda}")
xlabel!(p1,L"\textrm{Dimensionless\:time\:(t/t_c)}")
display((p1))

p1 = plot(out[2,:],l./eeta)
title!(p1,"L/\u03b7")
xlabel!(p1,L"\textrm{Dimensionless\:time\:(t/t_c)}")
display((p1))

p1 = plot(out[2,:],out[39,:], label="<u\u2081.u\u2082 >")        #*uc^2
plot!(out[2,:],out[40,:], label="<u\u2082.u\u2083 >")
plot!(out[2,:],out[41,:], label="<u\u2081.u\u2083 >")
title!(p1,"Velocity correlations (dimless)")
# title!(p1,"Velocity correlations $(m^2/s^2)$")
xlabel!(p1,L"\textrm{Dimensionless\:time\:(t/t_c)}")
display((p1))
