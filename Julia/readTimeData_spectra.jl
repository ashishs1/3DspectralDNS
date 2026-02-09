# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:43:52 2022

@author: singh
"""

using Plots
using Serialization
function save(filename,array)
    open(filename,"w") do io
        for value in array serialize(io,value) end
    end
end
function load(filename)
    array = []
    open(filename,"r") do io
        try while true append!(array,deserialize(io)) end
        catch end
    end
    return array
end

cd("G:\\MTech Project\\Julia\\test\\")
kmax = 11
N = 3*kmax - 1
l, rhoc, pc, muc, kc, gamma, R = 0.0005, 0.1664, 1e5, 1.96e-5*1.1, 15.34e-2, 1.67, 2079         #Indicative values of He@NTP (SI units: [m, kg/m3, Pa, kg/ms, W/mdegC, dimless, J/kgK])
urms = kmax^(4/3)*muc/rhoc/l   #Intended urms in SI units (Since n/L ~ 1/kmax)
cc = sqrt(gamma*pc/rhoc)        #Characteristic speed of sound, SI units
Mc = urms/cc                  #For TGV, intitially, urms~uc/2. So, for Mt~0.03, Mc~0.06.
uc, Tc = Mc*cc, pc/rhoc/R
Rec = rhoc*uc*l/muc
# const kx::Array{Int16,1} = append!(collect(0:(div(N,2)-1)),collect(-div(N,2):-1))
# const kz::Array{Int16,1} = collect(0:div(N,2))
# K = permutedims(stack(collect(Iterators.product(kz, kx, kx))),(2,3,4,1))
# K2 = dropdims(mapreduce(x->Int32(x)^2, +, K; dims=4), dims=4)
# K = nothing
# K1d = collect(Iterators.flatten(sqrt.(K2)))
# K2 = nothing
# const Nk = ceil(Int,sqrt(3)*N/2)
# sections = range(0.0, stop=sqrt(3)*N/2, length=Nk)
# ixbinN = Vector{Int}(undef,0)
# for i in 1:Nk-1 append!(ixbinN,mapreduce(x -> sections[i]<x<=sections[i+1], +, K1d)) end
# save("ixbinN.bin",ixbinN)
ixbinN2 = load("ixbinN.bin")

# file = "spectra_0145908.npz"
# # file = "spectra_0312310.npz"
# loaded = np.load(file)
# k_plot = loaded["arr_0"][:plotRange]
# spectarray = loaded["arr_1"][1][:plotRange]
# time = loaded["arr_2"]
# loaded.close()
# fig = plt.figure(figsize=(18,12))
# plt.loglog(k_plot, spectarray, "b")
# plt.loglog(k_plot[1:kmax+3], spectarray[2]*(k_plot[1:kmax+3]/k_plot[2])^(-7/3), "b--")
# plt.title("Time = "+np.format_float_scientific(time,precision=2))
# plt.xlabel("|K|")
# plt.ylabel("$E_K$")
# # plt.yscale("log")
# # plt.xscale("log")
# tick_pos = [1, 10, 16]
# labels = ["1/L", "10", "$k_{max}$"]
# plt.xticks(tick_pos, labels)
# # plt.legend()
# plt.ylim([1e3,1e16])
# plt.show()


######-----Read the Simulation Output Data-----######
pressure = false
ampIndex = 8            #Index for deciding position of slope
counter=0
list = readdir()
for file in list[end:-1:1]
    global amplitude
    if file[end-3:end]!=".bin" || file[1:7]!="spectra" continue end
    loaded = load(file)
    l1 = Int(length(loaded)/4)          #Length of each spectra stored in the file
    if pressure amplitude = loaded[3*l1+ampIndex]
    else amplitude = loaded[l1+ampIndex]
    end
    break
end

plotRange = maximum([floor(Int,kmax*3/2) - 3, kmax+3])

for file in readdir()
    if file=="k_for_plot.bin"
        loaded = load(file)
        global k_plot = loaded[1:plotRange]
        global slopeArray = (amplitude/N^3/ixbinN2[ampIndex]*k_plot[ampIndex]^2).*(k_plot[2:kmax+3]./k_plot[ampIndex]).^(-2)
        break
    end
end

if pressure path = pwd()*"\\pressure\\"
else path = pwd()
end
# for file in sort(readdir(path))
#     global pngsDone = 0
#     global images = []
#     if file[end-3:end]==".png"
#         try number = Int(file[1:end-4])
#         catch continue
#         end
#         pngsDone+=1
#         if length(images)==number images.append(Image.open(file))
#         elseif length(images)>number images[number] = Image.open(file)
#         else
#             while length(images)<number images.append(None) end
#             images.append(Image.open(file))
#         end
#     end
# end

using LaTeXStrings
# fig = plt.figure(figsize=(18,12))
anim = @animate for file in sort(readdir())
    global counter
    if length(file)<17 || file[end-9:end]!="_julia.bin" || file[1:7]!="spectra" continue end
    counter+=1
    # if counter<=pngsDone continue end
    loaded = load(file)
    l1 = Int(length(loaded)/4)          #Length of each spectra stored in the file
    # println(file)
    if pressure spectarray = loaded[3*l1+1:3*l1+plotRange]./ixbinN2[1:plotRange].*k_plot.^2/N^3
    else spectarray = [loaded[i]+loaded[l1+i]+loaded[2*l1+i] for i in 1:plotRange]./ixbinN2[1:plotRange].*k_plot.^2/N^3
    end
    # time = loaded["arr_2"]
    tstep = parse(Int,file[9:13])
    p1=plot(k_plot, spectarray, linecolor=:blue, xaxis=:log, yaxis=:log)
    plot!(k_plot[2:kmax+3], slopeArray, linecolor=:blue, linestyle=:dash, xaxis=:log, yaxis=:log)
    title!(p1, "Timestep = $tstep")
    xlabel!(p1,"|K|")
    if pressure ylabel!(L"P_K")
    else ylabel!(L"E_K")
    end
    tick_pos = [1, 10, kmax]
    labels = ["1", "10", L"k_{max}"]
    xticks!(p1, tick_pos, labels)
    if !pressure ylims!((1e-9,1e7))
    else ylims!((1e-6,1e3))
    end
    display(p1)
end

gif(anim, "EnergySpectrum_julia.gif", fps = 15)

# cd(path)
# images[0].save("EnergySpectrum.gif", save_all=True, append_images=images[1:], duration=300, loop=0)
# del fig
# plt.show()

##To estimate how much time it will take for energy spectrum to reach dissipation scales
# ks = []
# ts = []
# for file in readdir():
#     if file[-4:]!=".npz" or file[0:7]!="spectra": continue
#     loaded = np.load(file)
#     spectarray = sum(loaded["arr_0"][:3,:plotRange],axis=0)/ixbinN2[:plotRange]*k_plot^2
#     loaded.close()
#     temp = abs(spectarray - 1e-6)
#     idx = temp.argmin()
#     if spectarray[idx]>1e-6:idx2 = idx+1
#     else: idx2 = idx-1
#     kind = (temp[idx]*k_plot[idx2]+temp[idx2]*k_plot[idx])/(temp[idx]+temp[idx2])       #k indicative of 1e-6 value of spectrum
#     ks.append(kind)
#     ts.append(int(file[8:13]))
#
# ks = np.array(ks)
# ts = np.array(ts)
# coeffs = np.polyfit(log(ks),ts,1)
# l_by_n = 235.5              #Max k after which spectrum should fall sharply
# t_exp = 0
# for i in range(length(coeffs)): t_exp = t_exp*log(l_by_n) + coeffs[i]
# print("The spectrum is expected to mature after",round(t_exp),"timesteps.")
# plt.figure()
# plt.plot(log(ks),ts)
# ts_fit = ks*0
# for i in range(length(coeffs)):
#     ts_fit = ts_fit*log(ks) + coeffs[i]
# plt.plot(log(ks),ts_fit)
# plt.show()
