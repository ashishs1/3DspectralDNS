#Test file - for speed comparison with Python

##---NOTES:
##- Most efficient way of filling an array is a .= foobar, where foobar uses element-wise operations (eg. a .= b.+c). This is measurably better than a[:] = foobar.
##- In non-element-wise operations cases (eg. a=fft(b)), a[:]=foobar is very slightly better than a.=foobar in terms of memory (16bytes less).
##- In case of copying, a[:] = b is very slightly better than a.=b.
##- Most efficient way of operating on the elements of an array (with +,-,* etc) is a .+= foobar. This is much better than a[:] += foobar[:].
##- In addition to above point, if operations are done in parts (like for vector array U,M etc.), then it is much faster & memory efficient to use view(U,i,:,:,:) .*= foobar instead of U[i,:,:,:] .*= foobar.
##- Similarly, during computations also, slicing arrays using view() is more efficient than slicing through index & colons (as it creates a temporary copy).
##- It is better to keep the direction index of vector arrays at dim=4. This is because U[i,:,:,:] is non-contiguous and gives issues with fftw & MPI, where as U[:,:,:,1] is contiguous (and thus, can be used with mul!). This has something to do with julia being column-major while Python being row major.
##- It is also observed that element-wise operations are much faster for view(U,:,:,:,i) than for view(U,i,:,:,:).
##- In continuation to above, MPI.Alltoall will also require the Np sized dimension to be at the end. So, either permutedims is used repeatedly for every fft, or x,y,z indices are changed to y,z,x, or z,x,y with slicing on y.
##- Functions work the fastest when global arrays used/modified by it are passed as arguments. This is not true in Python, where too many input arguments slow down the function. CORRECTION: The increase in speed of function due to maximum arguments is unreliable (@btime sometimes reports no argument function as faster).

# set_zero_subnormals(true)   #May improve performance
macro fast(forloop); return esc(:(@inbounds @simd ivdep $forloop)); end    #Short form for SIMD for loops. Seems to be slightly faster than writing the expanded form
# using LoopVectorization
# macro fast2(forloop); return esc(:(@turbo warn_check_args=false $forloop)); end
const newaxis = [CartesianIndex()]
using FFTW
using MPI
MPI.Init()
const comm = MPI.COMM_WORLD
const rank  = MPI.Comm_rank(comm)
const num_processes = MPI.Comm_size(comm)

function printfile(x...)
    open("out.julia","a") do io   #joinpath(simName,"out.julia")
        for tmp in x print(io,tmp) end
        println(io)
    end
end
# printfile("My rank is ",rank," and I am 1 of ",num_processes," processes.")     #To check that MPI is working fine

simName = "G:\\MTech Project\\Julia\\test\\"
# simName = "/mnt/g/MTech\ Project/Julia/test/"
restartRun = false  #To decide if tsfile needs to be overwritten
const kmax = 11
const N = 3*kmax-1        #Try to keep this as a power of 2.
#The maximum wavenumber considered for calculating Fourier coeffs of non-linear terms is N//2. This should be >= 3/2*kmax-1. Hence, N>=3*kmax-2.

##Characteristic parameters
const l, rhoc, pc, muc, kc, gamma, R = 0.0005, 0.1664, 1e5, 1.96e-5*1.1, 15.34e-2, 1.67, 2079       #Indicative values of He@NTP (SI units: [m, kg/m3, Pa, kg/ms, W/mdegC, dimless, J/kgK])
const urms = kmax^(4/3)*muc/rhoc/l   #Intended urms in SI units (Since n/L ~ 1/kmax)
const cc = sqrt(gamma*pc/rhoc)        #Characteristic speed of sound, SI units
const Mc = urms/cc                  #Consider uc=urms.
const uc, Tc = Mc*cc, pc/rhoc/R
const Rec = rhoc*uc*l/muc
const Prc = muc*gamma*R/(gamma-1)/kc

const Np = div(N,num_processes)
U::Array{Float64,4} = Array{Float64}(undef,N,N,Np,3)    # empty((3, Np, N, N))        #Dimensional field is U*uc
P::Array{Float64,3} = Array{Float64}(undef,N,N,Np)    # empty((Np, N, N))           #Dimensional field is P*(gamma*pc)
rho::Array{Float64,3} = Array{Float64}(undef,N,N,Np)    # empty((Np, N, N))         #Dimensional field is rho*rhoc
U_hat::Array{ComplexF64,4} = Array{ComplexF64}(undef, div(N,2)+1, Np, N, 3)
M_hat::Array{ComplexF64,4} = Array{ComplexF64}(undef, div(N,2)+1, Np, N, 3)
P_hat::Array{ComplexF64,3} = Array{ComplexF64}(undef, div(N,2)+1, Np, N)
rho_hat::Array{ComplexF64,3} = Array{ComplexF64}(undef, div(N,2)+1, Np, N)
#The following 2 variables help in speed-up using the mutating mul! & permutedims!.
Uc_hat::Array{ComplexF64,3} = Array{ComplexF64}(undef,div(N,2)+1,Np,N)
Uc_hatT::Array{ComplexF64,3} = Array{ComplexF64}(undef,div(N,2)+1,N,Np)
const kx::Array{Int16,1} = append!(collect(0:(div(N,2)-1)),collect(-div(N,2):-1))      # fftfreq(N, 1. /N)
const kz::Array{Int16,1} = collect(0:div(N,2))
const K::Array{Int16,4} = permutedims(stack(collect(Iterators.product(kz, kx[rank*Np+1:(rank+1)*Np], kx))),(2,3,4,1))

# const K2 = dropdims(sum(K.^2, dims=4),dims=4)   #sum function doesn't reduce ndims
const K2 = dropdims(mapreduce(x->Int32(x)^2, +, K; dims=4), dims=4)    #faster than sum of .^2.

##Memory efficient way of dealiasing (Size of storage ~size(P)/124):
const dealias = (abs.(K[:,:,:,1]) .<= kmax).*(abs.(K[:,:,:,2]) .<= kmax).*(abs.(K[:,:,:,3]) .<= kmax)
function dealias!(A::Union{Array{ComplexF64, 3},SubArray{ComplexF64, 3}})
    @fast for i ∈ eachindex(dealias) A[i] *= dealias[i] end
end
##Computationally efficient way of dealiasing (Speed up by ~3x for single core; Size of storage ~size(P)/3):
##-Note: The following doesn't speed up the code much for higher number of cores (maybe because there are some cores where dealiasList contains all its indices)
# const dealiasList = findall(iszero,collect(Iterators.flatten((abs.(K[:,:,:,1]) .<= kmax).*(abs.(K[:,:,:,2]) .<= kmax).*(abs.(K[:,:,:,3]) .<= kmax))))
# function dealias!(A::Union{Array{ComplexF64, 3},SubArray{ComplexF64, 3}})
#     @fast for i ∈ dealiasList A[i] = 0.0 + 0.0im end #@turbo doesn't allow for loops over vectors (only over ranges)
# end

function dealias!(A::Array{ComplexF64, 4})
    @inbounds for i in 1:3 dealias!(view(A,:,:,:,i)) end
end

const planrfft = plan_rfft(P, (1, 2); flags=FFTW.PATIENT)   #; flags=FFTW.MEASURE/PATIENT, timelimit=Inf
const planfft = plan_fft(Array{ComplexF64}(undef,div(N,2)+1,Np,N), (3,); flags=FFTW.PATIENT)
import LinearAlgebra.mul!
using Strided   #Speeds up the permutedims! line
function fftn_mpi(u, fu)
    mul!(Uc_hatT,planrfft,u)
    @strided permutedims!(reshape(Uc_hat,(div(N,2)+1, Np, Np, num_processes)),reshape(Uc_hatT,(div(N,2)+1, Np, num_processes, Np)),(1,2,4,3))  #In python, Uc_hat
    MPI.Alltoall!(UBuffer(Uc_hat, Np*Np*(div(N,2)+1), nothing, MPI.Datatype(ComplexF64)), comm)
    mul!(fu, planfft, Uc_hat)
    return nothing
end

const planifft = plan_ifft(P_hat, (3,); flags=FFTW.PATIENT)
const planirfft = plan_irfft(Array{ComplexF64}(undef,div(N,2)+1,N,Np), N, (1,2); flags=FFTW.PATIENT)
function ifftn_mpi(fu, u)
    mul!(Uc_hat,planifft,fu)
    MPI.Alltoall!(UBuffer(Uc_hat, Np*Np*(div(N,2)+1), nothing, MPI.Datatype(ComplexF64)), comm)
    @strided permutedims!(reshape(Uc_hatT,(div(N,2)+1, Np, num_processes, Np)),reshape(Uc_hat,(div(N,2)+1, Np, Np, num_processes)),(1,2,4,3))
    mul!(u, planirfft, Uc_hatT)
    return nothing
end
##----------------------------------------------------------------##

#########------------INITIAL CONDITIONS STARTED----------###########
v = Array{Float64}(undef, N, N, Np, 3)        #Divergence free base flow (dimensionless)
vkvjk = Array{Float64}(undef, N, N, Np, 3)
#NOTE: To set up any IC with first order solution, populate v, rho, P and vkvjk matrices only.

#(5) Multi-scale random velocity field
kp = 5          #Scale with peak energy
A = 16/kp^5/sqrt(pi/2)*(2*pi)^3
@strided begin
    Umag = A*K2.*exp.(-2*K2/kp^2)/pi
    phi = rand(Float64, (div(N,2)+1, Np, N, 3)).*(2*pi)        #Random phases of U_hats
    ##Following loop is for direct comparison with Python results.
    for i in 1:3
        for j in 1:N
            for k in 1:Np
                for l in 1:div(N,2)+1
                    phi[l,k,j,4-i] = (i-2)*(j-(N+1)/2)*(k+rank*Np-(N+1)/2)*(l-div(N,4)-1)*16/3/N/N/(div(N,2)+1)   #4-i is used because this definition of phi is to compare results with Python, where dimensions of vectors are reversed (x-y-z to z-y-x)
                end
            end
        end
    end
    temp3 = -ifelse.(K .== 0, 0., sin.(cat(phi[:,:,:,newaxis,2]-phi[:,:,:,newaxis,3], phi[:,:,:,newaxis,3]-phi[:,:,:,newaxis,1], phi[:,:,:,newaxis,1]-phi[:,:,:,newaxis,2],dims=4)))./ifelse.(K .== 0, 1, K) #Multiplied by -ve sign because for comparison (with Python) purpose, the coordinate system is assumed to be left-handed here.
    temp2 = dropdims(mapreduce(x->x^2, +, temp3; dims=4), dims=4)        #Denominator of U_hat magnitudes
    temp2 = ifelse.(temp2.==0, 0., 1.0./temp2)
    for i in 1:3 view(temp3,:,:,:,i) .*= sqrt.(Umag.*temp2) end
    v_hat = temp3.*(cos.(phi)+1im*sin.(phi))*N^3        #Multiplied by N^3 because of backward normalisation of scipy fft
end
Umag = phi = temp3 = temp2 = nothing
dealias!(v_hat)
for i in 1:3 ifftn_mpi(view(v_hat,:,:,:,i), view(v,:,:,:,i)) end
urms_actual = [mapreduce(x->x^2, +, v)/3/N^3]
MPI.Allreduce!(urms_actual, MPI.SUM, comm)
urms_actual = sqrt(urms_actual[1])
vfactor = 1/urms_actual
# if rank==0
#     printfile("Required to increase urms by a factor of ",round(vfactor,sigdigits=6))
#     printfile("urms required is ",round(urms,sigdigits=6),". Actual dimless urms in IC is ", urms_actual,".")
# end
v *= vfactor
v_hat *= vfactor
vjk = Array{Float64}(undef, N, N, Np)
for j in 1:3
    vkvjk[:,:,:,j] .= 0.0
    # let vjk = Array{Float64}(undef, N, N, Np)
    for k in 1:3
        ifftn_mpi(view(K,:,:,:,k).*view(v_hat,:,:,:,j), vjk)
        view(vkvjk,:,:,:,j) .+= view(v,:,:,:,k).*vjk
    end
    # end
end

v_hat = vjk = urms_actual = sumdata = vfactor = nothing

##--Solve ICs upto first order approximation as in Ristorcelli (1997)--##
ep2 = gamma*(urms/cc)^2        #Epsilon square
p1_hat = zeros(ComplexF64, div(N,2)+1, Np, N)
let
    vi2_hat = Array{ComplexF64}(undef, div(N,2)+1, Np, N)
    vivj_hat = Array{ComplexF64}(undef, div(N,2)+1, Np, N)
    for i in 1:3
        fftn_mpi(view(v,:,:,:,i).^2, vi2_hat)
        fftn_mpi(view(v,:,:,:,i%3+1).*view(v,:,:,:,(i+1)%3+1), vivj_hat)
        p1_hat .-= (view(K,:,:,:,i).^2 .*vi2_hat + 2*view(K,:,:,:,i%3+1).*view(K,:,:,:,(i+1)%3+1).*vivj_hat)
    end
end
p1 = Array{Float64}(undef, N, N, Np)
dealias!(p1_hat)
p1_hat ./= ifelse.(K2 .== 0, 1, K2)
ifftn_mpi(p1_hat, p1)
@fast for i in eachindex(rho)
    rho[i] = (1 + (ep2/gamma)*p1[i])
    P[i] = (1 + ep2*p1[i])/gamma             #Since P is non-dimensionalised w.r.t. gamma*pc
end
p1 = nothing      #Pressure and density terms are found till first order. Now, find velocity.
p1t_hat = zeros(ComplexF64, div(N,2)+1, Np, N)
d_hat = zeros(ComplexF64, div(N,2)+1, Np, N)
let p1i = Array{Float64}(undef, N, N, Np)
    for i in 1:3
        ifftn_mpi(-1im*view(K,:,:,:,i).*p1_hat, p1i)
        let 
            vjp1i_hat = Array{ComplexF64}(undef, div(N,2)+1, Np, N)
            vivkvjk_hat = Array{ComplexF64}(undef, div(N,2)+1, Np, N)
            for j in 1:3
                fftn_mpi(view(v,:,:,:,i).*view(vkvjk,:,:,:,j), vivkvjk_hat)
                fftn_mpi(view(v,:,:,:,j).*p1i, vjp1i_hat)
                if i==j d_hat .-= vjp1i_hat end
                p1t_hat .+= view(K,:,:,:,i).*view(K,:,:,:,j).*(vivkvjk_hat + vjp1i_hat)
            end
        end
    end
end
dealias!(p1t_hat)
p1t_hat .*= 2 ./ifelse.(K2 .== 0, 1, K2)
d_hat .-= p1t_hat
dealias!(d_hat)
d_hat ./= gamma
p1_hat = p1t_hat = vjp1i_hat = p1i = vkvjk = vivkvjk_hat = nothing
w = Array{Float64}(undef, N, N, Np, 3)
for i in 1:3 ifftn_mpi(-1im*view(K,:,:,:,i).*d_hat./ifelse.(K2 .== 0, 1, K2), view(w,:,:,:,i)) end
U .= (v .+ ep2.*w).*(urms/uc)        #Non-dimensionalised w.r.t. uc.
d_hat = w = v = nothing

##--Transform initial conditions into the Fourier space - DO NOT COMMENT OUT--##
for i in 1:3
    fftn_mpi(view(U,:,:,:,i), view(U_hat,:,:,:,i))
    fftn_mpi(view(U,:,:,:,i).*rho, view(M_hat,:,:,:,i))
end
fftn_mpi(rho, rho_hat)
fftn_mpi(P, P_hat)
###-------------------INITIAL CONDITIONS DONE-------------------###

# if rank==0 printfile(U_hat[7,5,4,:],",",P[4,9,4:10]) end

###-------------------DEFINE USEFUL FUNCTIONS-------------------###
temp_hat::Array{ComplexF64,3} = Array{ComplexF64}(undef, div(N,2)+1, Np, N)    #Temporarily stores rhouuij term in Mom eqns, and 2 pu terms & p diffusion term in energy eqn
temp::Array{Float64,3} = Array{Float64}(undef, N, N, Np)                #Stores pdiff term, abs(sqrt_rho_Ui), Ek
# duidxi = Array{Float64}(undef, N, N, Np, 3))           #Also temporarily stores duidxi-1 and dui-1dxi. 9 variables are now reduced to 3.
dpdxi::Array{Float64,3} = Array{Float64}(undef, N, N, Np)
# Sdudx::Array{Float64,3} = Array{Float64}(undef,3,3,3)      #Stores 2nd, 3rd & 4th order moments of du/dx in all directions.
const N3 = N^3
const tmp = N*Np*(div(N,2)+1)   #For Fourier space arrays
const tmpi = N*N*Np     #For Cartesian space arrays (reduce(*,size(U)[1:3]))

if rank==0
    time_series_file = simName*"timeSeries_julia.bin"
    tsfile = restartRun ? open(time_series_file, "a") : open(time_series_file, "w")
end
using Serialization
storageVector::Array{Float64,1} = Array{Float64}(undef, 39) #storageVector will be ignored in non-root procs, but it is still passed into function, so needs to be defined as at least an empty array.
#NOTE: MPI.Reduce! with arguments (sendbuf, recvbuf, op, comm) is not working. Using storageVector[some indices] = MPI.Reduce is also problematic as non-root procs will have to have Nothing type storageVector, which will add too many if's.
function saveParams(rho, P, t, tstep)
    #Saves in order: tstep, t, urms, Mt, epsilon, rho_mean, KE, IE, U_avgs, (dudx)^2s (in the order: u1x1, u2x1, u3x1, u1x2,...), (dudx)^3s, (dudx)^4s, uiujs
    #urms, Mt & uiuj:
    if rank==0
        serialize(tsfile,tstep)
        serialize(tsfile,t)
    end
    storageVector[7:9] = sum(U, dims=(1,2,3))/N3            #Spatial average of U. Should be close to zero.
    # MPI.Reduce!(view(storageVector,7:9), MPI.SUM, comm)
    # temp[:] = mapreduce(x->x^2, +, U; dims=4)
    @fast for j in 1:tmpi temp[j] = U[j]^2 + U[j+tmpi]^2 + U[j+2*tmpi]^2 end    #~3x faster than mapreduce
    temp2 = sum(temp)/(3*N3)
    temp .*= rho./P
    storageVector[1:2] = [temp2, sum(temp)/(gamma*3*N3)]
    # MPI.Reduce!(view(storageVector,1:2), MPI.SUM, comm)
    # storageVector[37] = sum(view(U,:,:,:,1).*view(U,:,:,:,2))/N3
    # storageVector[38] = sum(view(U,:,:,:,2).*view(U,:,:,:,3))/N3
    # storageVector[39] = sum(view(U,:,:,:,1).*view(U,:,:,:,3))/N3   #This method is not very slow, but takes up lot of memory
    storageVector[37:39] = [0.,0.,0.]
    @fast for i in 1:tmpi
        j,k = i+tmpi,i+2*tmpi
        storageVector[37] += U[i]*U[j]
        storageVector[38] += U[j]*U[k]
        storageVector[39] += U[i]*U[k]
    end #SIMD loop gives correct result only if U is not included in the input arguments. >1.5x faster than above method. NOTE: There is small difference b/w sum & looped + (probably due to the nature of sum function), but it is very small.
    storageVector[37:39] ./= N3            #uiujs done. Should be close to zero for isotropy
    temp .= 0.0
    for i in 1:3
        tmp2 = (i-1)*tmp
        @fast for j in eachindex(temp_hat) temp_hat[j] = 1im*K[tmp2+j]*U_hat[tmp2+j] end
        ifftn_mpi(temp_hat, dpdxi)
        @fast for j in eachindex(temp) temp[j] += dpdxi[j] end
        indices = [6+4*i,15+4*i,24+4*i]     #indices = 9 + 3*(i-1) + i, 18+.. & 27+..
        storageVector[indices] = mapreduce(x->[x^2,x^3,x^4], +, dpdxi)/N3
        # for idx in eachindex(indices) MPI.Reduce!(view(storageVector,idx:idx), MPI.SUM, comm) end    #Reduce! can only work on strided arrays/subarrays, so it is needed to be done element wise, as indices are distributed
        # Sdudx[0,i,i] = MPI.Allreduce([temp2], MPI.SUM, comm)[1]
        # # # temp2 = mapreduce(x->x^3, +, dpdxi)/N3
        # # # MPI.Reduce!([temp2], storageVector[15+4*i:15+4*i], MPI.SUM, comm)
        # Sdudx[1,i,i] = MPI.Allreduce([temp2], MPI.SUM, comm)[1]
        # Sdudx[2,i,i] = MPI.Allreduce([temp2], MPI.SUM, comm)[1]
    end
    del2 = mapreduce(x->x^2, +, temp)/N3           #Spatial average of square of divU
    temp3 = 0.0
    for i in 1:3
        i2 = (i+1)%3+1
        tmp2, tmp3 = (i-1)*tmp, (i2-1)*tmp
        @fast for j in eachindex(temp_hat) temp_hat[j] = 1im*K[tmp3+j]*U_hat[tmp2+j] end
        ifftn_mpi(temp_hat, dpdxi)      #Read dpdxi as duidxi-1
        @fast for j in eachindex(temp_hat) temp_hat[j] = 1im*K[tmp2+j]*U_hat[tmp3+j] end
        ifftn_mpi(temp_hat, temp)       #Read temp as dui-1dxi
        indices = [6+3*i2+i,15+3*i2+i,24+3*i2+i]
        storageVector[indices] = mapreduce(x->[x^2,x^3,x^4], +, dpdxi)/N3
        temp3 += storageVector[indices[1]]
        # for idx in eachindex(indices) MPI.Reduce!(view(storageVector,idx:idx), MPI.SUM, comm) end
        indices = [6+3*i+i2,15+3*i+i2,24+3*i+i2]
        storageVector[indices] = mapreduce(x->[x^2,x^3,x^4], +, temp)/N3
        temp3 += storageVector[indices[1]]
        # for idx in eachindex(indices) MPI.Reduce!(view(storageVector,idx:idx), MPI.SUM, comm) end
        @fast for j in eachindex(temp) temp[j] *= dpdxi[j] end  #Destroy temp as it is not needed anymore
        temp3 -= (2/N3)*sum(temp)
    end
    #temp3 now contains Spatial average of square of vorticity magnitude, ω2.
    @fast for j in 1:tmpi temp[j] = rho[j]*(U[j]^2 + U[j+tmpi]^2 + U[j+2*tmpi]^2) end
    storageVector[3:6] = [(4/3*del2 + temp3)/Rec, sum(rho)/N3, rhoc*uc^2*l^3*sum(temp)/2*(8*pi^3)/N3, gamma*pc*l^3*sum(P)/(gamma-1)*(8*pi^3)/N3]
    #temp2 = [Dimensionless KE dissipation, epsilon, Kida & Orszog, 1990, rho_mean (dimless), Dimensional KE, Dimensional IE]
    # MPI.Reduce!(view(storageVector,3:6), MPI.SUM, comm)
    MPI.Reduce!(storageVector, MPI.SUM, comm)
    #Other parameters that can be computed post simulation also:
    # lamda = urms/Rec*sqrt(15/epsilon)
    # Rl = rho_mean*urms^2*sqrt(15/epsilon)
    # nu = muc/rho_mean
    #Print the above parameters to file:
    if rank==0
        storageVector[1] = sqrt(storageVector[1])   #Rms of fluctuating velocity, urms. U_avg assumed as zero.
        storageVector[2] = uc/cc*sqrt(storageVector[2])     #Mt
        for value in storageVector serialize(tsfile,value) end
        flush(tsfile)
    end
end

K1d = collect(Iterators.flatten(sqrt.(K2)))
const Nk = ceil(Int,sqrt(3)*N/2)
sections = range(0.0, stop=sqrt(3)*N/2, length=Nk)
temp2 = Vector{Vector{Int32}}(undef,0)          #Pre-calculate list of indices in each section. Size of this will be slightly more than 1/2th of that of K1d as int32 is being used, and headers of python list will increase some size.
for i in 1:Nk-1 append!(temp2,[findall(x -> sections[i]<x<=sections[i+1], K1d)]) end
const ixbin = temp2
spectra_k_plot = zeros(Nk - 1, 4)         #Stores spectrums of KE in each direction & IE
if rank==0
    k_plot = 0.5 * (sections[1:end-1] + sections[2:end])
    filename = simName * "k_for_plot.bin"
    open(filename,"w") do io   #joinpath(simName,"out.julia")
        # serialize(io,length(k_plot))     #For ease of reading, set the first byte as length of vector
        for k in k_plot serialize(io,k) end
    end
end     #k_plot is proc-independent, so can be written down by single proc
k_plot = sections = K1d = temp2 = nothing
using Printf
function saveSpectra(U, rho, P_hat, tstep)
    #Saves KE spectrum & pressure spectrum
    # temp[:] = 0.0
    for j in 1:3
        tmp2 = (j-1)*tmp
        # @fast for i in eachindex(temp) temp[i] = U[tmp2+i]*sqrt(rho[i]) end
        temp .= view(U,:,:,:,j).*sqrt.(rho)
        fftn_mpi(temp,temp_hat)      #Read temp_hat as sqrt_rho_Ui_hat
        # temp_hat[:,:,:N//2+1] /= N3       #Check if normalisation needed.
        tmp2 = (j-1)*(Nk-1)
        temp2 = (abs.(temp_hat)).^2
        @fast for i in 1:Nk-1 spectra_k_plot[tmp2+i] = sum(temp2[ixbin[i]]) end
        # temp[:,:,:N//2+1] += abs(temp_hat[:,:,:])^2      #Storing spectrum in different direction needed or not??
    end
    tmp2 = 3*(Nk-1)
    temp2 = (abs.(P_hat)).^2
    for i in 1:Nk-1 spectra_k_plot[tmp2+i] = sum(temp2[ixbin[i]]) end
    MPI.Reduce!(spectra_k_plot, MPI.SUM, comm)
    if rank == 0
        filename = joinpath(simName,@sprintf "spectra_%05d_julia.bin" tstep)
        open(filename,"w") do io
            for k in spectra_k_plot serialize(io,k) end
        end     #Length not stored here, as while reading, k_for_plot can be read first & length can be used from there.
    end
end

using HDF5
# @assert HDF5.has_parallel()
info = MPI.Info()
function saveSnapshot(U_hat, rho_hat, P_hat, tstep)
    filename = joinpath(simName,@sprintf "snapshot_%05d_julia.hdf5" tstep)
    #Parallel write to hdf5 file
    f = h5open(filename, "w", comm, info)
    dset = create_dataset(f, "U_hat", ComplexF64, (kmax+1,2*kmax+1,2*kmax+1,3))
    dset2 = create_dataset(f, "rho_hat", ComplexF64, (kmax+1,2*kmax+1,2*kmax+1))
    dset3 = create_dataset(f, "P_hat", ComplexF64, (kmax+1,2*kmax+1,2*kmax+1))
    if rank*Np<(kmax+1)
        temp3 = min(Np,kmax+1-rank*Np)
        dset[:,1+rank*Np:rank*Np+temp3,1:kmax+1,:] = U_hat[1:kmax+1,1:temp3,1:kmax+1,:]
        dset[:,1+rank*Np:rank*Np+temp3,end-kmax:end,:] = U_hat[1:kmax+1,1:temp3,end-kmax:end,:]
        dset2[:,1+rank*Np:rank*Np+temp3,1:kmax+1] = rho_hat[1:kmax+1,1:temp3,1:kmax+1]
        dset2[:,1+rank*Np:rank*Np+temp3,end-kmax:end] = rho_hat[1:kmax+1,1:temp3,end-kmax:end]
        dset3[:,1+rank*Np:rank*Np+temp3,1:kmax+1] = P_hat[1:kmax+1,1:temp3,1:kmax+1]
        dset3[:,1+rank*Np:rank*Np+temp3,end-kmax:end] = P_hat[1:kmax+1,1:temp3,end-kmax:end]
    end
    if (rank+1)*Np>(N-kmax)
        temp2 = 2*kmax + 1 + (rank+1)*Np - N
        temp3 = max(kmax+1,temp2-Np) + 1
        temp4 = temp3+Np-temp2
        dset[:,temp3:temp2,1:kmax+1,:] = U_hat[1:kmax+1,temp4:end,1:kmax+1,:]
        dset[:,temp3:temp2,end-kmax:end,:] = U_hat[1:kmax+1,temp4:end,end-kmax:end,:]
        dset2[:,temp3:temp2,1:kmax+1] = rho_hat[1:kmax+1,temp4:end,1:kmax+1]
        dset2[:,temp3:temp2,end-kmax:end] = rho_hat[1:kmax+1,temp4:end,end-kmax:end]
        dset3[:,temp3:temp2,1:kmax+1] = P_hat[1:kmax+1,temp4:end,1:kmax+1]
        dset3[:,temp3:temp2,end-kmax:end] = P_hat[1:kmax+1,temp4:end,end-kmax:end]
    close(f)
    end
end
##--------------------Useful Functions Done--------------------##

##----Some Optimisation Functions for Julia----##

function A_by_B!(A::Array{Float64, 4},B)
    t = 2*tmpi
    @fast for i ∈ eachindex(B) #Using @simd + @inbounds is very slightly faster than @turbo warn_check_args=false. Also, @turbo doesn't understand let.
        let tmp2 = 1/B[i]   #Division is not sped up by simd, but multiplication is!
            A[i] *= tmp2
            A[i+tmpi] *= tmp2
            A[i+t] *= tmp2
        end
    end
end
function RK4Update(Ahat1::Array{ComplexF64,3},a,dAhat::Array{ComplexF64,3})
    @fast for i in eachindex(Ahat1)
        Ahat1[i] += a*dAhat[i]
    end
end
function RK4Update(Ahat::Array{ComplexF64,3},Ahat0::Array{ComplexF64,3},a,dAhat::Array{ComplexF64,3})
    @fast for i in eachindex(Ahat)
        Ahat[i] = muladd(a, dAhat[i], Ahat0[i])
    end
end
function RK4Update(Ahat1::Array{ComplexF64,4},tmp2,a,dAhat::Array{ComplexF64,3})
    @fast for i in eachindex(dAhat)
        Ahat1[i+tmp2] += a*dAhat[i]
    end
end
function RK4Update(Ahat::Array{ComplexF64,4},Ahat0::Array{ComplexF64,4},tmp2,a,dAhat::Array{ComplexF64,3})
    @fast for i in eachindex(dAhat)
        Ahat[i+tmp2] = muladd(a, dAhat[i], Ahat0[i+tmp2])
    end
end
function momEqnLine1!(C,tmp2,P_hat,U_hat)     #element wise multiplication
    t = -1im*c1
    @fast for i ∈ eachindex(C)   #Here, @turbo is very slightly faster than @inbounds @simd.
        C[i] = muladd(-c2, muladd(K[i+tmp2]/3, muladd(K[i],U_hat[i], muladd(K[i+tmp],U_hat[i+tmp], K[i+2*tmp]*U_hat[i+2*tmp])), K2[i]*U_hat[i+tmp2]), t*K[i+tmp2]*P_hat[i])   #~5% faster than simple expression
    end
end
function tripleMult!(C::Array{Float64,3},A::Array{Float64,3},B::Array{Float64,4},i0,j)
    #This function solves C=A.*B[i0].*B[j]
    tmp2i = (i0-1)*tmpi
    tmp3i = (j-1)*tmpi
    @fast for i in eachindex(C)
        C[i] = A[i]*B[i+tmp3i]*B[i+tmp2i]
    end
end
##---------Optimisation functions done----------##

#RK4 coefficients:
const a = [1/6, 1/3, 1/3, 1/6]
const b = [0.5, 0.5, 1.]
#Non-dimensional N-S equations' coefficients:
const c1,c2,c3,c4 = 1/Mc^2, 1/Rec, (gamma-1)*Mc^2/Rec, gamma/Prc/Rec
#Find DNS timestep:
umax = [maximum(abs.(U[:,:,:,1])+abs.(U[:,:,:,2])+abs.(U[:,:,:,3]))]
MPI.Allreduce!(umax, MPI.MAX, comm)
const dt = 2*pi/N/(umax[1] + 3*cc/uc)         #Dimless time. To get dimensional time, multiply with (l/uc)
#Dealias ICs:
dealias!(U_hat); dealias!(M_hat); dealias!(rho_hat); dealias!(P_hat)

##--Allocate space for remaining variables--##
M_hat0::Array{ComplexF64,4} = Array{ComplexF64}(undef, div(N,2)+1, Np, N, 3)
M_hat1::Array{ComplexF64,4} = Array{ComplexF64}(undef, div(N,2)+1, Np, N, 3)
P_hat0::Array{ComplexF64,3} = Array{ComplexF64}(undef, div(N,2)+1, Np, N)
P_hat1::Array{ComplexF64,3} = Array{ComplexF64}(undef, div(N,2)+1, Np, N)
rho_hat0::Array{ComplexF64,3} = Array{ComplexF64}(undef, div(N,2)+1, Np, N)
rho_hat1::Array{ComplexF64,3} = Array{ComplexF64}(undef, div(N,2)+1, Np, N)
dU::Array{ComplexF64,3} = Array{ComplexF64}(undef, div(N,2)+1, Np, N)          #Stores timestep changes for rho_hat, M_hat, and P_hat
##------------------------------------------##

##----Define Solver Functions----##
function updateFields(rho,rho_hat,P,P_hat,U,U_hat,M_hat)
    #Update U_hat, P, rho and U, which will be required for computing RHS, or new timestep.
    ifftn_mpi(rho_hat, rho)
    for i in 1:3 ifftn_mpi(view(M_hat,:,:,:,i), view(U,:,:,:,i)) end
    A_by_B!(U,rho)         #Adds some aliasing errors, but they're expected to be negligible
    ifftn_mpi(P_hat, P)
    for i in 1:3 fftn_mpi(view(U,:,:,:,i),view(U_hat,:,:,:,i)) end
    dealias!(U_hat)
end

function contEqn(rk::Int8,dU,M_hat,rho_hat,rho_hat0,rho_hat1)
    #---Solve Continuity Eqn---#
    @fast for i ∈ eachindex(dU)
        dU[i] = -1im*muladd(K[i],M_hat[i], muladd(K[i+tmp],M_hat[i+tmp], K[i+2*tmp]*M_hat[i+2*tmp]))
    end
    RK4Update(rho_hat1,a[rk]*dt,dU)
    if rk < 4 RK4Update(rho_hat,rho_hat0,b[rk]*dt,dU) end     #Safe to update rho_hat as it is not needed in mom & energy eqns
end

function momEqn(rk::Int8,dU,rho,P_hat,U,U_hat,temp,temp_hat,M_hat,M_hat0,M_hat1)
    #---Momentum Eqns---#
    for i0 in 1:3
        # dU .= (-1im*c1).*view(K,:,:,:,i).*P_hat .- c2.*(K2.*view(U_hat,:,:,:,i) .+ view(K,:,:,:,i)./3 .*dropdims(sum(K.*U_hat, dims=4),dims=4))   #slow
        tmp2 = (i0-1)*tmp
        momEqnLine1!(dU,tmp2,P_hat,U_hat)
        for j in 1:3
            tmp3 = (j-1)*tmp
            tripleMult!(temp,rho,U,i0,j)
            fftn_mpi(temp, temp_hat)
            dealias!(temp_hat)
            @fast for i in eachindex(dU) dU[i] -= 1im*K[i+tmp3]*temp_hat[i] end    #RHS of ith Momentum equation is done when this loop ends
        end
        RK4Update(M_hat1,tmp2,a[rk]*dt,dU)
        if rk < 4 RK4Update(M_hat,M_hat0,tmp2,b[rk]*dt,dU) end
    end
end

function energyEqn(rk::Int8,dU,rho,P,U,U_hat,dpdxi,temp,temp_hat,P_hat,P_hat0,P_hat1)
    #---Energy Eqn---#
    dpdxi .= P./rho     #dpdxi is used as temporary variable
    fftn_mpi(dpdxi,dU)
    @fast for i in eachindex(dU) dU[i] *= (-c4)*K2[i] end
    temp .= 0.0           #Read as pdiff (pressure diffusion term)
    for i0 in 1:3
        tmp2i = (i0-1)*tmpi
        @fast for i ∈ eachindex(dpdxi) dpdxi[i] = U[i+tmp2i] * P[i] end #Again, dpdxi is temporary variable
        fftn_mpi(dpdxi, temp_hat)        #Read temp_hat as puTerm1_hat
        tmp2 = (i0-1)*tmp
        @fast for i in eachindex(dU)
            dU[i] -= (1im*gamma)*K[i+tmp2]*temp_hat[i]
            temp_hat[i] = 1im*K[i+tmp2]*P_hat[i]    #Now that the useful value is stored in dU, reuse this for temporary purpose
        end
        ifftn_mpi(temp_hat, dpdxi)
        @fast for i ∈ eachindex(dpdxi) dpdxi[i] *= U[i+tmp2i] end
        fftn_mpi(dpdxi, temp_hat)   #Read temp_hat as puTerm2_hat
        tmp3 = ((i0+1)%3)*tmp
        @fast for i ∈ eachindex(dU)
            dU[i] += (gamma-1)*temp_hat[i]
            temp_hat[i] = 1im*(K[i+tmp3]*U_hat[i+tmp2]+K[i+tmp2]*U_hat[i+tmp3])
        end
        ifftn_mpi(temp_hat, dpdxi)        #Read dpdxi as duidxi-1 + dui-1dxi
        @fast for i ∈ eachindex(temp) temp[i] += dpdxi[i]^2 end
        @fast for i ∈ eachindex(temp_hat)
            temp_hat[i] = 1im*(K[i+tmp2]*U_hat[i+tmp2]-K[i+tmp3]*U_hat[i+tmp3])
        end
        ifftn_mpi(temp_hat, dpdxi)        #Read dpdxi as duidxi - dui-1dxi-1
        @fast for i ∈ eachindex(temp) temp[i] += (2/3)*dpdxi[i]^2 end
    end
    fftn_mpi(temp, temp_hat)             #Read temp_hat as pdiff_hat
    @fast for i ∈ eachindex(dU) dU[i] += c3*temp_hat[i] end
    dealias!(dU)    #De-alias complete RHS, as all terms in third equation are non-linear
    RK4Update(P_hat1,a[rk]*dt,dU)
    if rk < 4 RK4Update(P_hat,P_hat0,b[rk]*dt,dU) end
end

function runRK4step(rho,rho_hat,P,P_hat,U,U_hat,M_hat,rho_hat0,P_hat0,M_hat0,rho_hat1,P_hat1,M_hat1, dU,dpdxi,temp,temp_hat)
    ##--Solve the 3D N-S equations for compressible flow--##
    rho_hat1[:] = rho_hat
    rho_hat0[:] = rho_hat
    M_hat1[:] = M_hat
    M_hat0[:] = M_hat
    P_hat1[:] = P_hat
    P_hat0[:] = P_hat
    for rk::Int8 in 1:4
        if rk > 1 updateFields(rho,rho_hat,P,P_hat,U,U_hat,M_hat) end
        contEqn(rk,dU,M_hat,rho_hat,rho_hat0,rho_hat1)
        momEqn(rk,dU,rho,P_hat,U,U_hat,temp,temp_hat,M_hat,M_hat0,M_hat1)
        energyEqn(rk,dU,rho,P,U,U_hat,dpdxi,temp,temp_hat,P_hat,P_hat0,P_hat1)
    end
    rho_hat[:] = rho_hat1
    M_hat[:] = M_hat1
    P_hat[:] = P_hat1
    updateFields(rho,rho_hat,P,P_hat,U,U_hat,M_hat)  #Find this timestep's solution in Cartesian space
end

function runSimulation(tsteps,rho,rho_hat,P,P_hat,U,U_hat,M_hat,rho_hat0,P_hat0,M_hat0,rho_hat1,P_hat1,M_hat1,dU,dpdxi,temp,temp_hat)
    tstep = 0
    t = 0.0
    while tstep<tsteps
        tstep += 1
        t += dt
        runRK4step(rho,rho_hat,P,P_hat,U,U_hat,M_hat,rho_hat0,P_hat0,M_hat0,rho_hat1,P_hat1,M_hat1,dU,dpdxi,temp,temp_hat)
        #Save, if necessary
        if tstep%ts_save1==0 saveParams(rho, P, t, tstep) end
        if tstep%ts_save2==0 saveSpectra(U, rho, P_hat, tstep) end
        if tstep%ts_save3==0 saveSnapshot(U_hat, rho_hat, P_hat, tstep) end
    end
end

######----------START SIMULATION RUNS---------######
# if rank==0 printfile("Starting simulation!!") end
const ts_save1 = 10      #Timestep intervals to save simulation parameters at.
const ts_save2 = 10      #Timestep intervals to save energy spectra at.
const ts_save3 = 20      #Timestep intervals to save snapshots at.
runSimulation(1,rho,rho_hat,P,P_hat,U,U_hat,M_hat,rho_hat0,P_hat0,M_hat0,rho_hat1,P_hat1,M_hat1,dU,dpdxi,temp,temp_hat)    #Compile the function
const steps = 10
const t0 = time()
runSimulation(steps,rho,rho_hat,P,P_hat,U,U_hat,M_hat,rho_hat0,P_hat0,M_hat0,rho_hat1,P_hat1,M_hat1,dU,dpdxi,temp,temp_hat)
if rank==0
    printfile("For n_proc=",num_processes,", time taken for ",steps," steps = ",round(time()-t0,sigdigits=6)," s.")
    # printfile(U_hat[7,5,4,end:-1:1],",",P[7:10,9,4])
end
if rank==0 close(tsfile) end
##RESULTS on AMD Ryzen 5600x:
##For n_procs = 1, 2, 4 : Speedup v/s Python ~ 1.9x, 2.05x, 2.5x to 3.5x (more for larger N)

