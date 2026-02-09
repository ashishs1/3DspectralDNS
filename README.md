# 3DspectralDNS
This repository contains the code for Compressible DNS in a 3-dimensional periodic turbulence box, which was part of my post-graduation project on 'Scaling of Weakly Compressible Turbulence'. The code can do the following:
1. Simulation of a 3-D periodic box of turbulence (by converting into Fourier space).
2. Parallel Computation for execution in HPC.
3. Parallel storage of raw DNS data in the most efficient data storage file type - HDF5!
4. Store simulation parameters like Mt, Rl, u'rms, lambda, epsilon, mean density, and vel derivative skewness & kurtosis.
5. Store energy spectrum after a pre-defined number of timesteps.
6. Compute triadic interactions - the generator scripts for triad indices in triadic\_interactions folder need to be run before this. _The numerical basis for triadic interaction computation can be found in Appendix-C of Reference.pdf attached in this repository._
Inspiration for the code structure was taken from [this repository](https://github.com/spectralDNS/spectralDNS), which is for incompressible spectral DNS.

---
### Initial Conditions
There are a few initial conditions to choose from. The first 2 are for testing the code. Taylor Greene Vortex can also be used for confirming the validity of the DNS output.
The reference for the initial solving of initial condition can be found in the Appendix-B of Reference.pdf attached in this repository.

---
### Port to Julia
There is an incomplete port to Julia (triadic interactions not implemented), which is of advantage to those who only want to run the DNS without the requirement of triadic interactions. The speed-up in Julia is > 2x for N >= 64. Performance of Julia also increases with increased number of cores.

---
### LICENSE
This code is released under MIT License.

---
### Footnotes
In case of any issues found/clarifications required, kindly open an issue here.

