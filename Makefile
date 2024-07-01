##################################################
#           Makefile of GandAlf 				 #
#                                                #
#  NOTE: environmental variables                 #
#     CUDAARCH, CUDA_INCLUDE                           #
#  need to be properly defined                   #
##################################################

####
# added by Nicholas Bidler on 2024/06/28
# source: https://github.com/petermcLBL/fftx-demo-extern-app/blob/main/Makefile
####

#
# FFTX
#
FFTX_INCLUDE_DIR=$(FFTX_HOME)/include
FFTX_MPI_INCLUDE_DIR=$(FFTX_HOME)/src/library/lib_fftx_mpi
FFTX_INCLUDE=-I$(FFTX_INCLUDE_DIR) -I$(FFTX_MPI_INCLUDE_DIR)

FFTX_LIB_DIR=$(FFTX_HOME)/lib
# The FFTX libraries (for either GPU or CPU) go after $(FFTX_LINK).
# Apple needs a space after rpath, while other platforms need an equal sign.
ifeq ($(VENDOR),apple)
  FFTX_LINK=-Wl,-rpath $(FFTX_LIB_DIR) -L$(FFTX_LIB_DIR)
else
  ifndef ROCM_PATH
    # This flag gets rid of annoying "DSO missing from command line" error.
    LDFLAGS=-Wl,--copy-dt-needed-entries
  endif
  FFTX_LINK=-Wl,-rpath=$(FFTX_LIB_DIR) -L$(FFTX_LIB_DIR)
endif

# Need to link to ALL CPU/GPU libraries, even though we don't call them all.
FFTX_CPU_LIBRARIES=-lfftx_mddft_cpu -lfftx_imddft_cpu -lfftx_mdprdft_cpu -lfftx_imdprdft_cpu -lfftx_dftbat_cpu -lfftx_idftbat_cpu -lfftx_prdftbat_cpu -lfftx_iprdftbat_cpu -lfftx_rconv_cpu
FFTX_MPI_LIBRARY=-lfftx_mpi
FFTX_GPU_LIBRARIES=-lfftx_mddft_gpu -lfftx_imddft_gpu -lfftx_mdprdft_gpu -lfftx_imdprdft_gpu -lfftx_dftbat_gpu -lfftx_idftbat_gpu -lfftx_prdftbat_gpu -lfftx_iprdftbat_gpu -lfftx_rconv_gpu

ifdef CUDATOOLKIT_HOME
  default: CUDA
else ifdef ROCM_PATH
  default: HIP
else ifdef ONEAPI_DEVICE_SELECTOR
  default: SYCL
else
  default: CPU
endif

#
## make CUDA: needs CUDATOOLKIT_HOME
#
CUDA: CC=nvcc
CUDA: CCFLAGS=-x cu -std=c++14
CUDA: PRESETS=-DFFTX_CUDA
# To get helper_cuda.h
CUDA: CC_INCLUDE=-I$(CUDATOOLKIT_HOME)/../../examples/OpenMP/SDK/include
CUDA: CC_LINK=-L$(CUDATOOLKIT_HOME)/lib64 -lcudart
CUDA: FFTX_LIBRARIES=$(FFTX_MPI_LIBRARY) $(FFTX_GPU_LIBRARIES)
# Targets to build.
CUDA: gandalf

#
## make HIP: needs ROCM_PATH and CRAY_MPICH_PREFIX
#
HIP: CC=hipcc
HIP: PRESETS=-DFFTX_HIP
# To get mpi.h
HIP: CC_INCLUDE=-I$(CRAY_MPICH_PREFIX)/include
HIP: CC_LINK=-L$(ROCM_PATH)/lib -lamdhip64 -lhipfft -lrocfft -lstdc++
HIP: FFTX_LIBRARIES=$(FFTX_MPI_LIBRARY) $(FFTX_GPU_LIBRARIES)
# Targets to build.
HIP: gandalf

#
## make SYCL: needs AURORA_PE_ONEAPI_ROOT
#
SYCL: CC=mpicc
SYCL: CCFLAGS=-std=c++17 -fsycl
SYCL: PRESETS=-DFFTX_SYCL
SYCL: CC_LINK=-L$(AURORA_PE_ONEAPI_ROOT) -lOpenCL -lmkl_core -lmkl_cdft_core -lmkl_sequential -lmkl_rt -lmkl_intel_lp64 -lmkl_sycl
SYCL: FFTX_LIBRARIES=$(FFTX_GPU_LIBRARIES)
# Targets to build.
SYCL: gandalf

#
## make CPU: default
#
CPU: CC=mpicxx
CPU: CCFLAGS=-std=c++11
CPU: FFTX_LIBRARIES=$(FFTX_CPU_LIBRARIES)
# Targets to build.
CPU: gandalf

#######
# end of new additions 2024/06/28
#######


TARGET    = gandalf
CU_DEPS := \
	c_fortran_namelist3.c \
	kernels.cu \
	maxReduc.cu \
	nlps.cu \
	nonlin.cu \
	courant.cu \
	timestep.cu \
	forcing.cu \
	diagnostics.cu

FILES     = *.cu *.c *.cpp Makefile
VER       = `date +%y%m%d`

# module load PrgEnv-gnu/8.3.3
# module load gcc/11.2.0
# module load nvhpc-mixed/22.7
# module load cudatoolkit/11.7
# module load cray-mpich/8.1.25
# module load cray-hdf5-parallel
# module load cray-netcdf-hdf5parallel

CUDAARCH  = 80
NVCC      = nvcc
NVCCFLAGS = --forward-unknown-to-host-compiler -arch=compute_$(CUDAARCH) -code=sm_$(CUDAARCH) -fPIC -rdc=true --disable-warnings
NETCDF_ROOT = ${NETCDF_DIR}
NVHPC_ROOT = ${NVIDIA_PATH}
NVCCINCS  = -I ${NVHPC_ROOT}/math_libs/include -I ${NVHPC_ROOT}/include -I${NETCDF_ROOT}/include
NVCCLIBS  = -L ${NVHPC_ROOT}/math_libs/lib64 -lcufft -L ${NVHPC_ROOT}/cuda/lib64 -lcudart -L ${NETCDF_ROOT}/lib -lnetcdf -L${HDF5_ROOT}/lib -lhdf5

ifeq ($(debug),on)
  NVCCFLAGS += -g -G
else
  NVCCFLAGS += -O3
endif

.SUFFIXES:
.SUFFIXES: .cu .o

.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $(NVCCINCS) $(PRESETS) $(CC_INCLUDE) $(FFTX_INCLUDE) $< 

# main program
$(TARGET): gandalf.o
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(NVCCLIBS) $(CC_LINK) $(FFTX_LINK) $(FFTX_LIBRARIES)

gandalf.o: $(CU_DEPS)

test_make:
	@echo TARGET=    $(TARGET)
	@echo CUDA_INCLUDE=    $(CUDA_INCLUDE)
	@echo NVCC=      $(NVCC)
	@echo NVCCFLAGS= $(NVCCFLAGS)
	@echo NVCCINCS=  $(NVCCINCS)
	@echo NVCCLIBS=  $(NVCCLIBS)
	@echo FFTX_HOME=  $(FFTX_HOME)

clean:
	rm -rf gandalf *.o *~ \#*

distclean: clean
	rm -rf $(TARGET)

tar:
	@echo $(TARGET)-$(VER) > .package
	@-rm -fr `cat .package`
	@mkdir `cat .package`
	@ln $(FILES) `cat .package`
	tar cvf - `cat .package` | bzip2 -9 > `cat .package`.tar.bz2
	@-rm -fr `cat .package` .package

fldfol: fldfol.o
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(NVCCLIBS)

fldfol.o:  $(FLDFOL_DEPS)

