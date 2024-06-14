##################################################
#           Makefile of GandAlf                  #
#                                                #
#  NOTE: environmental variables                 #
#     CUDAARCH, CUDA_INCLUDE                     #
#  need to be properly defined                   #
##################################################

# This Makefile has been rewritten to work on
# NERSC's Perlmutter GPU nodes. It will need editing to
# work on your GPU system of choice.

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

# Modules for running on NERSC Perlmutter
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
	$(NVCC) -c $(NVCCFLAGS) $(NVCCINCS) $< 

# main program
$(TARGET): gandalf.o
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(NVCCLIBS) 

gandalf.o: $(CU_DEPS)

test_make:
	@echo TARGET=    $(TARGET)
	@echo CUDA_INCLUDE=    $(CUDA_INCLUDE)
	@echo NVCC=      $(NVCC)
	@echo NVCCFLAGS= $(NVCCFLAGS)
	@echo NVCCINCS=  $(NVCCINCS)
	@echo NVCCLIBS=  $(NVCCLIBS)

clean:
	rm -rf *.o *~ \#*

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

