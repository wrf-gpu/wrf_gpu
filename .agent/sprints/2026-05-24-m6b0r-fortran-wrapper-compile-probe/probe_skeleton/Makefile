# Makefile — M6B0-R Fortran wrapper compile probe
#
# Builds savepoint_wrapper.F90 + test_savepoint.F90 against the WRF-bundled
# HDF5 1.14.5 Fortran API.  Must be invoked AFTER sourcing
#   /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh
# which sets FC=nvfortran and exports HDF5/LDFLAGS/CPPFLAGS.
#
# Pinned to CPU cores 0-3 via the workflow (taskset is invoked by the caller).

FC      ?= nvfortran
HDF5    ?= /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_libs

FFLAGS  := -O0 -g -Mfreeform -Mbackslash -I$(HDF5)/include
LDFLAGS := -L$(HDF5)/lib
LIBS    := -lhdf5_fortran -lhdf5 -lz -ldl -lm

OBJS    := savepoint_wrapper.o test_savepoint.o
TARGET  := test_savepoint

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "=== LINK ==="
	$(FC) $(FFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

savepoint_wrapper.o: savepoint_wrapper.F90
	@echo "=== COMPILE savepoint_wrapper.F90 ==="
	$(FC) $(FFLAGS) -c $<

test_savepoint.o: test_savepoint.F90 savepoint_wrapper.o
	@echo "=== COMPILE test_savepoint.F90 ==="
	$(FC) $(FFLAGS) -c test_savepoint.F90

clean:
	rm -f $(OBJS) $(TARGET) *.mod
