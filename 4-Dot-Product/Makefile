# Makefile for MPI + CUDA Dot Product

# Set CUDA path
CUDA_PATH ?= /usr

# Set compilers
MPICXX ?= mpicxx
NVCC ?= $(CUDA_PATH)/bin/nvcc -ccbin $(MPICXX)

# Compiler flags
NVCCFLAGS = -std=c++11 -O3 -m64 --gpu-architecture=compute_60 --gpu-code=sm_60
MPI_FLAGS = -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcudart -lmpi

# Targets
TARGET = dot_prod
OBJS = dot_prod.o dot_prod_cuda.o

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(MPICXX) -o $@ $^ $(MPI_FLAGS)

dot_prod.o: dot_prod.c
	$(MPICXX) -c $< -o $@

dot_prod_cuda.o: dot_prod_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run: $(TARGET)
	mpirun -np 4 ./$(TARGET) 10

clean:
	rm -f $(TARGET) $(OBJS)
