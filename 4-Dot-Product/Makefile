# Makefile for MPI + CUDA Dot Product

CUDA_PATH ?= /usr
MPICXX ?= mpicxx
NVCC ?= $(CUDA_PATH)/bin/nvcc -ccbin $(MPICXX)

NVCCFLAGS = -std=c++11 -O3 -m64 --gpu-architecture=compute_60 --gpu-code=sm_60
MPI_FLAGS = -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcudart -lmpi

TARGET = dot_prod
OBJS = dot_prod.o dot_prod_cuda.o dot_prod_utils.o  # Added dot_prod_utils.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(MPICXX) -o $@ $^ $(MPI_FLAGS)

dot_prod.o: dot_prod.c
	$(MPICXX) -c $< -o $@

dot_prod_cuda.o: dot_prod_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

dot_prod_utils.o: dot_prod_utils.c
	$(MPICXX) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
