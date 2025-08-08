# GPU Cluster Programming with MPI and CUDA

A comprehensive implementation demonstrating high-performance computing techniques that combine MPI (Message Passing Interface) for distributed computing across cluster nodes with CUDA for GPU acceleration within each node.

## 🎯 Overview

This project showcases advanced parallel programming concepts by implementing algorithms that scale across multiple GPUs in a cluster environment. It demonstrates how to effectively combine:

- **MPI**: For inter-node communication and task distribution
- **CUDA**: For GPU acceleration and parallel computation
- **Hybrid Programming**: Optimal workload distribution between CPUs and GPUs

## 🏗️ Architecture

```
Cluster Node 1          Cluster Node 2          Cluster Node N
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MPI Process   │◄──►│   MPI Process   │◄──►│   MPI Process   │
│                 │    │                 │    │                 │
│   ┌─────────┐   │    │   ┌─────────┐   │    │   ┌─────────┐   │
│   │  GPU 1  │   │    │   │  GPU 1  │   │    │   │  GPU 1  │   │
│   │ (CUDA)  │   │    │   │ (CUDA)  │   │    │   │ (CUDA)  │   │
│   └─────────┘   │    │   └─────────┘   │    │   └─────────┘   │
│   ┌─────────┐   │    │   ┌─────────┐   │    │   ┌─────────┐   │
│   │  GPU 2  │   │    │   │  GPU 2  │   │    │   │  GPU 2  │   │
│   │ (CUDA)  │   │    │   │ (CUDA)  │   │    │   │ (CUDA)  │   │
│   └─────────┘   │    │   └─────────┘   │    │   └─────────┘   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Features

- **Multi-Node GPU Computing**: Distribute workloads across multiple cluster nodes
- **Optimized Memory Management**: Efficient data transfer between host and device memory
- **Scalable Communication**: MPI-based communication patterns for cluster environments
- **Performance Benchmarking**: Built-in timing and performance measurement tools
- **Error Handling**: Comprehensive error checking for both MPI and CUDA operations
- **Multiple Algorithm Implementations**: Various parallel algorithms demonstrating different patterns

## 🛠️ Technologies Used

- **MPI**: Message Passing Interface for distributed computing
- **CUDA**: NVIDIA's parallel computing platform
- **C/C++**: Core programming languages
- **OpenMP**: Additional parallelization support
- **SLURM**: Job scheduling for cluster environments (optional)

## 📁 Project Structure

```
GPU-Cluster-Programming-MPI-CUDA/
├── src/
│   ├── matrix_operations/
│   │   ├── matrix_multiply.cu       # GPU matrix multiplication
│   │   ├── matrix_mpi.c            # MPI distribution logic
│   │   └── matrix_hybrid.cu        # Combined MPI+CUDA implementation
│   ├── numerical_methods/
│   │   ├── jacobi_solver.cu        # Jacobi iterative solver
│   │   ├── heat_equation.cu        # Heat equation simulation
│   │   └── monte_carlo.cu          # Monte Carlo methods
│   ├── communication/
│   │   ├── mpi_cuda_utils.c        # MPI-CUDA integration utilities
│   │   ├── data_transfer.cu        # Optimized data transfer routines
│   │   └── collective_ops.c        # Custom collective operations
│   └── benchmarks/
│       ├── performance_tests.c     # Performance measurement tools
│       ├── scaling_analysis.cu     # Scalability analysis
│       └── memory_bandwidth.cu     # Memory bandwidth tests
├── include/
│   ├── mpi_cuda_common.h          # Common definitions and macros
│   ├── gpu_utils.h                # GPU utility functions
│   └── timing.h                   # Performance timing utilities
├── scripts/
│   ├── compile.sh                 # Compilation script
│   ├── run_cluster.sh            # Cluster execution script
│   └── benchmark.sh              # Automated benchmarking
├── data/
│   ├── input/                    # Input data files
│   └── results/                  # Output and benchmark results
├── docs/
│   ├── setup_guide.md           # Cluster setup instructions
│   ├── performance_analysis.md  # Performance analysis documentation
│   └── algorithm_details.md     # Detailed algorithm explanations
├── Makefile                     # Build configuration
└── README.md                    # This file
```

## 🔧 Prerequisites

### Software Requirements
- **CUDA Toolkit** (version 11.0+)
- **MPI Implementation** (OpenMPI, MPICH, or Intel MPI)
- **GCC Compiler** (version 9.0+)
- **CMake** (version 3.18+)

### Hardware Requirements
- **Multi-node cluster** with NVIDIA GPUs
- **InfiniBand or Ethernet** interconnect
- **Sufficient GPU memory** (varies by algorithm)

## 🚀 Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pengwingokla/GPU-Cluster-Programming-MPI-CUDA.git
cd GPU-Cluster-Programming-MPI-CUDA
```

### 2. Environment Setup
```bash
# Load required modules (adjust for your cluster)
module load cuda/11.8
module load openmpi/4.1.0
module load gcc/9.3.0

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export MPI_HOME=/usr/local/openmpi
export PATH=$CUDA_HOME/bin:$MPI_HOME/bin:$PATH
```

### 3. Compilation
```bash
# Using provided script
./scripts/compile.sh

# Or manual compilation
make all

# For specific components
make matrix_operations
make numerical_methods
```

## 🏃‍♂️ Running the Code

### Single Node (Multiple GPUs)
```bash
# Run with 4 MPI processes on local node
mpirun -np 4 ./bin/matrix_multiply_hybrid 2048 2048

# Run with specific GPU assignment
mpirun -np 2 --bind-to socket ./bin/heat_equation 1000 1000 100
```

### Multi-Node Cluster
```bash
# Using hostfile
mpirun -np 8 -hostfile hostfile ./bin/jacobi_solver 4096 4096

# Using SLURM job scheduler
sbatch scripts/run_cluster.sh

# Interactive cluster run
srun -N 4 -n 16 --gres=gpu:2 ./bin/monte_carlo 1000000
```

## 📊 Example Algorithms

### 1. Distributed Matrix Multiplication
- **Problem**: Multiply large matrices across multiple GPUs
- **Approach**: Block decomposition with MPI communication
- **Files**: `src/matrix_operations/matrix_hybrid.cu`

### 2. Parallel Heat Equation Solver
- **Problem**: Solve 2D heat diffusion equation
- **Approach**: Domain decomposition with ghost cell communication
- **Files**: `src/numerical_methods/heat_equation.cu`

### 3. Distributed Monte Carlo Simulation
- **Problem**: Large-scale random sampling
- **Approach**: Independent sampling with MPI reduction
- **Files**: `src/numerical_methods/monte_carlo.cu`

## 📈 Performance Analysis

### Benchmarking
```bash
# Run comprehensive benchmarks
./scripts/benchmark.sh

# Specific performance tests
mpirun -np 8 ./bin/performance_tests --test=matrix_multiply --size=4096
mpirun -np 4 ./bin/scaling_analysis --algorithm=heat_equation --nodes=1,2,4,8
```

### Expected Performance Gains
- **Strong Scaling**: Near-linear speedup for compute-intensive tasks
- **Weak Scaling**: Maintained efficiency with proportional problem size increase
- **GPU Utilization**: 85-95% GPU utilization across cluster nodes

## 🔍 Key Programming Concepts

### MPI-CUDA Integration Patterns

#### 1. GPU-Aware MPI
```c
// Direct GPU memory communication
cudaMalloc((void**)&d_data, size * sizeof(float));
MPI_Send(d_data, size, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
```

#### 2. Asynchronous Communication
```c
// Overlap computation and communication
cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);
MPI_Isend(h_data, size, MPI_FLOAT, dest, tag, MPI_COMM_WORLD, &request);
```

#### 3. Load Balancing
```c
// Dynamic work distribution
int work_per_process = total_work / num_processes;
int remainder = total_work % num_processes;
int my_work = work_per_process + (rank < remainder ? 1 : 0);
```

## 🧪 Testing

### Unit Tests
```bash
# Run unit tests
make test
./bin/run_tests

# Test specific components
./bin/test_mpi_cuda_utils
./bin/test_gpu_kernels
```

### Validation
```bash
# Compare results with serial implementation
./bin/validate_results --algorithm=matrix_multiply --size=1024
```

## 📝 Configuration

### Runtime Configuration
Create a `config.ini` file:
```ini
[general]
num_gpus_per_node = 2
memory_pool_size = 1024MB
optimization_level = O3

[communication]
use_gpu_aware_mpi = true
async_communication = true
message_aggregation = true

[algorithms]
block_size = 256
tile_size = 32
convergence_threshold = 1e-6
```

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce problem size or increase cluster nodes
export CUDA_LAUNCH_BLOCKING=1  # For debugging
```

#### MPI Communication Errors
```bash
# Check network connectivity
mpirun -np 2 --hostfile hostfile hostname

# Debug MPI issues
export OMPI_MCA_btl_base_verbose=100
```

#### Performance Issues
```bash
# Profile with NVIDIA tools
nvprof mpirun -np 4 ./bin/matrix_multiply 2048
nsys profile --trace=cuda,mpi ./bin/heat_equation 1000
```

### Development Guidelines
- Follow CUDA and MPI best practices
- Include performance benchmarks for new algorithms
- Maintain backward compatibility
- Document all public APIs

## 🙏 Acknowledgments

- **NVIDIA** for CUDA toolkit and documentation
- **Open MPI Community** for MPI implementation

---

**Built for High-Performance Computing Excellence** 🚀
