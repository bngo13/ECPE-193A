import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import argparse
import time


mod = SourceModule(
"""
__global__
void gpu_kernel(int *d_a, int *d_b, int *d_c, int vec_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < vec_size) {
        d_c[tid] = d_a[tid] + d_b[tid];
    }
}
""")

vec_size = 10_000

def args():
    global vec_size
    parser = argparse.ArgumentParser()
    parser.add_argument("--vecsize", type=int, required=True)

    args = parser.parse_args()
    vec_size = args.vecsize

    print(f"Running program with vector size of {vec_size:,}")

def gpu_version():
    # If the types aren't int32, the program will only run up to 500 elements
    # I'm guessing that it's cause of larger int padding that happens with int64s
    h_a = np.arange(0, vec_size, 1).astype(np.int32)
    h_b = np.arange(0, vec_size, 1).astype(np.int32)
    h_c = np.empty((vec_size,))

    # Allocate device pointers with the same size as the host lists
    d_a = drv.mem_alloc(h_a.nbytes)
    d_b = drv.mem_alloc(h_b.nbytes)
    d_c = drv.mem_alloc(h_c.nbytes)

    # Copy data from host to device
    htod_start = time.time()
    drv.memcpy_htod(d_a, h_a)
    drv.memcpy_htod(d_b, h_b)
    htod_end = time.time()

    htod_time = htod_end - htod_start
    print(f"GPU | HtoD Time: {htod_time}")

    # Set block and grid sizes
    block_size = 1024
    grid_size = (vec_size + block_size - 1) // block_size

    # Get kernel and run it
    gpu_kernel = mod.get_function("gpu_kernel")

    kernel_start = time.time()
    gpu_kernel(d_a, d_b, d_c, np.int32(vec_size), block=(block_size, 1, 1), grid=(grid_size, 1))
    kernel_end = time.time()

    kernel_time = kernel_end - kernel_start
    print(f"GPU | Kernel Time: {kernel_time}")

    # Get results

    dtoh_start = time.time()
    drv.memcpy_dtoh(h_c, d_c)
    dtoh_end = time.time()
    dtoh_time = dtoh_end - dtoh_start
    print(f"GPU | DtoH Time: {dtoh_time}")

    print(f"GPU | Total Time: {htod_time + kernel_time + dtoh_time}")

def cpu_version():
    h_a = np.arange(0, vec_size, 1)
    h_b = np.arange(0, vec_size, 1)
    h_c = np.empty((vec_size,))

    cpu_computation_start = time.time()
    for i in range(0, vec_size):
        h_c = h_a[i] + h_b[i]
    cpu_computation_end = time.time()

    print(f"CPU | Total Time: {cpu_computation_end - cpu_computation_start}")

def main():
    args()
    gpu_version()
    print()
    cpu_version()

if __name__ == "__main__":
    main()