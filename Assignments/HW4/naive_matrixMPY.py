import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import time
from pycuda.compiler import SourceModule

gpu_kernel = SourceModule("""
__global__ void matrixmul_kernel(float *d_A, float *d_B, float *d_C, int width) {
    int row, col, k = 0;
    float temp = 0;

    // Indexes for row and columns
    row = threadIdx.x + blockIdx.x * blockDim.x;
    col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < width && col < width) {
        temp = 0;
        for (k = 0; k < width; k++) {
            temp += d_A[row * width + k] * d_B[k * width + col];
        }
    }
    d_C[row * width + col] = temp;
}
""")

def cpu_matmul(inmat1: np.matrix, inmat2: np.matrix):
    inmat1 = inmat1.astype(np.float32)
    inmat2 = inmat2.astype(np.float32)

    row, col = inmat1.shape
    res = np.zeros_like(inmat1)

    # We **love** matrix multiplication
    for i in range(0, row):
        for j in range(0, col):
            sum = 0
            for k in range(0, col):
                sum += inmat1[i, k] * inmat2[k, j]
            res[i, j] = sum

    return res

def gpu_matmul(inmat1: np.ndarray, inmat2: np.ndarray):
    # Get Prelim Sizes
    width, _ = inmat1.shape

    # Flatten literally everything
    h_A = inmat1.flatten().astype(np.float32)
    h_B = inmat2.flatten().astype(np.float32)
    h_C = np.empty((width * width,)).astype(np.float32)

    # Allocate GPU memory
    d_A = drv.mem_alloc(h_A.nbytes)
    d_B = drv.mem_alloc(h_B.nbytes)
    d_C = drv.mem_alloc(h_C.nbytes)

    # Memcpy to gpu
    drv.memcpy_htod(d_A, h_A)
    drv.memcpy_htod(d_B, h_B)

    # GPU Sizing Stuff
    block_size = 32
    grid_size = (width ** 2 + block_size - 1) // block_size

    # Get kernel and run it
    gKernel = gpu_kernel.get_function("matrixmul_kernel")
    width = np.int32(width)

    kernel_start = time.time()
    gKernel(d_A, d_B, d_C, width, block=(block_size, block_size, 1), grid=(grid_size, 1))
    kernel_end = time.time()

    kernel_time = kernel_end - kernel_start
    print(f"GPU | Time: {kernel_time}")

    # Grab Data
    drv.memcpy_dtoh(h_C, d_C)
    h_C = h_C.reshape((width, width))
    return h_C

def main():
    # Inputs
    inmat1 = np.array([
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
    ])

    inmat2 = np.array([
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
        [1, 2, 3, 4, 5, 2, 5, 9, 5, 2],
    ])

    # Run CPU Matrix Multiplication
    cpu_start = time.time()
    cpu_res = cpu_matmul(inmat1, inmat2)
    cpu_end = time.time()
    print(f"CPU | Time: {cpu_end - cpu_start}")

    # Run GPU Matrix Mult
    gpu_res = gpu_matmul(inmat1, inmat2)

    print("-- CPU Results --")
    print(cpu_res)
    print()

    print ("-- GPU Results -- ")
    print(gpu_res)
    print()

if __name__ == "__main__":
    main()