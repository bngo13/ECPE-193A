import argparse
import numpy as np
# import pycuda.autoinit
# import pycuda.driver as drv
import time
# from pycuda.compiler import SourceModule

# convolution_kernel = SourceModule("""
# __global__ void convolution(float *d_A, float *d_B, float *d_C, int width) {
# }
# """)

image = None
sigma = None
hcwin = None
block = None

def args():
    parser = argparse.ArgumentParser()

    # Args
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--hcwin", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    
    # Getting args back
    args = parser.parse_args()
    
    # Setting args
    global image, sigma, hcwin, block
    image = args.image
    sigma = args.sigma
    hcwin = args.hcwin
    block = args.block

def main():
    args()
    pass

if __name__ == "__main__":
    main()