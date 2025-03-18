import argparse
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import time
from pycuda.compiler import SourceModule

convolution_kernel = SourceModule(
"""
__global__ void convolution(int *image, float *kernel, int imageWidth, int imageHeight, int kernelWidth, int kernelHeight) {

}
"""
)

image_path = ""
image = None
sigma = None
hcwin = None
block = None

def imread(filename):
    with open(filename, 'rb') as f:
        # Read the magic number (P5)
        magic_number = f.readline().strip()
        if magic_number != b'P5':
            raise ValueError("Not a P5 PGM file")

        # Read image dimensions
        dimensions = f.readline().strip()
        width, height = map(int, dimensions.split())
        print(f"Image Size: {height},{width}")

        # Ignore max pixel size
        _ = f.readline().strip()

        # Read the pixel data
        pixel_data = f.read()
        image = np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width))

        return image

def imwrite(filename, image):
    height, width = image.shape

    # Open the file in binary mode for writing
    with open(filename, 'wb') as f:
        # Write the magic number
        f.write(b'P5\n')

        # Write the image dimensions
        f.write(f"{width} {height}\n".encode())

        # Write the max pixel value
        f.write(b'255\n')

        # Write the pixel data
        f.write(image.tobytes())

def get_args():
    parser = argparse.ArgumentParser()

    # Args
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--hcwin", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    
    # Getting args back
    args = parser.parse_args()
    
    # Setting args
    global image_path, sigma, hcwin, block
    image_path = args.image
    sigma = args.sigma
    hcwin = args.hcwin
    block = args.block

def parse_args():
    global image
    image = imread(image_path)

def GaussianKernel():
  # Calc a vertical gaussian kernel
  a = int(2.5 * sigma - 0.5)
  width = 2*a+1

  total = 0

  G = np.zeros((width,1))
  for i in range(0,width):
    G[i] = np.exp((-1 * (i-a) * (i-a)) / (2 * sigma * sigma))
    total = total + G[i]

  G = G / total
  return G

def GaussianDerivative():
  # Calc a vertical derivative gaussian kernel
  a = int(2.5 * sigma - 0.5)
  width = 2 * a + 1

  total=0

  G = np.zeros((width,1))
  for i in range(0,width):
    G[i] = (-1 * (i-a) * np.exp((-1 * (i-a) * (i-a)) / (2 * sigma * sigma)))
    total = total - i * G[i]

  G = G / total
  return G

def vertical_gaussian():
    # First perform a gaussian blur
    vertical_kernel = GaussianKernel(sigma)

def horizontal_gaussian(image):
   pass

def main():
    get_args()
    parse_args()

    (vertical_blur, horiz_grad) = vertical_gaussian()
    (horizontal_blur, vert_grad) = horizontal_gaussian()

    breakpoint()
    pass

if __name__ == "__main__":
    main()
