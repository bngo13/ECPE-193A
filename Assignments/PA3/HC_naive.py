import argparse
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import time
from pycuda.compiler import SourceModule

gpu_kernels = SourceModule(
r"""
__global__ void convolution(int *image, int *convImg, float *kernel, int imageHeight, int imageWidth, int kernelHeight, int kernelWidth) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < imageHeight && j < imageWidth) {
    float pixel_sum = 0.0;

    for (int ki = 0; ki < kernelHeight; ki++) {
      for (int kj = 0; kj < kernelWidth; kj++) {
        // Calculate offsets based on kernel
        int offset_i = -1 * (kernelHeight / 2) + ki;
        int offset_j = -1 * (kernelWidth / 2) + kj;

        // Get the image index based on the offset
        int pixel_i = i + offset_i;
        int pixel_j = j + offset_j;

        // Calculate Gradient
        if (pixel_i >= 0 && pixel_j >= 0 && pixel_i < imageHeight && pixel_j < imageWidth) {
          float blurredPixel = image[pixel_i * imageWidth + pixel_j] * kernel[ki * kernelWidth + kj];
          pixel_sum += blurredPixel;
        }
      }
    }
    convImg[i * imageWidth + j] = (int)pixel_sum;
  }
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

def convolve(image, kernel, image_height, image_width, kernel_height, kernel_width):
  image_height = np.int32(image_height)
  image_width = np.int32(image_width)
  kernel_height = np.int32(kernel_height)
  kernel_width = np.int32(kernel_width)

  # Create Convolved Image
  convImg = np.zeros((image_height * image_width,)).astype(np.int32)

  # Allocate Memory
  d_image = drv.mem_alloc(image.nbytes)
  d_kernel = drv.mem_alloc(kernel.nbytes)
  d_convImg = drv.mem_alloc(convImg.nbytes)
  drv.Context.synchronize()

  # Memcpy data
  drv.memcpy_htod(d_image, image)
  drv.memcpy_htod(d_kernel, kernel)
  # drv.memcpy_htod(d_convImg, convImg) # Don't need since convImg doesn't have any data
  drv.Context.synchronize()

  # Run Convolution
  grid_size = int((max(image_height, image_width) + block - 1) // block)
  convolution = gpu_kernels.get_function("convolution")
  convolution(d_image, d_convImg, d_kernel, image_height, image_width, kernel_height, kernel_width, block=(block, block, 1), grid=(grid_size, grid_size))
  drv.Context.synchronize()

  # Get data
  drv.memcpy_dtoh(convImg, d_convImg)
  drv.Context.synchronize()
  convImg = convImg.reshape((image_height, image_width))

  return convImg

def vertical_gaussian():
    ## Vertical Gaussian Blur ##
    # Get kernel
    vertical_kernel = GaussianKernel()
    (vert_kernel_height, vert_kernel_width) = vertical_kernel.shape
    (image_height, image_width) = image.shape

    # Flatten and convolve
    vertical_kernel_flat = vertical_kernel.flatten().astype(np.float32)
    image_flat = image.flatten().astype(np.int32)
    vertical_blur = convolve(image_flat, vertical_kernel_flat, image_height, image_width, vert_kernel_height, vert_kernel_width)

    ## Horizontal Gaussian Derivative ##
    # Get kernel
    horizontal_kernel = GaussianDerivative()
    horizontal_kernel = np.transpose(horizontal_kernel)
    (h_kernel_height, h_kernel_width) = horizontal_kernel.shape
    (blur_height, blur_width) = vertical_blur.shape
    
    # Flatten and convolve
    horizontal_kernel_flat = horizontal_kernel.flatten().astype(np.float32)
    blur_flat = vertical_blur.flatten().astype(np.int32)
    horizontal_gradient = convolve(blur_flat, horizontal_kernel_flat, blur_height, blur_width, h_kernel_height, h_kernel_width)
    return (vertical_blur.astype(np.uint8), horizontal_gradient.astype(np.uint8))

def horizontal_gaussian():
    ## Horizontal Gaussian Blur ##

    # Get Kernel
    vertical_kernel = GaussianKernel()
    vertical_kernel = np.transpose(vertical_kernel)
    (v_kernel_h, v_kernel_w) = vertical_kernel.shape
    (img_h, img_w) = image.shape

    # Flatten and convolve
    vertical_kernel_flat = vertical_kernel.flatten().astype(np.float32)
    image_flat = image.flatten().astype(np.int32)
    horizontal_blur = convolve(image_flat, vertical_kernel_flat, img_h, img_w, v_kernel_h, v_kernel_w)

    ## Vertical Gaussian Derivative ##

    # Get Kernel
    horizontal_kernel = GaussianDerivative()
    (h_kernel_h, h_kernel_w) = horizontal_kernel.shape
    (blur_h, blur_w) = horizontal_blur.shape
    
    # Flatten then convolve
    kernel_flat = horizontal_kernel.flatten().astype(np.float32)
    blur_flat = horizontal_blur.flatten().astype(np.int32)
    vertical_gradient = convolve(blur_flat, kernel_flat, blur_h, blur_w, h_kernel_h, h_kernel_w)
    return (horizontal_blur.astype(np.uint8), vertical_gradient.astype(np.uint8))

def main():
    get_args()
    parse_args()

    (vertical_blur, horiz_grad) = vertical_gaussian()
    (horizontal_blur, vert_grad) = horizontal_gaussian()

    breakpoint()
    pass

if __name__ == "__main__":
    main()
