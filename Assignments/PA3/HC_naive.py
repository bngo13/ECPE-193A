import argparse
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import time
from pycuda.compiler import SourceModule

gpu_kernels = SourceModule(
r"""
__global__ void convolution(float *image, float *convImg, float *kernel, int imageHeight, int imageWidth, int kernelHeight, int kernelWidth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

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

    if (i < imageHeight && j < imageWidth) {
        convImg[i * imageWidth + j] = pixel_sum;
    }
}

__global__ void covariance(float *vert_grad, float *horiz_grad, float *cov_mat, int image_height, int image_width, int window) {
    // Initialize Vars
    float ixx = 0;
    float iyy = 0;
    float ixy = 0;
    int w = window / 2;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int offset_i = -w; offset_i < w + 1; offset_i++) {
        for (int offset_j = -w; offset_j < w + 1; offset_j++) {
            int pixel_i = i + offset_i;
            int pixel_j = j + offset_j;
            float vert = 0;
            float horiz = 0;
            int idx = pixel_i * image_width + pixel_j;

            if (idx >= 0 && pixel_i < image_height && pixel_j < image_width) {
              vert = vert_grad[idx];
              horiz = horiz_grad[idx];
            }

            ixx += vert * vert;
            iyy += horiz * horiz;
            ixy += vert * horiz;
        }
    }

    if (i < image_height && j < image_width) {
        // Save the matrix values with an offset of 4 per pixel. Legit couldn't find a way to double pointer this so this is a workaround.
        int index = (i * image_width + j) * 3;
        cov_mat[index + 0] = ixx;
        cov_mat[index + 1] = ixy;
        cov_mat[index + 2] = iyy;
    }
}
"""
)

image_path = ""
image = None
sigma = None
hcwin = None
block = None

CORNERNESS = 0.04
KVAL = 50
MIN_DISTANCE = 5

gpu_ktime = 0
d2h_time = 0
h2d_time = 0

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
    global gpu_ktime, h2d_time, d2h_time
    image_height = np.int32(image_height)
    image_width = np.int32(image_width)
    kernel_height = np.int32(kernel_height)
    kernel_width = np.int32(kernel_width)

    # Create Convolved Image
    convImg = np.zeros((image_height * image_width,)).astype(np.float32)
    image = image.astype(np.float32)

    # Allocate Memory
    d_image = drv.mem_alloc(image.nbytes)
    d_kernel = drv.mem_alloc(kernel.nbytes)
    d_convImg = drv.mem_alloc(convImg.nbytes)
    drv.Context.synchronize()

    # Memcpy data
    ts = time.time()
    drv.memcpy_htod(d_image, image)
    drv.memcpy_htod(d_kernel, kernel)
    # drv.memcpy_htod(d_convImg, convImg) # Don't need since convImg doesn't have any data
    drv.Context.synchronize()
    te = time.time()
    h2d_time += te - ts
    print(f"\tHTOD Time:     {te - ts}")

    # Run Convolution
    grid_x = int((image_width + block - 1) // block)
    grid_y = int((image_height + block - 1) // block)
    ts = time.time()
    convolution = gpu_kernels.get_function("convolution")
    convolution(d_image, d_convImg, d_kernel, image_height, image_width, kernel_height, kernel_width, block=(block, block, 1), grid=(grid_x, grid_y))
    drv.Context.synchronize()
    te = time.time()
    gpu_ktime += te - ts
    print(f"\tKernel Time:   {te - ts}")

    # Get data
    ts = time.time()
    drv.memcpy_dtoh(convImg, d_convImg)
    drv.Context.synchronize()
    te = time.time()
    d2h_time += te - ts
    print(f"\tDTOH Time:     {te - ts}")
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
    image_flat = image.flatten().astype(np.float32)
    print("Vertical Gaussian Kernel Convolve:")
    vertical_blur = convolve(image_flat, vertical_kernel_flat, image_height, image_width, vert_kernel_height, vert_kernel_width)

    ## Horizontal Gaussian Derivative ##
    # Get kernel
    horizontal_kernel = GaussianDerivative()
    horizontal_kernel = np.transpose(horizontal_kernel)
    (h_kernel_height, h_kernel_width) = horizontal_kernel.shape
    (blur_height, blur_width) = vertical_blur.shape
    
    # Flatten and convolve
    horizontal_kernel_flat = horizontal_kernel.flatten().astype(np.float32)
    blur_flat = vertical_blur.flatten().astype(np.float32)
    print("Vertical Gaussian Deriv Convolve:")
    horizontal_gradient = convolve(blur_flat, horizontal_kernel_flat, blur_height, blur_width, h_kernel_height, h_kernel_width)
    return (vertical_blur.astype(np.uint8), horizontal_gradient)

def horizontal_gaussian():
    ## Horizontal Gaussian Blur ##

    # Get Kernel
    vertical_kernel = GaussianKernel()
    vertical_kernel = np.transpose(vertical_kernel)
    (v_kernel_h, v_kernel_w) = vertical_kernel.shape
    (img_h, img_w) = image.shape

    # Flatten and convolve
    vertical_kernel_flat = vertical_kernel.flatten().astype(np.float32)
    image_flat = image.flatten().astype(np.float32)
    print("Horizontal Gaussian Kernel Convolve:")
    horizontal_blur = convolve(image_flat, vertical_kernel_flat, img_h, img_w, v_kernel_h, v_kernel_w)

    ## Vertical Gaussian Derivative ##

    # Get Kernel
    horizontal_kernel = GaussianDerivative()
    (h_kernel_h, h_kernel_w) = horizontal_kernel.shape
    (blur_h, blur_w) = horizontal_blur.shape
    
    # Flatten then convolve
    kernel_flat = horizontal_kernel.flatten().astype(np.float32)
    blur_flat = horizontal_blur.flatten().astype(np.float32)
    print("Horizontal Gaussian Deriv Convolve:")
    vertical_gradient = convolve(blur_flat, kernel_flat, blur_h, blur_w, h_kernel_h, h_kernel_w)
    return (horizontal_blur.astype(np.uint8), vertical_gradient)

def covariance(vert_grad, horiz_grad):
    global gpu_ktime, h2d_time, d2h_time
    (image_height, image_width) = image.shape

    image_height = np.int32(image_height)
    image_width = np.int32(image_width)
    window = np.int32(hcwin)

    # Flatten everything
    vert_grad = vert_grad.flatten()
    horiz_grad = horiz_grad.flatten()
    cov_mat = np.zeros((image_height, image_width, 3)).flatten().astype(np.float32)

    # Mallocs
    d_vert = drv.mem_alloc(vert_grad.nbytes)
    d_horiz = drv.mem_alloc(horiz_grad.nbytes)
    d_cov_mat = drv.mem_alloc(cov_mat.nbytes)
    drv.Context.synchronize()

    # Memcpys
    ts = time.time()
    drv.memcpy_htod(d_vert, vert_grad)
    drv.memcpy_htod(d_horiz, horiz_grad)
    drv.Context.synchronize()
    te = time.time()
    h2d_time += te - ts
    print(f"Covariance HTOD Time:   {te - ts}")

    # Run the thing
    grid_size = int((max(image_height, image_width) + block - 1) // block)
    covariance_gpu = gpu_kernels.get_function("covariance")
    ts = time.time()
    covariance_gpu(d_vert, d_horiz, d_cov_mat, image_height, image_width, window, block=(block, block, 1), grid=(grid_size, grid_size))
    drv.Context.synchronize()
    te = time.time()
    gpu_ktime += te - ts
    print(f"Covariance Kernel Time: {te - ts}")

    # Grab data
    ts = time.time()
    drv.memcpy_dtoh(cov_mat, d_cov_mat)
    drv.Context.synchronize()
    te = time.time()
    d2h_time += te - ts
    print(f"Covariance DTOH Time:   {te - ts}")
    cov_mat = cov_mat.reshape((image_height, image_width, 3))
    return cov_mat

def find_corners(cov_mat, cornerness_val = CORNERNESS):
    # Grab values for det and trace
    Sxx = cov_mat[:, :, [0]]
    Sxy = cov_mat[:, :, [1]]
    Syy = cov_mat[:, :, [2]]

    # Get det and trace directly. Faster than calculating eigenvalues
    detM = (Sxx * Syy) - (Sxy ** 2)
    traceM = Sxx + Syy
    R = detM - cornerness_val * (traceM ** 2)

    # Reshape back to image to get features later
    features = []
    R = R.reshape(image.shape)
    (h, w) = R.shape
    for i in range(0, h):
        for j in range(0, w):
            features.append((i, j, R[i, j]))
    return features

def get_top_features(features: list, k_values = KVAL, val_distance = MIN_DISTANCE):
    top_features = []
    feature_count = 0

    # Sort the features first
    sorted_features = sorted(features, key = lambda item : item[2], reverse=True)

    # For each feature, make sure the features chosen are at least val_distance away from each other
    for feature in sorted_features:
        # Keep going until k_values is hit
        if len(top_features) >= k_values:
            return top_features
        
        addable = True
        for existing_features in top_features:
        # Manhatten Distance Eq
            distance = abs(existing_features[1] - feature[1]) + abs(existing_features[0] - feature[0])
            if distance < val_distance:
                addable = False
                break
    
        if not addable:
            continue
      
        feature_count += 1
        top_features.append(feature)

    return top_features

def choose_gpu():
    student_id = 989400138
    drv.init()

    num_gpus = drv.Device.count()

    gpu_index = student_id % num_gpus

    print(f"GPU Index: {gpu_index}")

    drv.Device(gpu_index)

def main():
    choose_gpu()
    ts = time.time()
    get_args()
    parse_args()

    print("-- Calculating Gradients --")
    (vertical_blur, horiz_grad) = vertical_gaussian()
    (horizontal_blur, vert_grad) = horizontal_gaussian()
    print()

    print("-- Calculating Covariance --")
    cov_mat = covariance(vert_grad, horiz_grad)
    print()

    print("-- Getting Corners --")
    feature_list = find_corners(cov_mat)
    print()

    print("-- Getting Features --")
    top_features = get_top_features(feature_list)
    print()

    with open('corners.txt', 'w') as f:
      for (y,x, _) in top_features:
        f.write(f"{x} {y}\n")

    te = time.time()

    # Print Format
    print()
    print(f"image name, sigma, HC window size, GPU kernel time, h2d time, d2htime, total time")
    print(f"{image_path},{sigma},{hcwin},{gpu_ktime},{h2d_time},{d2h_time},{te-ts}")

if __name__ == "__main__":
    main()
