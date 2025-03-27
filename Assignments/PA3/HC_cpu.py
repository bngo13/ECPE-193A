import argparse
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import time
from pycuda.compiler import SourceModule

image_path = ""
image = None
sigma = None
hcwin = None
block = None

CORNERNESS = 0.04
KVAL = 50
MIN_DISTANCE = 15

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


def convolve(image, kernel):
  img_shape = image.shape
  kernel_shape = kernel.shape

  # Create a new image to not overwrite the old one
  convolution_img = np.empty_like(image).astype(np.float32)

  for i in range(0, img_shape[0]):
    for j in range(0, img_shape[1]):
      # For each pixel, run the gradient over it
      pixel_sum = 0.0
      for ki in range(0, kernel_shape[0]):
        for kj in range(0, kernel_shape[1]):
          # Calculate offsets for kernel and image
          offset_i = -1 * (kernel_shape[0] // 2) + ki
          offset_j = -1 * (kernel_shape[1] // 2) + kj
          if (i + offset_i >= 0 and i + offset_i < img_shape[0] and j + offset_j >= 0 and j + offset_j < img_shape[1]):
            pixel_sum += image[i + offset_i, j + offset_j] * kernel[ki, kj]

      # Update new image with values
      convolution_img[i][j] = pixel_sum

  return convolution_img

def vertical_gaussian():
  # First perform a gaussian blur
  vertical_kernel = GaussianKernel()
  vertical_blur = convolve(image, vertical_kernel)

  # Then calculate the derivative in the other direction
  horizontal_kernel = GaussianDerivative()
  flipped_horizontal_kernel = np.transpose(horizontal_kernel)
  horizontal_gradient = convolve(vertical_blur, flipped_horizontal_kernel)
  return (vertical_blur.astype(np.uint8), horizontal_gradient)

def horizontal_gaussian():
  # First perform a gaussian blur
  vertical_kernel = GaussianKernel()
  horizontal_kernel = np.transpose(vertical_kernel)
  horizontal_blur = convolve(image, horizontal_kernel)

  # Then calculate the derivative in the other direction
  horizontal_kernel = GaussianDerivative()
  vertical_gradient = convolve(horizontal_blur, horizontal_kernel)
  return (horizontal_blur.astype(np.uint8), vertical_gradient)

def covariance(vert_grad, horiz_grad):
  (height, width) = image.shape
  window = np.int32(hcwin)
  Z = np.zeros((height, width, 3))
  for i in range(0, height):
    for j in range(0, width):
      ixx = 0
      iyy = 0
      ixy = 0

      w = window // 2

      for offseti in range(-w, w + 1):
        for offsetj in range(-w, w + 1):
          pixel_offset_i = i + offseti
          pixel_offset_j = j + offsetj
          if (pixel_offset_i >= 0 and pixel_offset_j >= 0 and pixel_offset_i < height and pixel_offset_j < width):
            vert = vert_grad[pixel_offset_i][pixel_offset_j]
            horiz = horiz_grad[pixel_offset_i][pixel_offset_j]

            ixx += vert.astype(np.float32) * vert.astype(np.float32)
            iyy += horiz.astype(np.float32) * horiz.astype(np.float32)
            ixy += vert.astype(np.float32) * horiz.astype(np.float32)
      
      Z[i, j, 0] = ixx
      Z[i, j, 1] = ixy
      Z[i, j, 2] = iyy
  return Z

def find_corners(cov_mat, cornerness_val = CORNERNESS):
    # Get individual covariance for calculation
    Sxx = cov_mat[:, :, [0]].flatten()
    Sxy = cov_mat[:, :, [1]].flatten()
    Syy = cov_mat[:, :, [2]].flatten()

    # Find best corners
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

def main():
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

    print("-- Saving Results --")
    with open('corners.txt', 'w') as f:
      for (y,x, _) in top_features:
        f.write(f"{x} {y}\n")

    import cv2
    img = image.copy()
    for (y,x, _) in top_features:
        cv2.putText(img, 'X', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    imwrite("corners.pgm", img)

    te = time.time()
    print(f"GPU S Total Time:       {te - ts}")

if __name__ == "__main__":
    main()
