import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import math
import os

def GaussianKernel(sigma):
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

def GaussianDerivative(sigma):
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
  convolution_img = np.empty_like(image)

  for i in range(0, img_shape[0]):
    for j in range(0, img_shape[1]):
      # For each pixel, run the gradient over it
      pixel_sum = 0
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

def vertical_gaussian(image, sigma):
  # First perform a gaussian blur
  vertical_kernel = GaussianKernel(sigma)
  vertical_blur = convolve(image, vertical_kernel)

  # Then calculate the derivative in the other direction
  horizontal_kernel = GaussianDerivative(sigma)
  flipped_horizontal_kernel = np.transpose(horizontal_kernel)
  horizontal_gradient = convolve(vertical_blur, flipped_horizontal_kernel)
  return (vertical_blur, horizontal_gradient)

def horizontal_gaussian(image, sigma):
  # First perform a gaussian blur
  vertical_kernel = GaussianKernel(sigma)
  horizontal_kernel = np.transpose(vertical_kernel)
  horizontal_blur = convolve(image, horizontal_kernel)

  # Then calculate the derivative in the other direction
  horizontal_kernel = GaussianDerivative(sigma)
  vertical_gradient = convolve(horizontal_blur, horizontal_kernel)
  return (horizontal_blur, vertical_gradient)

def covariance(image, vert_grad, horiz_grad, window):
  (height, width) = image.shape
  Z = np.zeros((height, width), dtype=object)
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

            ixx += vert.astype(np.int64) * vert.astype(np.int64)
            iyy += horiz.astype(np.int64) * horiz.astype(np.int64)
            ixy += vert.astype(np.int64) * horiz.astype(np.int64)
      
      Z[i][j] = np.array([ [ixx, ixy], [ixy, iyy] ], dtype=object)
  
  return Z

def find_corners(cov_mat, cornerness_val = 0.04):
  (height, width) = cov_mat.shape
  features = []
  # For each pixel, calculate the eigen values based on covariance matrix
  for i in range(0, height):
    for j in range(0, width):
      eval, _ = np.linalg.eig(cov_mat[i][j].astype(np.int64))
      val = (eval[0] * eval[1]) - (cornerness_val * ((eval[0] + eval[1]) ** 2))
      # Append to features including the pixel coordinates
      features.append((i, j, val))

  return features

def get_top_features(features: list, k_values = 50, val_distance = 5):
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
  # Program Constants
  sigma = 0.6
  window = 7

  # Get file
  Tk().withdraw()
  filename = askopenfilename()

  # Read the image as grayscale
  image = cv2.imread(filename, 0)

  # Calculate gaussian blur first
  (vertical_blur, horiz_grad) = vertical_gaussian(image, sigma)
  (horizontal_blur, vert_grad) = horizontal_gaussian(image, sigma)

  # Calculate covariance on the gradients of both vertical and horizontal
  cov_mat = covariance(image, vert_grad, horiz_grad, window)

  # Then get the features
  feature_list = find_corners(cov_mat)
  top_features = get_top_features(feature_list)

  # Add features to the image
  for (x,y, _) in top_features:
    cv2.putText(image, 'X', (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

  cv2.imshow("initial frame", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
