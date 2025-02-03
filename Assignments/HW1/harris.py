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

  convolution_img = np.empty_like(image)

  for i in range(0, img_shape[0]):
    for j in range(0, img_shape[1]):
      pixel_sum = 0
      for ki in range(0, kernel_shape[0]):
        for kj in range(0, kernel_shape[1]):
          offset_i = -1 * (kernel_shape[0] // 2) + ki
          offset_j = -1 * (kernel_shape[1] // 2) + kj
          if (i + offset_i >= 0 and i + offset_i < img_shape[0] and j + offset_j >= 0 and j + offset_j < img_shape[1]):
            pixel_sum += image[i + offset_i, j + offset_j] * kernel[ki, kj]
      convolution_img[i][j] = pixel_sum

  return convolution_img

def vertical_gaussian(image, sigma):
  vertical_kernel = GaussianKernel(sigma)
  vertical_blur = convolve(image, vertical_kernel)

  horizontal_kernel = GaussianDerivative(sigma)
  flipped_horizontal_kernel = np.transpose(horizontal_kernel)
  horizontal_gradient = convolve(vertical_blur, flipped_horizontal_kernel)
  return (vertical_blur, horizontal_gradient)

def horizontal_gaussian(image, sigma):
  vertical_kernel = GaussianKernel(sigma)
  horizontal_kernel = np.transpose(vertical_kernel)
  horizontal_blur = convolve(image, horizontal_kernel)

  horizontal_kernel = GaussianDerivative(sigma)
  vertical_gradient = convolve(horizontal_blur, horizontal_kernel)
  return (horizontal_blur, vertical_gradient)

def covariance(image, vert_grad, horiz_grad, window):
  (height, width) = image.shape
  Z = []
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
      
      Z.append(
        [
          [ixx, ixy], 
          [ixy, iyy]
        ]
      )
  
  return Z

def find_corners(image, cov_mat, cornerness_val = 0.04):
  (height, width) = image.shape
  evalue, evect = np.linalg.eig(cov_mat)

  features = np.zeros((height, width))

  i = 0
  j = 0
  for eval in evalue:
    if j >= width:
      i += 1
      j = 0
    
    val = (eval[0] * eval[1]) - (cornerness_val * ((eval[0] + eval[1]) ** 2))
    features[i][j] = val

    j += 1
  
  return features
  

def main():
  # Get file
  Tk().withdraw()
  filename = askopenfilename()

  # Read the image as grayscale
  image = cv2.imread(filename, 0)

  (vertical_blur, horiz_grad) = vertical_gaussian(image, 1)
  (horizontal_blur, vert_grad) = horizontal_gaussian(image, 1)

  cov_mat = covariance(image, vert_grad, horiz_grad, 7)

  top_features = find_corners(image, cov_mat)

  breakpoint()

  cv2.waitKey(0)
  cv2.destroyAllWindows()



if __name__ == "__main__":
  main()
