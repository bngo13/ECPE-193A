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

def horizontal_gradient(image, sigma):
  vertical_kernel = GaussianKernel(sigma)
  temp_horizontal = convolve(image, vertical_kernel)

  horizontal_kernel = GaussianDerivative(sigma)
  flipped_horizontal_kernel = np.transpose(horizontal_kernel)
  horizontal = convolve(temp_horizontal, flipped_horizontal_kernel)
  return horizontal

def vertical_gradient(image, sigma):
  vertical_kernel = GaussianKernel(sigma)
  flipped_vertical_kernel = np.transpose(vertical_kernel)
  temp_vertical = convolve(image, flipped_vertical_kernel)

  horizontal_kernel = GaussianDerivative(sigma)
  vertical = convolve(temp_vertical, horizontal_kernel)
  return vertical
  pass

def covariance(image, window):
  # Get image shape (height, width)
  img_shape = image.shape
  image_height = img_shape[0]
  image_width = img_shape[1]
  for i in range(0, image_height):
    for j in range(0, image_width):
      x_squared = 0
      y_squared = 0
      xy = 0

      w = window // 2

      for offset_i in range(-w, w):
        for offset_j in range(-w, w):
          if (i + offset_i >= 0 and j + offset_j >= 0 and i + offset_i < image_height and j + offset_j < image_width):
            pass



def main():
  # Get file
  Tk().withdraw()
  filename = askopenfilename()

  # Read the image as grayscale
  image = cv2.imread(filename, 0)

  image = np.array([[8, 24, 48, 32, 16]])
  kernel = np.array([[1/4, 1/2, 1/4]])

  new_image = convolve(image, kernel)
  breakpoint()


if __name__ == "__main__":
  main()
