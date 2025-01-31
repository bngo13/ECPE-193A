import cv2
import numpy as np
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import math
import os
#NOTE: IMPORT YOUR LIBRARY

def mouse_callback(in1: int, in2: int, in3: int, in4: int, in5):
  pass

def main():
   

  Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
  filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
  print(filename)

# Create a window and set the callback function
  cv2.namedWindow("Image")
  cv2.setMouseCallback("Image", mouse_callback)

# Display an image (or any placeholder)
 #load image
  img = cv2.imread(filename,0) #0 for grayscale

  #NOTE: Create a libra
 # topfeatures = lb.Corners(img)

  directory = os.path.dirname(filename)
 
  image = cv2.imread(filename,1) #1 for color

  #NOTE: OpenCV2 mouse clicks stores in (column,row) format
  for (y,x) in topfeatures:  
    cv2.putText(image, 'X', (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

  cv2.imshow("initial frame",image)

  cv2.destroyAllWindows()
 

 

if __name__ == "__main__":
  main()
