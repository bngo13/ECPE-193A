#Coding Credits: ECPE 124/293 students Zhao Do and Robert Miller
#Instructor: Vivek K. Pallipuram
#Cite: Digital Image Processing, University of the Pacific, 2024, Dr. Pallipuram
import cv2
import numpy as np
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import library as lb
from tkinter import filedialog
import math
import os
import argparse


class ClickCapture:
  def __init__(self):
    self.mouse_clicks = []  # List to store clicked locations

  def mouse_callback(self,event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
      self.mouse_clicks.append((x, y))
      print("Click at:", x, y)

  def analysis(self,filename):

  
    f=open('training.csv','a')
    image = cv2.imread(filename)
    file_name=os.path.basename(filename)
    if image is None:
      print("Error: Image not found.")
      return

    # Resize the image while maintaining aspect ratio
    target_width = 500
    aspect_ratio = image.shape[1] / image.shape[0]
    target_height = int(target_width / aspect_ratio)
    image = cv2.resize(image, (target_width, target_height))

    imgarea=image.shape[0]*image.shape[1]

    # Crop 30 pixels from the left
    image = image[:, 30:]

    # Convert the image to HSV color space for color-based segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for high-temperature colors (white, red, orange)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([11, 50, 50])
    upper_orange = np.array([25, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])

    # Create masks for red, orange, and white colors
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Combine the masks
    combined_mask = cv2.bitwise_or(mask_red1, mask_red2)
    combined_mask = cv2.bitwise_or(combined_mask, mask_orange)
    combined_mask = cv2.bitwise_or(combined_mask, mask_white)

    # Find contours in the combined mask (thermal areas)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height=image.shape[0]
    width=image.shape[1]

    binary_image = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)

# Draw the contours on the binary image
#  cv2.drawContours(binary_image, contours, -1, (255), thickness=cv2.FILLED)

    thresholdarea=0.01*height*width
    for contour in contours:
      if cv2.contourArea(contour) > thresholdarea:
        cv2.drawContours(binary_image, [contour], -1, (255), thickness=cv2.FILLED)
# Now 'binary_image' has contours filled with 255 and the rest as 0
    cv2.namedWindow("binary image")
    cv2.imshow('Binary Image', binary_image)
    cv2.waitKey(0)

    components,comp_labels,compimg = lb.CC_FloodFill(binary_image,height,width)
    print(comp_labels)

    cv2.namedWindow("components image")
    cv2.setMouseCallback("components image", self.mouse_callback)
  #cv2.waitKey(0)

    while True:
       cv2.imshow("components image",compimg.astype(np.uint8))
       if cv2.waitKey(1) == 27:  # Press Esc to exit
          break

    print(self.mouse_clicks)

    labels=[]
    for features in self.mouse_clicks:
      labels.append(compimg[int(features[1]),int(features[0])])
    print(labels)  
  
  #m00,xc,yc,lambda1,lambda2,theta = lb.Region_Properties(compimg,height,width,components,comp_labels)
    m00,xc,yc,lambda1,lambda2,theta,avg_hue,avg_sat = lb.Region_Properties2(compimg,hsv,height,width,len(labels),labels)

    perimeter=[]
    for label in labels:
      perimeter.append(lb.WallFollow2(compimg,height,width,label))

    print("distinct objects:")
    print(len(xc))
    color=(0,0,255) 

    for i in range(0,len(xc)):
      max_index=i
      if lambda1[i]<=0 or lambda2[i]<=0:
        print("not objs")
        print(comp_labels[i],m00[i],xc[i],yc[i],lambda1[i],lambda2[i],0.01*height*width)
        continue
      print(comp_labels[i],xc[i],yc[i],abs(lambda1[i]-lambda2[i])/lambda1[i],m00[i])
      ecc=abs(lambda1[i]-lambda2[i])/lambda1[i]
      objectclass=input('for location'+str(xc[i])+','+str(yc[i])+' enter object class')
      temperature=input('for location'+str(xc[i])+','+str(yc[i])+' enter estimated temperature')

      f.write('\n'+file_name+','+str(xc[i])+','+str(yc[i])+','+str(m00[i]/imgarea)+','+str(ecc)+','+str(theta[i])+','+str(perimeter[i])+','+str(avg_hue[i]) + ',' + str(avg_sat[i]) + ',' + objectclass+','+str(temperature))
      image = cv2.circle(image, (int(yc[max_index]),int(xc[max_index])), 20, (255,255,0), 2)
      x1=int(xc[i]+math.sqrt(lambda1[i])*math.cos(theta[i]))
      y1=int(yc[i]+math.sqrt(lambda1[i])*math.sin(theta[i]))
      image=cv2.line(image,(int(yc[i]),int(xc[i])),(y1,x1),color,2)
      x1=int(xc[i]-math.sqrt(lambda1[i])*math.cos(theta[i]))
      y1=int(yc[i]-math.sqrt(lambda1[i])*math.sin(theta[i]))
      image=cv2.line(image,(int(yc[i]),int(xc[i])),(y1,x1),color,2)

      x1=int(xc[i]+math.sqrt(lambda2[i])*math.cos(math.pi/2 + theta[i]))
      y1=int(yc[i]+math.sqrt(lambda2[i])*math.sin(math.pi/2 + theta[i]))
      image=cv2.line(image,(int(yc[i]),int(xc[i])),(y1,x1),color,2)
      #the other way
      x1=int(xc[i]-math.sqrt(lambda2[i])*math.cos(math.pi/2+theta[i]))
      y1=int(yc[i]-math.sqrt(lambda2[i])*math.sin(math.pi/2+theta[i]))
      image=cv2.line(image,(int(yc[i]),int(xc[i])),(y1,x1),color,2)
                   
    cv2.namedWindow("color image")
    cv2.imshow("color image",image)
    cv2.waitKey(0)
#close windows
    cv2.destroyAllWindows()
    f.close()
  #mouse_clicks=[]

def read_files_from_starting_point(x):
    # Open a file dialog to select a starting file
    starting_file = filedialog.askopenfilename(title="Select a starting file", filetypes=[("Text files", "*.jpg")])
    
    if not starting_file:
        print("No file selected.")
        return

    directory = os.path.dirname(starting_file)
    file_paths = []

    # List all text files in the directory
    all_files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])

    # Find the index of the starting file
    try:
        start_index = all_files.index(os.path.basename(starting_file))
    except ValueError:
        print("Selected file is not in the directory.")
        return

    # Read the next 20 files from the starting point
    for i in range(start_index, min(start_index + x, len(all_files))):
        current_file = os.path.join(directory, all_files[i])
        print(f"Reading {current_file}:")
        capture=ClickCapture()
        capture.analysis(current_file)

def read_files_with_offset(offset):
    # Open a file dialog to select the directory containing text files
    directory = filedialog.askdirectory(title="Select a directory containing text files")
    
    if not directory:
        print("No directory selected.")
        return

    # List all text files in the directory and sort them
    all_files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])

    # Calculate start and end index based on the offset
    start_index = offset * 10
    end_index = start_index + 10

    # Read the files from start_index to end_index
    selected_files = all_files[start_index:end_index]
    
    if not selected_files:
        print(f"No files to read for offset {offset}.")
        return

    for file_name in selected_files:
        current_file = os.path.join(directory, file_name)
        print(f"Reading {current_file}:")
        capture=ClickCapture()
        capture.analysis(current_file)

if __name__ == "__main__":
  ##Search for File##
 # parser = argparse.ArgumentParser(description="Read text files from a directory with an offset.")
  #parser.add_argument('offset', type=int, help='Offset to determine which set of files to read (0 for first 60, 1 for next 60, etc.)')
  parser = argparse.ArgumentParser(description="Read text files from a directory sequentially from a starting point (file selected by user). User provides the number of files to go over.")
  parser.add_argument('x', type=int, help='determine how many files to read one by one starting at the starting file')
  args = parser.parse_args()
  #read_files_with_offset(args.offset)
  read_files_from_starting_point(args.x)
  #analysis(f,filename)


