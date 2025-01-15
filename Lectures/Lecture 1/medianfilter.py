import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfile

Tk().withdraw()
filename = askopenfile().name

imagedata = cv2.imread(filename, 0)

newimage = np.zeros((len(imagedata), len(imagedata[0])), np.uint8)

for i in range(0, len(imagedata)):
    for j in range(0, len(imagedata[0])):
        medList = [imagedata[i][j]]
        
        if (i - 1 >= 0):
            # right
            medList.append(imagedata[i - 1][j])
        
        if (i + 1 < len(imagedata)):
            # left
            medList.append(imagedata[i + 1][j])
        
        if (j - 1 >= 0):
            # up
            medList.append(imagedata[i][j - 1])
        
        if (j + 1 < len(imagedata[0])):
            # down
            medList.append(imagedata[i][j + 1])
        
        if (i - 1 >= 0 and j - 1 >= 0):
            # upright
            medList.append(imagedata[i - 1][j - 1])

        if (j - 1 >= 0 and i + 1 < len(imagedata)):
            # upleft
            medList.append(imagedata[i + 1][j - 1])
        
        if (j + 1 < len(imagedata[0]) and i + 1 < len(imagedata)):
            # downleft
            medList.append(imagedata[i + 1][j + 1])
        
        if (i - 1 >= 0 and j + 1 < len(imagedata[0])):
            # downright
            medList.append(imagedata[i - 1][j + 1])
        
        newimage[i][j] = np.median(medList)

cv2.imshow("asdf", newimage)
cv2.waitKey(0)

cv2.destroyAllWindows()