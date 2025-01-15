import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfile

# Get chosen file
Tk().withdraw()
fileName = askopenfile()
print(fileName.name)

# Load image
imageData = cv2.imread(fileName.name, 0)

cv2.imshow("Sheesh", imageData)
cv2.waitKey(0)

cv2.destroyAllWindows()