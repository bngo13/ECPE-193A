import numpy as np
import cv2 
import math

def inrange(q0,i,q1,j,h,w):
  if (q0+i>=h) or (q0+i)<0 or (q1+j>=w) or (q1+j<0):
    return 0
  else:
    return 1 

def click_event(event, x, y, flags, params): 
  
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x,y)



def floodfill(image,height,width,seedx,seedy,new_color):
  frontier=[]
  old_color=image[seedy][seedx]
  if(old_color==new_color):
    return
  frontier.append([seedy,seedx])
  while len(frontier)!=0:
    q = frontier.pop(len(frontier)-1)
    for i in range(-1,2):
      for j in range(-1,2):
        if inrange(q[0],i,q[1],j,height,width):
          if image[q[0]+i][q[1]+j] == old_color:
            frontier.append([q[0]+i,q[1]+j])
            image[q[0]+i][q[1]+j]=new_color

  return image 

def floodfill_separate(image,height,width,seedx,seedy,new_color):
  """seedx is column, seedy is row"""
  frontier=[]
  old_color=image[seedy][seedx]
  outimg = image
  if(old_color==new_color):
    return
  frontier.append([seedy,seedx])
  outimg[seedy][seedx] = new_color
  while len(frontier)!=0:
    q = frontier.pop(len(frontier)-1)
    for i in range(-1,2):
      for j in range(-1,2):
        if inrange(q[0],i,q[1],j,height,width):
          if image[q[0]+i][q[1]+j] == old_color and outimg[q[0]+i][q[1]+j]!=new_color:
            frontier.append([q[0]+i,q[1]+j])
            outimg[q[0]+i][q[1]+j]=new_color

  return outimg 

def floodfill_separate_dt(image,outimg,height,width,seedx,seedy,new_color):
  """seedx is column, seedy is row"""
  frontier=[]
  old_color=image[seedy][seedx]
  if(old_color==new_color):
    return
  frontier.append([seedy,seedx])
  outimg[seedy][seedx] = new_color
  while len(frontier)!=0:
    q = frontier.pop(len(frontier)-1)
    for i in range(-1,2):
      for j in range(-1,2):
        if inrange(q[0],i,q[1],j,height,width):
          if image[q[0]+i][q[1]+j] == old_color and outimg[q[0]+i][q[1]+j]!=new_color:
            frontier.append([q[0]+i,q[1]+j])
            outimg[q[0]+i][q[1]+j]=new_color

  return outimg 


def Double_Threshold(img,height,width,thi,tlo):
  low_thresh = np.zeros((height,width),dtype=np.uint8)
  high_thresh = np.zeros((height,width),dtype=np.uint8)
  outimg = np.zeros((height,width),dtype=np.uint8)
  
  for i in range(0,height):
    for j in range(0,width):
      if(img[i][j]>tlo):
        low_thresh[i][j]=254
      if(img[i][j]>thi):
        high_thresh[i][j]=254

  for i in range(0,height):
    for j in range(0,width):
      if(high_thresh[i][j]==254):
        outimg=floodfill_separate_dt(low_thresh,outimg,height,width,j,i,255)

  return outimg




def CC_FloodFill(img,height,width):
  outimg = np.zeros((height,width),dtype=int)
  label = 40
  components=0
  comp_labels=[]
  for i in range(0,height):
    for j in range(0,width):
      if img[i][j]==255 and outimg[i][j]==0:
        outimg=floodfill_separate_dt(img,outimg,height,width,j,i,label) 
        comp_labels.append(label)
        components = components + 1
        label = label + 20

  return components,comp_labels,outimg


def Region_Properties2(img,hsvimg,height,width,num_comps,comp_labels):

  lambda1=[0]*num_comps
  lambda2=[0]*num_comps
  theta=[0]*num_comps
  
  m10=[0]*num_comps
  m01=[0]*num_comps 
  m00=[0]*num_comps
  m11=[0]*num_comps
  mu11=[0]*num_comps
  m02=[0]*num_comps
  m20=[0]*num_comps
  mu02=[0]*num_comps
  mu20=[0]*num_comps
  xc=[0]*num_comps
  yc=[0]*num_comps
  avg_hue=[0]*num_comps
  avg_sat=[0]*num_comps

  for index in range(0,num_comps):
    comp=comp_labels[index]
    rows,cols=np.where(img==comp)

    for(i,j) in zip(rows,cols): 
      m10[index]=m10[index]+i
      m01[index]=m01[index]+j
      m00[index]=m00[index]+1
      m11[index]=m11[index]+i*j
      m02[index] = m02[index]+j*j
      m20[index] = m20[index]+i*i
      avg_hue[index] = avg_hue[index]+hsvimg[i,j,0]
      avg_sat[index] = avg_sat[index]+hsvimg[i,j,1]
    i=index
    avg_hue[i]=avg_hue[i]/m00[i]
    avg_sat[i]=avg_sat[i]/m00[i]
    xc[i]=m10[i]/m00[i] 
    yc[i]=m01[i]/m00[i] 
    mu20[i]=m20[i]-xc[i]*m10[i] 
    mu02[i]=m02[i]-yc[i]*m01[i]
    mu11[i]=m11[i]-yc[i]*m10[i]
    lambda1[i]=(1/(2*m00[i]))*(mu20[i]+mu02[i]+math.sqrt((mu20[i]-mu02[i])**2+4*(mu11[i]**2))) 
    lambda2[i]=(1/(2*m00[i]))*(mu20[i]+mu02[i]-math.sqrt((mu20[i]-mu02[i])**2+4*(mu11[i]**2))) 
    theta[i]=0.5*math.atan2(2*mu11[i],mu20[i]-mu02[i])

#     print("component " + str(comp_labels[i]) + " center: "+str(xc[i])+","+str(yc[i]))

  
  return m00,xc,yc,lambda1,lambda2,theta,avg_hue,avg_sat

def isFrontOn(img,height,width,x,y,dir,label):
  if dir==0: #north
    if inrange(x,-1,y,0,height,width):
      if img[x-1][y]==label:
        return 1
  elif dir==1: #east
    if inrange(x,0,y,1,height,width):
      if img[x][y+1]==label:
        return 1
  elif dir==2: #south
    if inrange(x,1,y,0,height,width):
      if img[x+1][y]==label:
        return 1
  elif dir==3: #west
    if inrange(x,0,y,-1,height,width):
      if img[x][y-1]==label:
        return 1
  
  return 0



def isLeftOn(img,height,width,x,y,dir,label):
  if dir==0: #north
    if inrange(x,0,y,-1,height,width):
      if img[x][y-1]==label:
        return 1
  elif dir==1: #east
    if inrange(x,-1,y,0,height,width):
      if img[x-1][y]==label:
        return 1
  elif dir==2: #south
    if inrange(x,0,y,1,height,width):
      if img[x][y+1]==label:
        return 1
  elif dir==3: #west
    if inrange(x,1,y,0,height,width):
      if img[x+1][y]==label:
        return 1

  return 0

def moveFwd(dir,x,y):
  if dir==0:
    x=x-1
  elif dir==1:
    y=y+1
  elif dir==2:
    x=x+1
  elif dir==3:
    y=y-1

  return x,y

      
def WallFollow2(img,height,width,label):

  dir=0 #North
  x=0
  y=0
  flag=0
  perimeter=[]
  for i in range(0,height):
    for j in range(0,width):
      if img[i][j]==label:
        x=i
        y=j
        flag=1
        break
    if flag:
      break

 # col_img[x][y]=np.array(color)
  perimeter=0
  perimeter=perimeter+1 
  while isFrontOn(img,height,width,x,y,dir,label):
    dir=(dir+1)%4

  #turn right
  dir=(dir+1)%4
  x_orig=x
  y_orig=y
  
  while 1:
    if isLeftOn(img,height,width,x,y,dir,label): 
      if(dir==0):
        dir=3
      else:
        dir=(dir-1)%4
      x,y=moveFwd(dir,x,y)
      perimeter=perimeter+1
      #col_img[x][y]=np.array(color)
    elif not isFrontOn(img,height,width,x,y,dir,label):
      dir=(dir+1)%4
    else:
      x,y=moveFwd(dir,x,y)
      #col_img[x][y]=np.array(color)
      perimeter=perimeter+1
    if x_orig==x and y_orig==y and dir==0:
      break

  return perimeter
     
 

  
