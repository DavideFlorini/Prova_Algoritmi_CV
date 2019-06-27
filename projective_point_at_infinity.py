import cv2
import numpy as np
from matplotlib import pyplot as plt

img=np.zeros([500,500])
im_dst=np.zeros([1000,1000])
img=cv2.rectangle(img, (100, 100), (400,400), 255, -1)

#define trasformation matrix (if h12 and/or h21 are zero==> points at infinity: x_1 and y_2 = 0)
theta=0*np.pi/180

#h11=np.cos(theta)
#h12=np.sin(theta)
#h21=-np.sin(theta)
#h22=np.cos(theta)

h13=300
h23=300
h31=0.0025#0.0028
h32=0.001#-0.0035

h11=200*h31 #x position of the point at infinity 1
h21=70*h31  #y position of the point at infinity 1
h12=200*h32 #x position of the point at infinity 2
h22=400*h32 #y position of the point at infinity 2
h33=1-h31*img.shape[0]/2-h32*img.shape[1]/2



h=np.array([[h11,h12,h13],[h21,h22,h23],[h31,h32,h33]],dtype=np.float32)
#####################################

print(h)
im_dst = cv2.warpPerspective(img, h, (im_dst.shape[0],im_dst.shape[1]))


if h31!=0:
    cv2.circle(im_dst,(int(h11/h31), int(h21/h31)), radius=10, color=255,thickness=5)
    #print('x1=', int(h11 / h31), ' y1=', int(h21 / h31))
    pt=h@(np.array([[100],[400],[1]]))
    pt=pt/pt[2]
    pt=pt[0:2].astype('uint32')
    cv2.line(im_dst, (int(h11 / h31),int(h21 / h31)),  (pt[0],pt[1]), thickness=2, color=255, lineType=cv2.LINE_AA)
    pt=h@(np.array([[100],[100],[1]]))
    pt=pt/pt[2]
    pt=pt[0:2].astype('uint32')
    cv2.line(im_dst, (int(h11 / h31),int(h21 / h31)),  (pt[0],pt[1]), thickness=2, color=255, lineType=cv2.LINE_AA)
if h32!=0:
    cv2.circle(im_dst,(int(h12/h32), int(h22/h32)), radius=10, color=255,thickness=5)
    #print('x2=',int(h12/h32), ' y2=',int(h22/h32))
    pt=h@(np.array([[400],[100],[1]]))
    pt=pt/pt[2]
    pt=pt[0:2].astype('uint32')
    cv2.line(im_dst, (int(h12 / h32),int(h22 / h32)),  (pt[0],pt[1]), thickness=2, color=255, lineType=cv2.LINE_AA)
    pt=h@(np.array([[100],[100],[1]]))
    pt=pt/pt[2]
    pt=pt[0:2].astype('uint32')
    cv2.line(im_dst, (int(h12 / h32),int(h22 / h32)),  (pt[0],pt[1]), thickness=2, color=255, lineType=cv2.LINE_AA)

h = np.array([[1, 0, 100], [0, 1, 100], [0, 0, 1]], dtype=np.float32)
cv2.line(img, ( 0, int(-h31/h32*0-h33/h32)), (1000,int(-h31/h32*1000-h33/h32)), color=255, thickness=1, lineType=cv2.LINE_AA)
#print(h31/h32,(-h31/h32*0-h33/h32),(-h31/h32*img.shape[0]-h33/h32))
cv2.imshow('img',img)
#cv2.imshow('persp',im_dst)
plt.figure()
plt.style.use('grayscale')
plt.imshow(im_dst)
plt.show()
cv2.waitKey()