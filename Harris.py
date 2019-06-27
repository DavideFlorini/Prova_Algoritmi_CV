import cv2
import numpy as np
from matplotlib import pyplot as plt

im=cv2.imread('Harris\\IMG_20190429_104552794_HDR.jpg', 0)
im=cv2.resize(im,dsize=(0,0), fx=0.5, fy=0.5)
X = np.arange(0, 3, 1)
Y = np.arange(0, 3, 1)
X, Y = np.meshgrid(X, Y)
sigmax=5
sigmay=5

imx = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=5)
imy = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=5)

imx_sq=np.square(imx)
imy_sq=np.square(imy)
imxy=imx*imy

Sx2=cv2.filter2D(imx_sq, cv2.CV_64F, cv2.getGaussianKernel(3,1))
Sy2=cv2.filter2D(imy_sq, cv2.CV_64F, cv2.getGaussianKernel(3,1))
Sxy=cv2.filter2D(imxy, cv2.CV_64F, cv2.getGaussianKernel(3,1))

k=0.05
R=np.zeros(im.shape)
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        R[i,j]=np.linalg.det(np.array([[Sx2[i,j], Sxy[i,j]],[Sxy[i,j], Sy2[i,j]]]))-k*(np.trace(np.array([[Sx2[i,j], Sxy[i,j]],[Sxy[i,j], Sy2[i,j]]])))**2

keypoints=np.zeros(im.shape, dtype=np.dtype('uint8'))
flat=np.zeros(im.shape, dtype=np.dtype('uint8'))
borders=np.zeros(im.shape, dtype=np.dtype('uint8'))
th=0.1*R.max()

keypoints[R>th]=im[R>th]
flat[np.abs(R)<th]=im[np.abs(R)<th]
borders[R<-th]=im[R<-th]

plt.figure()
plt.imshow(R)

R=(R-R.min())/(R.max()-R.min())*255
R=R.astype('uint8')
cv2.imshow('keypoints', keypoints)
cv2.imshow('R', R)
cv2.imshow('flat', flat)
cv2.imshow('borders', borders)
cv2.imshow('im', im)

cv2.imwrite('Harris\\keypoints.jpg', keypoints)
cv2.imwrite('Harris\\flat.jpg', flat)
cv2.imwrite('Harris\\borders.jpg', borders)
cv2.imwrite('Harris\\R.jpg', R)



plt.show()
cv2.waitKey()

