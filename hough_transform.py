import numpy as np
import cv2
from math import hypot, pi, cos, sin, ceil
from matplotlib import pyplot as plt



def hough(im, ang_res, rho_res, threshold):
    hy=int(180/ang_res)
    hx=int(2*(np.sqrt(im.shape[0]**2+im.shape[1]**2))/rho_res)
    him=np.zeros([hx, hy])
    #print(him)
    #print(hx,hy, him.shape)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j]>200:
                for n in np.arange(0, 180, ang_res):
                    ang=int(n/ang_res)
                    rho=int(hx/2+(i*cos((n)*np.pi/180)+j*sin((n)*np.pi/180))/rho_res)
                    him[rho, ang]=him[rho, ang]+1

    #him = cv2.GaussianBlur(him, (5, 5), 0)
    him = him.astype('float')
    him=him*255/him.max()
    him = him.astype('uint8')

    lines=np.argwhere(him > threshold)
    lines=lines.astype('float')
    lines[:, 1]=lines[:,1]*ang_res/180*np.pi
    lines[:, 0] = (lines[:, 0]*rho_res - hx/2*rho_res )

    return him, lines

#im = cv2.imread('sudoku.png',0)
#img = cv2.imread('a.jpg',0)
#im=cv2.resize(im, dsize=(0,0), fx=0.5, fy=0.5)
#blur = cv2.GaussianBlur(im,(5,5),0)
#im=cv2.Canny(im, 50, 200, 3)
#ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#im=np.eye(100)*255

# im=np.zeros([100,100])
# for i in range(0,im.shape[0]):
#     for j in range(0,im.shape[1]):
#         if j==im.shape[1]-i or j==i or j==10 or i==10 or j==im.shape[1]-10 or i==im.shape[0]-10:
#             im[i,j]=255
# cv2.imwrite('Hough\\im.png',im)

im=cv2.imread('Hough\\300px-Pentagon.png',0)
im=im.astype('float')
im=-1*im+255
im=im.astype('uint8')

rho_res=1
ang_res=0.1

him, lines = hough(im,ang_res,rho_res,threshold=100)

im0=np.zeros([im.shape[0], im.shape[1], 3], dtype=np.dtype('uint8'))
im0[:,:,0]=im
im0[:,:,1]=im
im0[:,:,2]=im

for rho, theta in lines:
     #cv2.circle(him, (theta, rho), 4, (255, 0, 255), 1)

     a=sin(theta)
     b=cos(theta)
     p0=(a*rho, b*rho)
     pt1=(int(p0[0]+10000*(-b)),int(p0[1]+10000*(a)))
     pt2=(int(p0[0]-10000*(-b)),int(p0[1]-10000*(a)))
     print(theta,rho, pt1, pt2)
     cv2.line(im0,pt1,pt2,(0,255,0),1)


plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

plt.imshow(him,extent=[0,180,him.shape[0]/2*rho_res,-him.shape[0]/2*rho_res])
plt.xlabel(r'$\theta (°)$', fontsize=18)
plt.ylabel(r'$\rho (pix)$', fontsize=18)
plt.title('Hough space',fontsize=25)
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(180/(him.shape[0]*rho_res))
ax.xaxis.set_label_position('top')



cv2.imshow('im',im)
cv2.imshow('im0',im0)
cv2.imshow('him',cv2.resize(him, dsize=(0,0), fx=0.75, fy=0.75))

cv2.imwrite('Hough\\hough_space.jpg',him)
cv2.imwrite('Hough\\hough.jpg',im)
cv2.imwrite('Hough\\im0.jpg',im0)


plt.show()
cv2.waitKey()
