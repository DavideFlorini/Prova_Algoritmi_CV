import cv2
import numpy as np
from matplotlib import pyplot as plt

img=np.zeros([500,500])
im_dst=np.zeros([1000,1000])
img=cv2.rectangle(img, (100, 100), (400,400), 255, -1)



points = np.array([[100, 100,1], [100, 400,1], [400, 100,1], [400, 400,1]])
points_a = np.array([[100, 100, 400, 400], [100, 400,100,400], [1,1,1,1]])

# points_dst=points@np.array([[0.5,2,0],[0.5,0.8,0.001],[10,15,1]])
# points_dst=points_dst.astype('uint16')
# print(points_dst)
points_dst = np.array([[50, 50,1], [100, 500,1], [350, 100,1], [500, 600,1]])
points_dst_a = np.array([[50, 100, 350, 500], [50, 500,100,600], [1,1,1,1]])

h_lib, status = cv2.findHomography(points[:,0:-1], points_dst[:,0:-1])

#h=np.linalg.inv(points.T@points)@points.T@points_dst
h=np.linalg.pinv(points)@points_dst
h=h.T.astype('float32')
#h1=points_dst_a@points.T@np.linalg.inv(points@points.T)
print(h)
print(h_lib)
im_dst = cv2.warpPerspective(img, h, (im_dst.shape[0],im_dst.shape[1]))
im_dst2 = cv2.warpPerspective(img, h_lib, (im_dst.shape[0],im_dst.shape[1]))
for p in points_dst:
    cv2.circle(im_dst, (p[0],p[1]),radius=10,color=255, thickness=5)
    cv2.circle(im_dst2, (p[0],p[1]),radius=10,color=255, thickness=5)

# cv2.line(im_dst,(0,int(41.66)),(1000,int(0.16*1000+41)),color=255,thickness=1)
# cv2.line(im_dst,(0,int(400)),(1000,int(0.25*1000+400)),color=255,thickness=1)
# cv2.line(im_dst,(0,int(-400)),(1000,int(9*1000-400)),color=255,thickness=1)
# cv2.l-400
# line(im_dst,(0,int(-1066)),(1000,int(3.333*1000-1066)),color=255,thickness=1)
m1=0.166666
m2=0.25
m3=9
m4=3.33
q1=41.666
q2=400
q3=-400
q4=-1066
x1=(-q1+q2)/(m1-m2)
y1=m1*x1+q1
x2=(-q3+q4)/(m3-m4)
y2=m3*x2+q3

cv2.circle(im_dst2, (int(x1),int(y1)),radius=10,color=255, thickness=5)
print(x1,y1,x2,y2)
# h13=-48.1
# h23=-69.9588
# h31=-0.00016461#0.0028
# h32=-0.000699588#-0.0035
#
# h11=x1*h31 #x position of the point at infinity 1
# h21=y1*h31  #y position of the point at infinity 1
# h12=x2*h32 #x position of the point at infinity 2
# h22=y2*h32 #y position of the point at infinity 2
# h33=1
#
#
#
# h=np.array([[h11,h12,h13],[h21,h22,h23],[h31,h32,h33]],dtype=np.float32)
# print(h)
# print(h_lib[0,0]/h_lib[2,0], h_lib[1,0]/h_lib[2,0], h_lib[0,1]/h_lib[2,1], h_lib[1,1]/h_lib[2,1])

# im_dst3 = cv2.warpPerspective(img, h, (im_dst.shape[0],im_dst.shape[1]))

plt.figure()
plt.subplot(121)
plt.imshow(im_dst)
plt.subplot(122)
plt.imshow(im_dst2)
# plt.subplot(223)
# plt.imshow(im_dst3)
plt.show()