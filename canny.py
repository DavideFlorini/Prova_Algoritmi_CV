import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter




def canny(im, thh, thl, sigma):
    plotta=0
    sigmax = sigma
    sigmay = sigma
    kernel_dim=3*max(sigmax, sigmay)
    if kernel_dim%2 == 0:
        kernel_dim=kernel_dim+1
    X = np.arange(0, kernel_dim, 1)
    Y = np.arange(0, kernel_dim, 1)
    X, Y = np.meshgrid(X, Y)
    GAU=1/(2*np.pi*sigmax*sigmay)*np.exp(-((X-np.mean(X[1,:]))**2/(2*sigmax**2))-((Y-np.mean(Y[:,1]))**2/(2*sigmay**2)))
    DOGx=-(X-np.mean(X[1,:])) * 1/(sigmax**2) * 1/(2*np.pi*sigmax*sigmay)*np.exp(-((X-np.mean(X[1,:]))**2/(2*sigmax**2))-((Y-np.mean(Y[:,1]))**2/(2*sigmay**2)))
    DOGy=-(Y-np.mean(Y[:,1])) * 1/(sigmay**2) * 1/(2*np.pi*sigmax*sigmay)*np.exp(-((X-np.mean(X[1,:]))**2/(2*sigmax**2))-((Y-np.mean(Y[:,1]))**2/(2*sigmay**2)))

    if plotta==1:
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, GAU, cmap=cm.plasma, linewidth=1,alpha=0.5)
        cset = ax.contour(X, Y, GAU, zdir='z', offset=GAU.min() ,cmap=cm.plasma)
        cset = ax.contour(X, Y, GAU, zdir='x', offset=X.min() , cmap=cm.plasma)
        cset = ax.contour(X, Y, GAU, zdir='y', offset=Y.max() ,cmap=cm.plasma)
        ax.set_zlim(GAU.min(), GAU.max())
        plt.xlabel(r'X')
        plt.ylabel(r'Y')
        plt.title(r'$\frac{1}{2 \pi \cdot \sigma_x \sigma_y} \cdot e^{{-\frac{(X-\mu_X)^2} {2 \cdot {\sigma_x}^2}} - \frac{(Y-\mu_Y)^2}{2 \cdot {\sigma_y}^2}}}$',fontsize=20)
        #plt.title(r'$ -(X-\mu_x) \frac{1}{2 \pi \cdot {\sigma_x}^2} \cdot e^{{-\frac{(X-\mu_X)^2} {2 \cdot {\sigma_x}^2}} - \frac{(Y-\mu_Y)^2}{2 \cdot {\sigma_y}^2}}}$',fontsize=20)

        fig = plt.figure(2)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, DOGx, cmap=cm.plasma, linewidth=1,alpha=0.8)
        cset = ax.contour(X, Y, DOGx, zdir='z', offset=DOGx.min() ,cmap=cm.plasma)
        cset = ax.contour(X, Y, DOGx, zdir='x', offset=X.min() , cmap=cm.plasma)
        cset = ax.contour(X, Y, DOGx, zdir='y', offset=Y.max() ,cmap=cm.plasma)
        ax.set_zlim(DOGx.min(), DOGx.max())
        plt.xlabel(r'X')
        plt.ylabel(r'Y')
        plt.title(r'$\frac{\partial{\frac{1}{2 \pi \cdot \sigma_x \sigma_y} \cdot e^{{-\frac{(X-\mu_X)^2} {2 \cdot {\sigma_x}^2}} - \frac{(Y-\mu_Y)^2}{2 \cdot {\sigma_y}^2}}}}{\partial{x}}$', fontsize=20)

        fig = plt.figure(3)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, DOGy, cmap=cm.plasma, linewidth=1, alpha=0.8)
        cset = ax.contour(X, Y, DOGy, zdir='z', offset=DOGy.min() ,cmap=cm.plasma)
        cset = ax.contour(X, Y, DOGy, zdir='x', offset=X.min() , cmap=cm.plasma)
        cset = ax.contour(X, Y, DOGy, zdir='y', offset=Y.max() ,cmap=cm.plasma)
        ax.set_zlim(DOGy.min(), DOGy.max())
        plt.xlabel(r'X')
        plt.ylabel(r'Y')
        plt.title(r'$\frac{\partial{\frac{1}{2 \pi \cdot \sigma_x \sigma_y} \cdot e^{{-\frac{(X-\mu_X)^2} {2 \cdot {\sigma_x}^2}} - \frac{(Y-\mu_Y)^2}{2 \cdot {\sigma_y}^2}}}}{\partial{y}}$',fontsize=20)
        plt.show()

    imx = cv2.filter2D(im,cv2.CV_64F,DOGx)
    imy = cv2.filter2D(im,cv2.CV_64F,DOGy)


    #imx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=9)
    #imy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=9)

    dir = np.arctan2(imy,imx)*180/np.pi+180
    mag=np.sqrt(imx**2+imy**2)
    mag=(mag-mag.min()) / (mag.max() - mag.min())* 255

    img_HSV=np.ones([dir.shape[0], dir.shape[1], 3])
    img_HSV[:,:,0] = dir/2
    img_HSV[:,:,1] = img_HSV[:,:,1]*255
    img_HSV[:,:,2] = mag


    cv2.imshow('grad', cv2.cvtColor(img_HSV.astype('uint8'), cv2.COLOR_HSV2BGR))
    cv2.imwrite('Canny\\gradient.jpg', cv2.cvtColor(img_HSV.astype('uint8'), cv2.COLOR_HSV2BGR))

    cv2.imshow('mag', img_HSV[:,:,2].astype('uint8'))
    cv2.imwrite('Canny\\magnitude.jpg', img_HSV[:,:,2].astype('uint8'))

    #imx=(imx-imx.min())/(imx.max()-imx.min())*255
    #imy=(imy-imy.min())/(imy.max()-imy.min())*255
    #cv2.imshow('imx', imx.astype('uint8'))
    #cv2.imshow('imy', imy.astype('uint8'))
    edges=mag.astype('uint8').copy()

    for i in range(1, mag.shape[0]-1):
        for j in range(1, mag.shape[1]-1):
            if(abs(imx[i,j])>=abs(imy[i,j])):
                if imx[i,j]!=0:
                    m=imy[i,j]/imx[i,j]
                    if m>=0:
                        mag1=mag[i-1,j+1]*m+mag[i,j+1]*(1-m)
                        mag2=mag[i+1,j-1]*m+mag[i,j-1]*(1-m)
                    if m<0:
                        m=abs(m)
                        mag1=mag[i+1,j+1]*m+mag[i,j+1]*(1-m)
                        mag2=mag[i-1,j-1]*m+mag[i,j-1]*(1-m)
                else:
                    mag1=0
                    mag2=0
            if(abs(imy[i,j])>abs(imx[i,j])):
                if imy[i,j]!=0:
                    m=imx[i,j]/imy[i,j]
                    if m>=0:
                        mag1=mag[i-1,j+1]*m+mag[i-1,j]*(1-m)
                        mag2=mag[i+1,j-1]*m+mag[i+1,j]*(1-m)
                    if m<0:
                        m=abs(m)
                        mag1=mag[i-1,j-1]*m+mag[i-1,j]*(1-m)
                        mag2=mag[i+1,j+1]*m+mag[i+1,j]*(1-m)
                else:
                    mag1=0
                    mag2=0

            comp=np.array([mag2, mag[i,j], mag1])
            compmax=np.argmax(comp)
            if compmax!=1:
                edges[i, j] = 0
    edges1=edges.copy()
    # plt.figure(1)
    # plt.imshow(edges)
    # plt.show()

    edges[edges<thl]=0
    edges[edges>thh]=255

    # strongEdges = (edges >= thh)
    # thresholdedEdges = np.array(strongEdges, dtype=np.uint8) + (edges > thl)
    # finalEdges = strongEdges.copy()
    # currentPixels = []
    # for r in range(1, im.shape[0] - 1):
    #     for c in range(1, im.shape[1] - 1):
    #         if thresholdedEdges[r, c] != 1:
    #             continue  # Not a weak pixel
    #
    #         # Get 3x3 patch
    #         localPatch = thresholdedEdges[r - 1:r + 2, c - 1:c + 2]
    #         patchMax = localPatch.max()
    #         if patchMax == 2:
    #             currentPixels.append((r, c))
    #             finalEdges[r, c] = 1
    # while len(currentPixels) > 0:
    #     newPix = []
    #     for r, c in currentPixels:
    #         for dr in range(-1, 2):
    #             for dc in range(-1, 2):
    #                 if dr == 0 and dc == 0: continue
    #                 r2 = r + dr
    #                 c2 = c + dc
    #                 if thresholdedEdges[r2, c2] == 1 and finalEdges[r2, c2] == 0:
    #                     # Copy this weak pixel to final result
    #                     newPix.append((r2, c2))
    #                     finalEdges[r2, c2] = 1
    #     currentPixels = newPix

    return edges#np.asmatrix(finalEdges*255, dtype='uint8')



    # for i in range(1, edges.shape[0]-1):
    #     for j in range(1, edges.shape[1]-1):
    #         patch=edges[i-1:i+2, j-1:j+2]
    #         # if patch.max()>thh:
    #         #     conn=1
    #         if patch.max()>=thh and edges[i,j]>=thl:
    #             edges[i,j]=255
    #         # else:
    #         #     edges[i,j]=0
    #
    #     #edges[edges!=0]=255

    # return edges, edges1

im=cv2.imread('Canny\\11f3bd8.png', 0)
#im=im[0:-100, :]

#cv2.imshow('im', im)
# im=cv2.resize(im, dsize=(0,0), fx=0.1, fy=0.1)
edges=canny(im.copy(), 70, 30, 3)
cv2.imshow('edges', edges)
cv2.imshow('canny', cv2.Canny(im,150,80,7))
cv2.imwrite('Canny\\canny_mio.jpg ', edges)
#cv2.imwrite('Canny\\lena_maxima_suppression.jpg ', edges1)

cv2.waitKey()