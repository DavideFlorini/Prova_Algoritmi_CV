import cv2
import numpy as np

video_path='C:\\Users\\Davide\\Desktop\\prova_online_video\\output\\Salem,_NH_-_RT28_@_Lawrence_RD.mp4'
index1 = video_path.rfind('\\') + 1
index2 = video_path.rfind('.')

vid_name = video_path[index1:index2]

cap = cv2.VideoCapture(video_path)
if (cap.isOpened() == False):
    print("Error opening video stream or file")

ret, frame = cap.read()
f = 600 / max(frame.shape)
frame1 = cv2.resize(frame, dsize=(0, 0), fx=f, fy=f)
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vid_out = cv2.VideoWriter('Optical_Flow\\'+vid_name+'_OpticalFlow_comp.avi',fourcc, 30.0, (frame1.shape[1], frame1.shape[0]))
vid_out2 = cv2.VideoWriter('Optical_Flow\\'+vid_name+'_OpticalFlow.avi',fourcc, 30.0, (frame1.shape[1], frame1.shape[0]))

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        f = 600 / max(frame.shape)
        frame2 = cv2.resize(frame, dsize=(0, 0), fx=f, fy=f)
        out_arrow=frame2.copy()
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        frame1=frame1.astype('float')
        frame2=frame2.astype('float')

        It=cv2.GaussianBlur(frame2, (5, 5), sigmaX=1, sigmaY=1)-cv2.GaussianBlur(frame1, (5, 5), sigmaX=1, sigmaY=1)
        Ix=cv2.Sobel(frame2, ddepth=-1,  dx=1, dy=0, ksize=5)
        Iy=cv2.Sobel(frame2, ddepth=-1,  dx=0, dy=1, ksize=5)
        out=np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.dtype('float'))
        patch_size=(5,5)
        for i in range(int(patch_size[0]/2), frame1.shape[0]-int(patch_size[0]/2)):
            for j in range(int(patch_size[1]/2), frame1.shape[1]-int(patch_size[1]/2)):

                A=np.hstack((Ix[i-int(patch_size[0]/2):i+int(patch_size[0]/2)+1, j-int(patch_size[1]/2):j+int(patch_size[1]/2)+1].reshape((patch_size[0]*patch_size[1],1)), Iy[i-int(patch_size[0]/2):i+int(patch_size[0]/2)+1, j-int(patch_size[1]/2):j+int(patch_size[1]/2)+1].reshape((patch_size[0]*patch_size[1],1))))
                B=-It[i-int(patch_size[0]/2):i+int(patch_size[0]/2)+1,j-int(patch_size[1]/2):j+int(patch_size[1]/2)+1].reshape((patch_size[0]*patch_size[1],1))
                if np.linalg.det(A.T.dot(A)) !=0:
                    v=(np.linalg.inv(A.T.dot(A)).dot(A.T)).dot(B)
                    out[i, j, 2] = np.sqrt(v[0]**2+v[1]**2)*30*30
                    out[i, j, 0] = (np.arctan2(v[1], v[0])+np.pi)*255/(2*np.pi)
                    #if v[0]**2>0.001 and v[1]**2>0.001:
                    #    cv2.arrowedLine(out_arrow,(j,i), (j+100*v[1], i+100*v[0]), color=255,thickness=2)

        # It=(It-It.min())/(It.max()-It.min())*255
        # It=It.astype('uint8')
        # Ix=(Ix-Ix.min())/(Ix.max()-Ix.min())*255
        # Ix=Ix.astype('uint8')
        # Iy=(Iy-Iy.min())/(Iy.max()-Iy.min())*255
        # Iy=Iy.astype('uint8')
        #print(out.max())
        #out=(out-out.min())/(out.max()-out.min())*255
        #out=out.astype('uint8')
        # frame1=frame1.astype('uint8')
        # frame2=frame2.astype('uint8')
        # cv2.imshow('frame1', frame1)
        # cv2.imshow('frame2', frame2)
        out[:, :, 1] = np.ones(frame1.shape) * 255
        out=out.astype('uint8')
        out=cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
        comp=(out+out_arrow/2).astype('float')
        comp[comp>255]=255
        comp=comp.astype('uint8')

        cv2.imshow('comp', comp)
        vid_out.write(comp)
        vid_out2.write(out)

        cv2.imshow('out', out)

        #cv2.imshow('out_arrow', out_arrow)

        # cv2.imshow('Ix', Ix)
        # cv2.imshow('Iy', Iy)
        # cv2.imshow('It', It)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
        frame1=frame2.copy()
    else:
        vid_out.release()
        vid_out2.release()