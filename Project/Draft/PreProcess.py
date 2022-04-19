import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imgShow as disp

'''
Algorithm:

1. BGR -> GRAY
2. Gaussian Blurr with (21, 21) and 2, seems to be better.
3. Gaussian Adaptive with 5 and 2, seems to be better so far.
4. Edge finding with Laplacian
5. Dilation with MORPH_RECT (3, 3) and iterations=2.

'''

def PreProcess (img,
                gray=True,
                blur=[(21, 21), 2],
                threshold=['AdaptiveGaussian', 5, 2],
                edge=['Laplacian'],
                kernel=['MorphRect', (3, 3)],
                dilate=[2],
                log=[0, 1, 2, 3, 4], figsize=(5, 5)):

    if not gray:
        img=cv.cvtColor (img, cv.COLOR_BGR2GRAY)
    
    if 0 in log:
        title="gray" if not gray else "original"
        col="gray" if not gray else "viridis"
        disp.imgShow (img, title=title, cmap=col, figsize=figsize)

    if blur is not None:
        img=cv.GaussianBlur (img, blur[0], blur[1])

    if 1 in log:
        disp.imgShow (img, title=f"Blurred image: {blur}", figsize=figsize)
    
    if threshold is not None:
        if threshold[0].lower() in ["adaptivegaussian", 'adaptive', 'adaptive gaussian',
                                    'gaussian', 'gaus', 'gauss']:
            img=cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY_INV, threshold[1], threshold[2])
        else:
            print (f"method {threshold[0]} not understood")

    if 2 in log:
        disp.imgShow (img, title=f"Thresholded image: {threshold}", figsize=figsize)

    if edge is not None:
        if edge[0].lower() in ['laplacian', 'laplace']:
            img=cv.Laplacian(img, cv.CV_8UC1)
        elif edge[0].lower() in ['canny']:
            img=cv.Canny(img, *edge[1:])
    
    if 3 in log:
        disp.imgShow (img, title=f"Edged image: {edge}", figsize=figsize)

    if dilate is not None:
        assert kernel!=None, "Kernel cannot be None if dilate is not None"
        if kernel[0].lower() in ["morphrect", "morph rect", "rect"]:
            kernel=cv.getStructuringElement(cv.MORPH_RECT, kernel[1])
        
        img=cv.dilate(img, kernel, iterations=dilate[0])

    if 4 in log:
        disp.imgShow (img, title=f"", figsize=figsize)
    
    return img

def clear (img, kernel, stride, tol):
    l=img.shape[1]*tol
    flag=0
    while True:
        a=img[flag:flag+kernel].sum()
        if a>l:
            break
        flag+=stride

    return flag+kernel

def clearVoid (img, biny, kernel=5, stride=1, tol=0.1, all=True, down=0, up=0, left=0, right=0):
    assert len(biny.shape)==2, "Image channel must be 2"
    dirr=[]

    if down:
        dirr.append('dir')
    if up:
        dirr.append('up')
    if left:
        dirr.append('left')
    if right:
        dirr.append('right')

    if all:
        dirr=['up', 'down', 'left', 'right']

    for i in dirr:
        if i=='up':
            upr=clear(biny, kernel=kernel, stride=stride, tol=tol)
            img=img[upr:]
        elif i=='left':
            tmp=biny.T
            lft=clear(tmp, kernel=kernel, stride=stride, tol=tol)
            img=img[:, lft:, :]
        elif i=='down':
            tmp=np.flip(biny, 0)
            dwn=clear(tmp, kernel=kernel, stride=stride, tol=tol)
            img=img[:-dwn]
        else:
            tmp=np.flip(biny, 1).T
            rht=clear(tmp, kernel=kernel, stride=stride, tol=tol)
            img=img[:, :-rht, :]

    return img