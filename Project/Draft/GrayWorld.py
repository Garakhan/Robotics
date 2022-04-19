import numpy as np

def grayWorld (img):
    
    img=np.uint32(img)
    K=img.mean()
    avr=img.mean(axis=0)
    img=img*K/avr.mean(axis=0)[None, None, :]
    # print (img.max())
    # print (np.uint8(img).max())
    # img=cv.normalize(img, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC3)

    
    return np.uint8(img)