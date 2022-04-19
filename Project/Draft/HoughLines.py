from readline import append_history_file
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from collections import deque
from itertools import islice

def HoughLinesP (img, orig, args=[1, np.pi/180, 80, 30, 10], draw=True, only_draw=False, color=[255, 0, 0], trail=[8, 3]):

    Lines=cv.HoughLinesP(img, rho=args[0], theta=args[1], threshold=args[2], \
        minLineLength=args[3], maxLineGap=args[4]).reshape(-1, 4)
    # print (Lines)
    if draw:
        for i in range(len(Lines)):
            cv.line (orig, (Lines[i][0], Lines[i][1]),
                    (Lines[i][2], Lines[i][3]), color, *trail)
        return orig, Lines

    if not only_draw: 
        return Lines

def findAngles (lines, tol=5):

    angles={i:0 for i in np.arange(0, 2*np.pi, np.pi/180)}
    anglesPerLine={i:"PASSED" for i in range(len(lines))}
    for i in range(len(lines)):
        p=lines[i]
        if np.sqrt((p[3]-p[1])**2+(p[2]-p[0])**2)<tol:
            continue
        tang=(p[3]-p[1])/(p[2]-p[0]+1e-6)
        ang=np.arctan(tang)
        if ang<0:
            ang+=np.pi
        anglesPerLine[i]=ang
        # print (list(angles.keys(i)))
        # print (ang)
        idx=np.isclose(list(angles.keys()), ang, atol=0.01)
        if not True in idx:
            idx=np.isclose(list(angles.keys()), ang, atol=0.02)
        if not True in idx:
            idx=np.isclose(list(angles.keys()), ang, atol=0.03)
        if True in idx:
            angles[list(angles.keys())[idx.tolist().index(True)]]+=1
    
    return angles, anglesPerLine

def distLines (lines, shape):
    q1, q2, q3, q4=[], [], [], []
    h, w=int(shape[0]/2), int(shape[1]/2)
    height, width=np.arange(0, shape[0]), np.arange(0, shape[1])
    fir=[height[:h], width[:w]]
    sec=[height[:h], width[w:]]
    # thi=[height[h:], width[:w]]
    # fou=[height[h:], width[w:]]

    for line in lines:
        if line[0] in fir[1]:
            if line[1] in fir[0]:
                q1.append(line)
            else:
                q3.append(line)
        elif line[0] in sec[1]:
            if line[1] in sec[0]:
                q2.append(line)
            else:
                q4.append(line)
    
    return q1, q2, q3, q4

def getMax(ang, n=4):
    ang=ang.copy()
    store={}
    for _ in range(n):
        store[ang.index(max(ang))]=max(ang)
        ang[ang.index(max(ang))]=0
    
    return store

def splitVH (angles):
    hor, ver={}, {}
    for (alpha, c) in angles.items():
        if abs(90-alpha)<=45:
            hor[alpha]=c
        else:
            ver[alpha]=c

    return ver, hor
    

def makeAvg (angles, weights="linear", power=1, tol=45):
    if not type(list(angles)[0])==dict:
        angles=[angles]
    
    store=[]
    # print (angles)
    
    for ang in angles:
        # print ("ang:", ang)
        keys=list(ang)
        for k in keys:
            if ang[k]==0 or ang[k] is None:
                del ang[k]
                continue
            if abs(k-180)<=tol:
                ang[k-180]=ang.pop(k)
        # print (ang)
        # print ("length:", len(ang))
        if len(ang)==0:
            store.append({None:None})
            continue
           
        if weights=='linear':
            w=np.array(list(ang.values()))**power

        # print ("weights: ", w)
        ave=np.average(list(ang.keys()), weights=w)
        # print (ave)
        if ave<0:
            ave+=180
        store.append({ave:np.sum(w)})
    
    return store

def makeTemplate (angles, average=True):
    UR, BL=[], []
    P1, P2, P3, P4=[], [], [], []
    a, b, c, d=angles
    a[0].update(b[0])
    # print (a[0], a[1])
    a[1].update(c[1])
    d[0].update(c[0])
    d[1].update(b[1])

    UR.extend(list(makeAvg(a[0])[0].keys()))
    UR.extend(list(makeAvg(a[1])[0].keys()))
    BL.extend(list(makeAvg(d[0])[0].keys()))
    BL.extend(list(makeAvg(d[1])[0].keys()))

    return UR, BL

def makeLine (angles, x1, y1, x4, y4):

    l1={"k":np.tan(angles[0][0]*np.pi/180), 'x':x1, 'y':y1}
    l2={"k":np.tan(angles[0][1]*np.pi/180), 'x':x1, 'y':y1}
    l3={"k":np.tan(angles[1][0]*np.pi/180), 'x':x4, 'y':y4}
    l4={"k":np.tan(angles[1][1]*np.pi/180), 'x':x4, 'y':y4}
    l1['b']=l1['y']-l1['k']*l1['x']
    l2['b']=l2["y"]-l2['k']*l2['x']
    l3['b']=l3["y"]-l3['k']*l3['x']
    l4['b']=l4["y"]-l4['k']*l4['x']

    return l1, l2, l3, l4


def makePoint (angles, img, line=False):

    h, w=img.shape
    x1, y1=0, 0
    x4, y4=w-1, h-1

    while True:
        count=0
        l1, l2, l3, l4 = makeLine(angles, x1,  y1, x4, y4)
        x2=(l1['b']-l4['b'])/(l4['k']-l1['k'])
        if x2>w-1:
            x4-=x2+1-w
            count+=1
            continue
        y2=l1['k']*x2+l1['b']
        if y2<0:
            y1-=y2
            count+=1
            continue
        x3=(l2['b']-l3['b'])/(l3['k']-l2['k'])
        if x3<0:
            x1-=x3
            count+=1
            continue
        y3=l2['k']*x3+l2['b']
        if y3>h-1:
            y4-=y3+1-h
            count+=1
            continue
        if count==0:
            break
    
    return (x1, y1), (x2, y2), (x3, y3), (x4, y4)

def makePoints (angles, pos=[(50, 50), (50, 50)], line=False):

    p1=pos[0]
    p2=pos[1]
    x1=p1[0]
    x2=p2[0]
    y1=p1[1]
    y2=p2[1]

    l1={"k":np.tan(angles[0][0]*np.pi/180), 'x':x1, 'y':y1}
    l2={"k":np.tan(angles[0][1]*np.pi/180), 'x':x1, 'y':y1}
    l3={"k":np.tan(angles[1][0]*np.pi/180), 'x':x2, 'y':y2}
    l4={"k":np.tan(angles[1][1]*np.pi/180), 'x':x2, 'y':y2}
    l1['b']=l1['y']-l1['k']*l1['x']
    l2['b']=l2["y"]-l2['k']*l2['x']
    l3['b']=l3["y"]-l3['k']*l3['x']
    l4['b']=l4["y"]-l4['k']*l4['x']

    x3=(l1['b']-l4['b'])/(l4['k']-l1['k'])
    x4=(l2['b']-l3['b'])/(l3['k']-l2['k'])
    y3=l1['k']*x3+l1['b']
    y4=l2['k']*x4+l2['b']

    if line:
        return [l1, l2, l3, l4]

    return [(x1, y1), (x3, y3), (x4, y4), (x2, y2)]

def makeTrans (points, img, neg=-1):
    a, b, c, d=points.astype(np.float32)
    l1=np.sqrt(np.sum((a-b)**2))
    l2=np.sqrt(np.sum((a-c)**2))
    l3=np.sqrt(np.sum((b-d)**2))
    l4=np.sqrt(np.sum((c-d)**2))

    if neg*l1>neg*l4:
        w=l4
    else:
        w=l1
    
    if neg*l2>neg*l3:
        h=l3
    else:
        h=l2
    
    pt=np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # print (points, type(points))
    # print(pt, type(pt))

    M=cv.getPerspectiveTransform(points.astype(np.float32), pt)
    dst=cv.warpPerspective(img, M, np.int32([w, h]))

    return dst




