import cv2 as cv
import matplotlib.pyplot as plt

def make_name (path):
    ind=path[::-1].index(".")
    try:
        ins=path[::-1].index("/")
        root=path[-ins:-ind]
        dirc=path[:-ins]
    except:
        dirc=""
        root=path[:-ind]

    return dirc, root


def make_pgm (path):

    dirc, root=make_name(path)

    img=cv.imread(path)
    img=cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    cv.imwrite(dirc+root+"pgm", img)


def make_yaml (path):

    dirc, root=make_name(path)

    with open(dirc+root+"yaml", "w") as yaml:
        yaml.write("image: " + "./" + root+"pgm")
        yaml.write("\nresolution: "+str(0.050000))
        origin=[-1.000000, -1.000000, 0.000000]
        yaml.write("\norigin: "+str(origin))
        yaml.write("\nnegate: "+str(0))
        yaml.write("\noccupied_thresh: " + str(0.65))
        yaml.write("\nfree_thresh: "+str(0.196))
