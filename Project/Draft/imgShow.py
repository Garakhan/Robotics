import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def imgShow (img, title="An image", cmap="gray", figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow (img, cmap=cmap)
    plt.show()
