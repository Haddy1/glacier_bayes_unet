import cv2

def bilateral(img, diameter=20, sigmaColor=80, sigmaSpace=80):
    return cv2.bilateralFilter(img, diameter, sigmaColor,sigmaSpace)
