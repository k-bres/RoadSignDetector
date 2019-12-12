from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
import numpy as np
import cv2 
import os
import argparse
import imutils

def main():
    compare()



def compare():
    best = 0
    for res in os.listdir("results"):
        imageResult = cv2.imread('results/'+ res)
        height = imageResult.shape[0]
        width = imageResult.shape[1]
        cv2.imshow('result image ',imageResult)
        
        for file in os.listdir("test"):
            imageTest = cv2.imread('test/'+ file)
            imageTest = cv2.resize(imageTest, (width, height))
            s = ssim(imageResult, imageTest, multichannel=True)
            if s > best:
                best = s
                nom = file
        imageTest = cv2.imread('test/'+ nom)
        cv2.imshow('result image ',imageResult)
        print(res)
        print(nom +"-------->"+ str(best))
        cv2.waitKey(0)
        
    


if __name__ == "__main__":
	main()