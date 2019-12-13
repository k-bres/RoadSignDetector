from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
import numpy as np
import cv2 
import math
import os
import argparse
import imutils

def main():
    compare()

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

def compute_psnr(img1, img2):
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    m = mse(img1, img2)
    if m == 0:
        return "Same Image"
    return 10 * math.log10(1. / m)

def compare():
    
    for res in sorted(os.listdir("results")):
        bestMSE = 0
        bestSSIM = -1
        bestPSNR = -55
        imageResult = cv2.imread('results/'+ res)
        imageResult = cv2.resize(imageResult, (100, 100))

        lab = cv2.cvtColor(imageResult, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit = 3., tileGridSize = (8, 8))
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        imageResult = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        for file in os.listdir("test"):
            imageTest = cv2.imread('test/'+ file)
            imageTest = cv2.resize(imageTest, (100, 100))
            s = ssim(imageResult, imageTest, multichannel=True)
            m = mse(imageResult, imageTest)
            p = compute_psnr(imageResult, imageTest)
            if m > bestMSE:
                bestMSE = m
                nomMSE = file
                imageBestMSE = imageTest.copy()
            if s > bestSSIM:
                bestSSIM = s
                nomSSIM = file
                imageBestSSIM = imageTest.copy()
            if p > bestPSNR:
                bestPSNR = p
                nomPSNR = file
                imageBestPSNR = imageTest.copy()
)
        print(res)
        print(nomMSE +"-------->"+ str(bestMSE))
        print(nomSSIM +"-------->"+ str(bestSSIM))
        print(nomPSNR +"-------->"+ str(bestPSNR))
        numpy_horizontal = np.hstack((imageResult, imageBestSSIM, imageBestPSNR, imageBestMSE))
 
        cv2.imshow('Original: '+ res +' --> SSIM --> PSNR --> MSE', numpy_horizontal)

        k = 0
        while k != 27:
            k = cv2.waitKey(0)
        cv2.destroyAllWindows();
        


if __name__ == "__main__":
	main()