import numpy as np
import cv2 
import os
import imutils

def main():
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('inputs'):
        os.makedirs('inputs')
    process()

def process():
    
    for file in os.listdir("inputs"):
        imageName = 'inputs/'+ file
        raw_image = cv2.imread(imageName)
        process_image = raw_image.copy()
        scale_percent = 60 # percent of original size
        width = int(process_image.shape[1] * scale_percent / 100)
        height = int(process_image.shape[0] * scale_percent / 100)
        dim = (width, height)


        gray = cv2.cvtColor(process_image, cv2.COLOR_BGR2GRAY)
        bilateral_filtered_image = cv2.GaussianBlur(gray,(3,3),0)
        #cv2.imshow('Bilateral', bilateral_filtered_image)
        # cv2.waitKey(0)

        edge_detected_image = cv2.Canny(bilateral_filtered_image, 90, 170)
        #cv2.imshow('Edge', edge_detected_image)
        # cv2.waitKey(0)

        contours = cv2.findContours(edge_detected_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
        contour_list = []
        maxLength = 0
        for contour in contours:
            length = cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,0.01*length, True)
            area = cv2.contourArea(contour)
            if ((len(approx) > 4) & (len(approx) < 20) ):
                if (area > maxLength):
                    maxLength = area
                    del contour_list[:]
                    contour_list.append(contour)
                elif (area == maxLength):
                    contour_list.append(contour)
        
        (x,y,w,h) = cv2.boundingRect(contour_list[0])
        #cv2.rectangle(raw_image, (x,y), (x+w,y+h), (0,255,0), 2)
        #cv2.drawContours(raw_image, contour_list,  -1, (255,0,0), 2)
        raw_image = raw_image[y:y + h, x:x + w]
        cv2.imshow('Nombre de contour -> '+str(len(contour_list)),raw_image)
       
        cv2.imwrite("results/"+file, raw_image)
        print("Saving ...")
        cv2.waitKey(0)

    


if __name__ == "__main__":
	main()