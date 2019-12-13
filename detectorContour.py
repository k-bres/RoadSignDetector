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

        edge_detected_image = cv2.Canny(bilateral_filtered_image, 90, 170)
        
        contours = cv2.findContours(edge_detected_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
        contour_list = []
        maxLength = 0
        for contour in contours:
            length = cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,0.01*length, True)
            area = cv2.contourArea(contour)
            if ((len(approx) > 3) & (len(approx) < 20) ):
                if (area > maxLength):
                    maxLength = area
                    del contour_list[:]
                    contour_list.append(contour)
                elif (area == maxLength):
                    contour_list.append(contour)
        
        (x,y,w,h) = cv2.boundingRect(contour_list[0])
        #cv2.rectangle(raw_image, (x,y), (x+w,y+h), (0,255,0), 2)
        # cv2.drawContours(raw_image, contour_list,  -1, (255,0,0), 2)
        # cv2.waitKey(0)

        # ------ REMPLIR DE FOND DE L'IMAGE EN NOIR (ABANDONER) -----------
        #https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour
        # img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        # idx = 0 # The index of the contour that surrounds your object
        # mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
        # cv2.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
        # out = np.zeros_like(raw_image) # Extract out the object and place into output image
        # out[mask == 255] = raw_image[mask == 255]

        # # Now crop
        # (y, x) = np.where(mask == 255)
        # (topy, topx) = (np.min(y), np.min(x))
        # (bottomy, bottomx) = (np.max(y), np.max(x))
        # out = out[topy:bottomy+1, topx:bottomx+1]
        # cv2.imshow('Output', out)
        # ------------------------------------------------------
        raw_image = raw_image[y:y + h, x:x + w]
        #cv2.drawContours(raw_image, contour_list,  -1, (255,0,0), 2)
        cv2.imshow(file,raw_image)

       
        cv2.imwrite("results/"+file, raw_image)
        print("Saving ...")
        k = 0
        while k != 27:
            k = cv2.waitKey(0)
        cv2.destroyAllWindows();

    


if __name__ == "__main__":
	main()

