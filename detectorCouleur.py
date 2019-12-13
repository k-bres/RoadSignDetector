import sys
import os
import cv2
import numpy as np
import imutils

def main(argv):
    # Creation des dossiers si ils n'existent pas
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("inputs"):
        os.makedirs("inputs")

    # On applique le traitement sur tous les fichiers du dossier 'inputs'
    for root, dirs, files in os.walk("inputs"):
        for filename in files:
            process("inputs/" + filename)

def process(file):
    # Import de l'image
    imgoriginale = cv2.imread(file, cv2.IMREAD_COLOR)
    # cv2.imshow("Originale", imgoriginale)

    # On resize si grande image
    height, width = imgoriginale.shape[:2]
    max_height = 500
    max_width = 500
    isresized = False

    if max_height < height or max_width < width:
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        resized = cv2.resize(imgoriginale, None, fx = scaling_factor, fy = scaling_factor, interpolation = cv2.INTER_AREA)
        isresized = True

    # Augmentation du contraste
    if not isresized:
        resized = imgoriginale
    
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit = 3., tileGridSize = (8, 8))
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    imgprocess = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # On floute avec du gaussian puis du median blur
    blur = cv2.GaussianBlur(imgprocess, (9, 9), 0)
    blur = cv2.medianBlur(blur, 9)

    # On transforme l'image en image HSV
    img_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Maque rouge valeurs basses
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # Masque rouge valeurs hautes
    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # Jointure des masques
    mask = mask0 + mask1

    # On enleve tout ce qui n'est pas rouge
    imgrouge = img_hsv.copy()
    imgrouge[np.where(mask == 0)] = 0
    imgrouge[np.where(imgrouge != 0)] = 255

    # On converti en niveau de gris
    gray = cv2.cvtColor(imgrouge, cv2.COLOR_BGR2GRAY)

    # On converti en noir et blanc
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # On recupere les contours, on les trie par aire maximum et on garde les 10 plus grands
    cnts = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

    # On ne garde que les contours ayant un nombre de contours coherent
    cnt_list = []
    for cnt in cnts:
        approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
        if ((len(approx) > 3) & (len(approx) < 29)):
            cnt_list.append(cnt)

    # On calcule le rectangle autour du panneau puis on crop
    tmp = []
    if len(cnt_list) > 0:
        tmp = cnt_list[0]
    else:
        tmp = cnts[0]

    x, y, w, h = cv2.boundingRect(tmp)
    crop = resized[y:y + h, x:x + w]

    # Sauvegarde du resultat
    cv2.imwrite("results/detected-" + file[7:], crop)

    # On affiche le resultat
    # cv2.imshow("Detecte", crop)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])
