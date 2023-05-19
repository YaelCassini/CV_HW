
import cv2
import numpy as np


def grey_scale(image):
    img_gray = image

    rows, cols = img_gray.shape
    flat_gray = img_gray.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    print('A = %d,B = %d' % (A, B))
    output = np.uint8(255 / (B - A) * (img_gray - A) + 0.5)
    return output


for i in range(7, 8):
    inpath="./att_faces/s45/masked_"+str(i)+".pgm"
    src = cv2.imread(inpath, 1)
    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    result1 = cv2.equalizeHist(src_gray)

    result2 = grey_scale(result1)
    cv2.imshow('src', cv2.cvtColor(src, cv2.COLOR_BGR2GRAY))
    cv2.imshow('result1', result1)
    cv2.imshow("result2", result2)

    outpath = "./att_faces/s45/" + str(i) + ".pgm"
    cv2.imwrite(outpath,result2)

cv2.waitKey()
cv2.destroyAllWindows()