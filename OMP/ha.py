import cv2

img = cv2.imread("lena.jpg")
cv2.imwrite("lena256_B.jpg",img[128:384,128:384,0])
