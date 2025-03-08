import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((500,500,3), np.uint8)

cv.line(img,(0,0),(500,500),(0,0,255),5)

cv.rectangle(img,(100,200),(400,300),(0,255,0),5)

cv.circle(img,(250,250), 50, (255,0,0), -1)

font = cv.FONT_HERSHEY_PLAIN
img = cv.putText(img,'Albert',(100,199), font, 6, (255,255,255), 2, cv.LINE_AA)

plt.imsave("original.png", img)

#convertir a escalas de grises
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgGray

cv.imwrite("gray_scale.png", imgGray)

imgResize = cv.resize(img, (0,0), fx=2, fy=2)
imgResize

plt.imsave("resize.png", imgResize)

#rotar imagen a 45 grados
RotateMatrix = cv.getRotationMatrix2D((250,250), 45, 1)
imgRotate = cv.warpAffine(img, RotateMatrix, dsize=None)
imgRotate

plt.imsave("rotate.png", imgRotate)

rotatedmatrix30 = cv.getRotationMatrix2D((250,250), 30, 1)
imgRotate30 = cv.warpAffine(img, rotatedmatrix30, (500,500))
imgRotate30

height, width = img.shape[:2]
pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
pts2 = np.float32([[50, 50], [width-50, 0], [0, height], [width, height-50]])

matrix = cv.getPerspectiveTransform(pts1, pts2)
imgOutput = cv.warpPerspective(img, matrix, (width, height))
imgOutput

plt.imsave("perspective.png", imgOutput)

imgMask = img.copy()
imgMask = cv.cvtColor(imgMask, cv.COLOR_BGR2BGRA)
mask = np.zeros((500, 500), dtype=np.uint8)
cv.circle(mask, (250, 250), 150, 255, -1)

imgMask[:, :, 3] = mask

plt.figure(figsize=(6, 6))
plt.imshow(imgMask)
plt.axis("off")
plt.show()

plt.imsave("imagen_transparente.png", imgMask)
