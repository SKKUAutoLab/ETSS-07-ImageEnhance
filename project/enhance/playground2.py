import cv2
import numpy as np

# reading the damaged image
damaged_img = cv2.imread(filename=r"data/10.png")
# get the shape of the image
height, width = damaged_img.shape[0], damaged_img.shape[1]
# Converting all pixels greater than zero to black while black becomes white
thres = 5
for i in range(height):
	for j in range(width):
		if damaged_img[i, j, 0] > thres:
			damaged_img[i, j] = 0
		elif damaged_img[i, j, 1] > thres:
			damaged_img[i, j] = 0
		elif damaged_img[i, j, 2] > thres:
			damaged_img[i, j] = 0
		else:
			damaged_img[i, j] = [255, 255, 255]
# saving the mask
mask = damaged_img
kernel = np.ones((5, 5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)
cv2.imwrite('mask.jpg', mask)

# Open the image.
img = cv2.imread(r"data/10.png")
# Load the mask.
mask = cv2.imread('mask.jpg', 0)
# Inpaint.
dst  = cv2.inpaint(img, mask, 9, cv2.INPAINT_NS)
# Write the output.
cv2.imwrite('cat_inpainted.png', dst)

# displaying mask
cv2.imshow("damaged image mask", mask)
cv2.imshow('cat_inpainted.png', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
