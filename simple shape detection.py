import cv2
import numpy as np


def center_crop(img, new_width=None, new_height=None):

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img


img = cv2.imread("C:/Users/farsh/Documents/GitHub/Cat_Or_NotCat/Training set/image29.jpg", cv2.IMREAD_GRAYSCALE)
width = img.shape[1]
height = img.shape[0]
print(width)
print(height)
#img = center_crop(img, 500, 500)
_, threshold = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

font = cv2.FONT_HERSHEY_COMPLEX

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    cv2.drawContours(img, [approx], 0, (0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    if len(approx) == 4:
        cv2.putText(img, "Rectangle", (x, y), font, 1, (0))

cv2.imshow("shapes", img)
cv2.imshow("Threshold", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()


