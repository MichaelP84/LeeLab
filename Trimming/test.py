import os
import cv2
import torch as pt
import numpy as np




path = './RawVideo/Vid1.avi'

cap = cv2.VideoCapture(path)
cap.set(cv2.CAP_PROP_POS_MSEC, 10.9*60000)
max = 0
min = 100

while cap.isOpened():
    ret, frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rows, cols = grayFrame.shape
    rows = int(rows/2)
    cols = int(cols/2)
    total = 0
    k = 0

    cv2.imshow('frame', grayFrame)
    cv2.waitKey(100)

    for i in range(rows):
        for j in range(cols):
            k += grayFrame.item(i, j)
            total += 1

    total = k / total
    print(total)

    if total < min:
        min = total
    if total > max:
        max = total


print(min, max)


cap.release()
cv2.destroyAllWindows()

# cap = cv2.VideoCapture(path)
 
# while(True):
#     ret, frame = cap.read()S
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()