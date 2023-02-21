import cv2 as cv
import PIL.Image as pil

import numpy as np

from monodepth import Monodepth

model = Monodepth("mono_640x192")

cv.namedWindow("Depth", cv.WINDOW_NORMAL)
cv.resizeWindow("Depth", 640, 192)

cap = cv.VideoCapture(0)
show_monodepth = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (640, 192))
    frame = cv.flip(frame, 1)
    frame = pil.fromarray(frame)
    
    if show_monodepth:
        frame, disp = model.parse_frame(frame)

    frame = np.array(frame)
    cv.imshow("Depth", frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("m"):
        show_monodepth = not show_monodepth