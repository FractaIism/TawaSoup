import cv2.cv2 as cv
import numpy as np
import pyautogui as pag

while True:
    scrRGB = np.array(pag.screenshot())
    scrBGR = cv.cvtColor(scrRGB, cv.COLOR_RGB2BGR)
    cv.imshow("Screen", scrBGR)
    if cv.waitKey(200) == ord('q'):
        break
