import cv2.cv2 as cv
import numpy as np

img = cv.imread("../assets/game3.png", cv.IMREAD_COLOR)  # color mode: BGR
method = cv.TM_CCORR_NORMED  # cv.TM_CCORR_NORMED works best
threshold = 0.95  # higher = stricter
items = ['cannonball', 'gold', 'ice', 'rock', 'wood', 'chest']  # template asset names

for index, item in enumerate(items):
    # a copy of img to draw rectangles on
    img2 = img.copy()
    # image to search for in game screen
    template = cv.imread(f"../assets/{item}.png", cv.IMREAD_UNCHANGED)  # color mode: BGRA
    # width,height,channels of the template
    h, w, c = template.shape
    # isolate template's alpha channel to create mask
    # this mask will allow us to ignore the grass background when template matching
    alpha = template[:, :, 3]
    template_mask = cv.merge([alpha, alpha, alpha])
    # remove alpha channel to make the dimensions same as img
    template = cv.cvtColor(template, cv.COLOR_BGRA2BGR)

    # create a map of template match score, larger value = whiter = better match (with method cv.TM_CCORR_NORMED)
    res = cv.matchTemplate(img, template, method, mask = template_mask)
    # find array indexes of matching locations in img
    loc = np.where(res >= threshold)
    # draw rectangles around locations that match the template
    for pt in zip(*loc[::-1]):
        cv.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    # make img2 smaller to fit on screen (uncomment and imshow res_small to view the template matching map)
    img2_small = cv.resize(img2, None, fx = 0.5, fy = 0.5)
    # res_small = cv.resize(res, None, fx = 0.5, fy = 0.5)
    cv.imshow("img", img2_small)
    cv.setWindowTitle("img", item)  # to reuse the same window
    cv.waitKey(0)  # required to update window canvas

cv.destroyAllWindows()
