import os
import random
from unittest import TestCase

import cv2.cv2 as cv
import numpy as np
import pyautogui as pag

import config
import utils
from config import Basis

def initGame() -> None:
    """ Open game and click click """

    os.system("microsoftedge https://towerswap.app/play")
    btn_box_1 = utils.waitUntilImageLocated("assets/play_now.png")
    pag.click(x = btn_box_1.centerX(), y = btn_box_1.centerY())
    btn_box_2 = utils.waitUntilImageLocated("assets/play.png")
    pag.click(x = btn_box_2.centerX(), y = btn_box_2.centerY())
    utils.waitUntilScreenStable()

def getTemplateMatchingScore(img: np.ndarray, template_file: str, method = cv.TM_CCORR_NORMED, use_mask: bool = False) -> float:
    """ Get max matching score for template matching """

    if use_mask:
        # image to search for in game screen
        template = cv.imread(template_file, cv.IMREAD_UNCHANGED)  # color mode: BGRA
        # isolate template's alpha channel to create mask
        # this mask will allow us to ignore the grass background when template matching
        alpha = template[:, :, 3]
        template_mask = cv.merge([alpha, alpha, alpha])
        # remove alpha channel to make the dimensions same as img
        template = cv.cvtColor(template, cv.COLOR_BGRA2BGR)
        # create a map of template match score, larger value = whiter = better match (with method cv.TM_CCORR_NORMED)
        res = cv.matchTemplate(img, template, method, mask = template_mask)
    else:
        template = cv.imread(template_file, cv.IMREAD_COLOR)
        res = cv.matchTemplate(img, template, method)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return max_val

class Test(TestCase):
    def setUp(self) -> None:
        os.chdir("..")

    def test_find_board_contour(self):
        """ Draw rectangle through center of corner items, and outside of corner items, and diagonal from (x0,y0)
        (requires manual checking) """

        screen = cv.imread("assets/game3.png")
        basis: Basis = utils.findBoardContour(screen)
        x0 = basis.x0
        y0 = basis.y0
        ps = basis.plotsize

        # draw rectangle through center of corner plots
        topleftCenter = (x0, y0)
        bottomrightCenter = (x0 + 5 * ps, y0 + 5 * ps)
        cv.rectangle(screen, topleftCenter, bottomrightCenter, (255, 0, 0), 2)
        # draw rectangle containing all plots
        topleftCorner = (int(x0 - 0.5 * ps), int(y0 - 0.5 * ps))
        bottomrightCorner = (int(x0 + 5.5 * ps), int(y0 + 5.5 * ps))
        cv.rectangle(screen, topleftCorner, bottomrightCorner, (0, 0, 255), 2)
        cv.line(screen, (x0, y0), (x0 + ps, y0 + ps), (0, 255, 0), 5)

        img_small = cv.resize(screen, None, fx = 0.5, fy = 0.5)
        cv.imshow("Game", img_small)
        cv.waitKey(0)

    def test_find_board_contour_2(self):
        """ Do the same thing but with a screenshot
        (opens game window, requires manual checking) """

        initGame()
        screen = cv.cvtColor(np.array(pag.screenshot()), cv.COLOR_RGB2BGR)
        basis: Basis = utils.findBoardContour(screen)
        x0 = basis.x0
        y0 = basis.y0
        ps = basis.plotsize

        # draw rectangle through center of corner plots
        topleftCenter = (x0, y0)
        bottomrightCenter = (x0 + 5 * ps, y0 + 5 * ps)
        cv.rectangle(screen, topleftCenter, bottomrightCenter, (255, 0, 0), 2)
        # draw rectangle containing all plots
        topleftCorner = (int(x0 - 0.5 * ps), int(y0 - 0.5 * ps))
        bottomrightCorner = (int(x0 + 5.5 * ps), int(y0 + 5.5 * ps))
        cv.rectangle(screen, topleftCorner, bottomrightCorner, (0, 0, 255), 2)
        cv.line(screen, (x0, y0), (x0 + ps, y0 + ps), (0, 255, 0), 5)

        img_small = cv.resize(screen, None, fx = 0.5, fy = 0.5)
        cv.imshow("Game", img_small)
        cv.waitKey(0)

    def test_get_board_state(self):
        """ Get board state from game3.png """

        screen = cv.imread("assets/game3.png")
        basis = utils.findBoardContour(screen)
        board = utils.getBoardState(screen)

        print(board, end = "\n\n")
        for idx, item in enumerate(config.items):
            print(idx, item)

        np.testing.assert_array_equal(board, np.array([[0, 2, 1, 4, 3, 2], [2, 4, 3, 0, 1, 3], [3, 0, 2, 0, -1, 1], [3, -1, 1, -1, 3, 2], [2, 3, 1, -1, 0, -1], [1, 2, 2, -1, 2, -1]]))

    def test_move_item(self):
        """ Move some arbitrary items around
        (opens game window, requires manual checking) """

        initGame()
        config.basis = utils.findBoardContour(utils.screenshotBGR())

        utils.moveItem(1, 0, 'left')
        utils.moveItem(1, 2, 'right')
        utils.moveItem(5, 5, 'up')
        utils.moveItem(3, 3, 'down')

    def test_detect_night(self):
        """ Check if waitUntilImageLocated can detect night time
        (opens game window, requires manual checking) """

        initGame()
        screenRGB = np.array(pag.screenshot())
        screenBGR = cv.cvtColor(screenRGB, cv.COLOR_RGB2BGR)
        basis = utils.findBoardContour(screenBGR)

        for i in range(10):
            x = random.randint(0, 5)
            y = random.randint(1, 5)
            utils.moveItem(x, y, 'down')

        det = utils.waitUntilImageLocated("assets/zero_swaps.png")
        print("Night detected")
        print(det)

    def test_image_is_present(self):
        """ Check that zero_swaps.png and swaps.png can be detected properly """
        screen = cv.imread("assets/game3.png")
        zeroswaps = "assets/zero_swaps.png"
        swaps = "assets/swaps.png"
        self.assertFalse(utils.imageIsPresent(screen, zeroswaps))
        self.assertTrue(utils.imageIsPresent(screen, swaps))

    def test_find_item_bounding_boxes(self):
        """ Identify each type of item in game3.png
        (requires manual checking) """

        img = cv.imread("assets/game3.png", cv.IMREAD_COLOR)  # color mode: BGR
        items = ['cannonball', 'gold', 'ice', 'rock', 'wood', 'chest']  # template asset names
        quantities = [5, 6, 9, 7, 2, 2]

        for idx, item in enumerate(items):
            img2 = img.copy()
            match_boxes = utils.findItemBoundingBoxes(img, f"assets/{item}.png")

            # draw bounding boxes around locations that match the template
            for box in match_boxes:
                cv.rectangle(img2, (box.x1, box.y1), (box.x2, box.y2), (0, 0, 255), 2)

            # make img2 smaller to fit on screen
            img2_small = cv.resize(img2, None, fx = 0.5, fy = 0.5)
            cv.imshow("img", img2_small)
            cv.setWindowTitle("img", f"{item} = {len(match_boxes)}")  # to reuse the same window
            cv.waitKey(0)  # required to update window canvas

            # make sure the number of matches is correct (none missing or duplicate)
            self.assertEqual(len(match_boxes), quantities[idx])

    def test_find_item_bounding_boxes_2(self):
        """ Check that findItemCoordinates() returns [] when a template is not matched """

        screen = cv.imread("assets/game3.png")
        match_boxes = utils.findItemBoundingBoxes(screen, "assets/watch_ad.png", use_mask = False)
        self.assertListEqual(match_boxes, [])

    def test_blocking(self):
        """ Find the min threshold required to detect each and every item
        Result: 0.93 """

        least_threshold = 100.0

        for item in ['cannonball', 'chest', 'gold', 'ice', 'rock', 'wood']:
            img = cv.imread(f"assets/blocked/blocked_{item}.png")
            template_file = f"assets/{item}.png"
            max_val = getTemplateMatchingScore(img, template_file, use_mask = True)
            print(item, max_val)
            if max_val < least_threshold:
                least_threshold = max_val

        print("least threshold ", least_threshold)

    def test_detect_king(self):
        """ Check king's template matching score """
        screen = cv.imread("assets/testking.png")
        template_file = "assets/king.png"
        score = getTemplateMatchingScore(screen, template_file, use_mask = True)
        print(score)

    def test_find_region_of_interest(self):
        """ Check ROI correctness """
        screen = cv.imread("assets/game3.png")
        config.basis = utils.findBoardContour(screen)
        utils.findRegionOfInterest()
        r = config.roi
        cv.rectangle(screen, (r.x1, r.y1), (r.x2, r.y2), (0, 0, 255), 2)
        cv.imshow("roi", cv.resize(screen, None, fx = .5, fy = .5))
        cv.waitKey(0)
