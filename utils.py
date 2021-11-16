import random
import time
from typing import Optional

import cv2.cv2 as cv
import numpy as np
import pyautogui as pag
import torch
import torchvision

import config

class BoundingBox:
    """ Bounding box of detected objects in image
    (x1,y1) = upper left corner
    (x2,y2) = lower right corner """

    x1: int
    y1: int
    x2: int
    y2: int

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def toTuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def centerX(self) -> int:
        return int((self.x1 + self.x2) / 2)

    def centerY(self) -> int:
        return int((self.y1 + self.y2) / 2)

class Basis:
    """ Similar to the linear algebra basis in the sense that these 3 values define the board.
    Can this really be called a basis? Couldn't find a better name. """

    x0: int  # screen coord of the center of the topleft plot
    y0: int  # screen coord of the center of the topleft plot
    plotsize: int  # the height/width of a plot

    def __init__(self, tlx, tly, ps):
        self.x0 = tlx
        self.y0 = tly
        self.plotsize = ps

def getBoardState(screen: np.ndarray, basis: Basis) -> np.ndarray:
    """ Get board state as array from screenshot.
    Input: screenshot of the game board and a basis
    Output: numpy array representing the game board """

    # screen coordinates and codes of all items matched
    all_matched_items: list[tuple[BoundingBox, int]] = []  # list of (BoundingBox, item_code) tuples
    # same as above, but without item codes
    all_matched_boxes: list[BoundingBox] = []

    # for each game item, find all their bounding boxes
    for item_code, item in enumerate(config.items):
        match_boxes: list[BoundingBox] = findItemCoordinates(screen, f"assets/{item}.png")
        if match_boxes is not None:
            all_matched_boxes.extend(match_boxes)
            all_matched_items.extend([(box, item_code) for box in match_boxes])

    # create board representation
    board = np.ones(shape = (6, 6), dtype = np.int8) * -1
    for box, item_code in all_matched_items:
        board_x, board_y = screenToBoardCoords(basis.x0, basis.y0, box.centerX(), box.centerY(), basis.plotsize)
        board[board_y, board_x] = item_code

    return board

def findBoardContour(screen: np.ndarray) -> Basis:
    """ Get a basis to create mapping between screen and board coordinates
    Input: screenshot of the game
    Output: (x,y) coordinates of center of topleft plot, and plot size """

    # first find the bounding boxes of all known items
    boxes: list[BoundingBox] = []
    for item_code, item in enumerate(config.items):
        match_boxes: list[BoundingBox] = findItemCoordinates(screen, f"assets/{item}.png")
        if match_boxes is not None:
            boxes.extend(match_boxes)

    # then find the indexes of the outermost items
    x1_list, y1_list, x2_list = [], [], []
    for box in boxes:
        x1_list.append(box.x1)
        y1_list.append(box.y1)
        x2_list.append(box.x2)

    arg_xmin = np.argmin(x1_list)
    arg_ymin = np.argmin(y1_list)
    arg_xmax = np.argmax(x2_list)

    # finally find the center of the top-left corner and plot size
    plotsize = int((boxes[arg_xmax].centerX() - boxes[arg_xmin].centerX()) / 5)  # calculate using center-to-center distance for better accuracy
    topleftX = boxes[arg_xmin].centerX()
    topleftY = boxes[arg_ymin].centerY()

    return Basis(topleftX, topleftY, plotsize)

def screenToBoardCoords(x0, y0, xx, yy, ps) -> tuple[int, int]:
    """ Inputs:
    x0, y0: screen coords of center of topleft plot
    xx, yy: screen coords of center of current plot
    ps: plot size
    Output: (x,y) board coords of current plot """

    board_x: int = round((xx - x0) / ps)
    board_y: int = round((yy - y0) / ps)

    return board_x, board_y

def moveItem(board_x: int, board_y: int, direction: str, basis: Basis) -> None:
    """ Input: item board coordinates
    Output: None """

    board_move_vector: Optional[np.ndarray] = None

    if direction == 'left':
        board_move_vector = np.array((-1, 0))
    elif direction == 'right':
        board_move_vector = np.array((1, 0))
    elif direction == 'up':
        board_move_vector = np.array((0, -1))
    elif direction == 'down':
        board_move_vector = np.array((0, 1))

    screen_move_vector = board_move_vector * basis.plotsize
    screen_src_coord = (basis.x0 + basis.plotsize * board_x, basis.y0 + basis.plotsize * board_y)
    screen_dst_coord = screen_src_coord + screen_move_vector

    pag.moveTo(*screen_src_coord)
    time.sleep(0.2)
    pag.dragTo(*screen_dst_coord, 0.5, pag.easeInOutQuad)

def waitUntilImageLocated(img_path: str, threshold: float = 0.95, use_mask: bool = False) -> BoundingBox:
    screen = cv.cvtColor(np.array(pag.screenshot()), cv.COLOR_RGB2BGR)
    tries = 0
    detection: Optional[list[BoundingBox]]

    while not (detection := findItemCoordinates(screen, img_path, threshold = threshold, use_mask = use_mask, deduplicate = False)):
        print(f"{img_path} not found {tries}")
        time.sleep(0.2)
        tries += 1
        screen = cv.cvtColor(np.array(pag.screenshot()), cv.COLOR_RGB2BGR)

    centerCoord: BoundingBox = detection[0]

    return centerCoord

def waitUntilScreenStable(interval: float = 0.1):
    while True:
        scr1 = pag.screenshot()
        time.sleep(interval)
        scr2 = pag.screenshot()
        if scr1 == scr2:
            break

def screenshotBGR() -> np.ndarray:
    screenRGB = np.array(pag.screenshot())
    screenBGR = cv.cvtColor(screenRGB, cv.COLOR_RGB2BGR)
    return screenBGR

def imageIsPresent(screen: np.ndarray, template_file: str, threshold: float = 0.95, use_mask: bool = False):
    """ Check if an image is present on the screen """
    return bool(findItemCoordinates(screen, template_file, threshold = threshold, use_mask = use_mask, deduplicate = False))

def strategize(board: np.ndarray) -> tuple[int, int, str]:
    """ Derive the next move from the current gameboard state
    Input: gameboard
    Output: (y,x) of item to move and the direction (up,down,left,right) """

    # strategy: accumulate the same items at the bottom of the
    # leftmost column / rightmost column / bottom row
    # and throw other items into the sea :D

    if (board[:, 0] != -1).all() and (board[:, 0] != 5).all():
        item_code = board[5, 0]
        for i in [4, 3, 2, 1, 0]:
            if board[i, 0] != item_code:
                return i, 0, 'left'

    elif (board[:, 5] != -1).all() and (board[:, 5] != 5).all():
        item_code = board[5, 5]
        for i in [4, 3, 2, 1, 0]:
            if board[i, 5] != item_code:
                return i, 5, 'right'

    elif (board[5, :] != -1).all() and (board[5, :] != 5).all():
        item_code = board[5, 0]
        for i in [1, 2, 3, 4, 5]:
            if board[5, i] != item_code:
                return 5, i, 'down'

    else:
        return random.randint(0, 5), random.randint(1, 5), 'up'

def findItemCoordinates(img: np.ndarray, template_file: str, method: int = cv.TM_CCORR_NORMED, threshold: float = 0.95, use_mask: bool = True, deduplicate: bool = True) -> list[BoundingBox]:
    """ Find occurrences of an item in a game screenshot, deduplication included.
    Input: image (screenshot), template file path
    Output: list of item bounding boxes [BoundingBox(x1,y1,x2,y2), ...] where x1 < x2 and y1 < y2 """

    # use mask to apply BGRA templates, don't use mask for BGR templates
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

    # find coordinates [[y1, y2, ...], [x1, x2, ...]] of matching locations in img
    loc = np.where(res >= threshold)
    # if nothing matched, return None
    if len(loc[0]) == 0:
        return []

    # get the bounding boxes of each match
    # boxes are defined by their top-left corner (x1,y1) and bottom-right corner (x2,y2), where x1 < x2 and y1 < y2
    h, w, _ = template.shape
    x1 = loc[1]
    y1 = loc[0]
    x2 = x1 + w
    y2 = y1 + h

    if deduplicate:
        # there may be duplicate matches for each item, so we must perform Non-Maximum Suppression (NMS) to eliminate duplicates
        # we shall shamelessly let TorchVision do the dirty work for us ;)
        # here's an article that explains the concept: https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
        boxes = torch.tensor(list(zip(x1, y1, x2, y2)), dtype = torch.float32)
        scores = torch.tensor(res[loc], dtype = torch.float32)
        iou_threshold = 0.5
        # get indexes of matched templates after filtering out duplicates
        filt_idxs = torchvision.ops.nms(boxes = boxes, scores = scores, iou_threshold = iou_threshold)

        return list(map(lambda t: BoundingBox(*(t.int().tolist())), boxes[filt_idxs]))
    else:
        return list(map(lambda c: BoundingBox(*c), list(zip(x1, y1, x2, y2))))
