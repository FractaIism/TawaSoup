from random import randint
import time
from typing import Optional

import cv2.cv2 as cv
import numpy as np
import pyautogui as pag
import torch
import torchvision

import config
from config import BoundingBox, Basis

def getBoardState(screen: np.ndarray) -> np.ndarray:
    """ Get board state as array from screenshot.
    Input: screenshot of the game board and a basis
    Output: numpy array representing the game board """

    # screen coordinates and codes of all items matched
    all_matched_items: list[tuple[BoundingBox, int]] = []  # list of (BoundingBox, item_code) tuples
    # same as above, but without item codes
    all_matched_boxes: list[BoundingBox] = []

    # for each game item, find all their bounding boxes
    for item_code, item in enumerate(config.items):
        match_boxes: list[BoundingBox] = findItemBoundingBoxes(screen, f"assets/{item}.png")
        if match_boxes is not None:
            all_matched_boxes.extend(match_boxes)
            all_matched_items.extend([(box, item_code) for box in match_boxes])

    # create board representation
    board = np.ones(shape = (6, 6), dtype = np.int8) * -1
    for box, item_code in all_matched_items:
        board_x, board_y = screenToBoardCoords(box.centerX(), box.centerY())
        board[board_y, board_x] = item_code

    return board

def findBoardContour(screen: np.ndarray) -> Basis:
    """ Get a basis to create mapping between screen and board coordinates
    Input: screenshot of the game
    Output: a basis, including coords of center of topleft plot, and plot size """

    # first find the bounding boxes of all known items
    boxes: list[BoundingBox] = []
    for item_code, item in enumerate(config.items):
        match_boxes: list[BoundingBox] = findItemBoundingBoxes(screen, f"assets/{item}.png")
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

def screenToBoardCoords(sx: int, sy: int) -> tuple[int, int]:
    """ Inputs: (x,y) screen coords of center of plot
    Output: (x,y) board coords of plot """

    B = config.basis
    board_x: int = round((sx - B.x0) / B.plotsize)
    board_y: int = round((sy - B.y0) / B.plotsize)
    return board_x, board_y

def boardToScreenCoords(bx: int, by: int) -> tuple[int, int]:
    """ Inputs:
    bx,by: board coords
    Output: (x,y) screen coords """

    B = config.basis
    screen_x = B.x0 + bx * B.plotsize
    screen_y = B.y0 + by * B.plotsize
    return screen_x, screen_y

def moveItem(board_x: int, board_y: int, direction: str) -> None:
    """ Move an item one plot left/right/up/down
    Input: item board coordinates
    Output: None """

    board_move_vector: np.ndarray = np.array((0, 0))

    if direction == 'left':
        board_move_vector = np.array((-1, 0))
    elif direction == 'right':
        board_move_vector = np.array((1, 0))
    elif direction == 'up':
        board_move_vector = np.array((0, -1))
    elif direction == 'down':
        board_move_vector = np.array((0, 1))

    screen_move_vector = board_move_vector * config.basis.plotsize
    screen_src_coord = boardToScreenCoords(board_x, board_y)
    screen_dst_coord = screen_src_coord + screen_move_vector

    if pag.position() != pag.Point(*screen_src_coord):
        pag.moveTo(*screen_src_coord, 0.5, pag.easeInOutQuad)
    pag.dragTo(*screen_dst_coord, 0.5, pag.easeInOutQuad)

def waitUntilImageLocated(img_path: str, threshold: float = 0.95, use_mask: bool = False, timeout: int = 60) -> Optional[BoundingBox]:
    screen = cv.cvtColor(np.array(pag.screenshot()), cv.COLOR_RGB2BGR)
    interval = 0.1
    detection: Optional[list[BoundingBox]]
    start = time.perf_counter()
    elapsed = 0

    while not (detection := findItemBoundingBoxes(screen, img_path, threshold = threshold, use_mask = use_mask, deduplicate = False)):
        print(f"{img_path} not found {round(elapsed, 3)}")
        time.sleep(interval)
        screen = cv.cvtColor(np.array(pag.screenshot()), cv.COLOR_RGB2BGR)
        if (elapsed := time.perf_counter() - start) > timeout:
            print("waitUntilImageLocated(): timeout")
            return None

    centerCoord: BoundingBox = detection[0]

    return centerCoord

def waitUntilImageNotLocated(img_path: str, threshold: float = 0.95, use_mask: bool = False, timeout: int = 60) -> None:
    interval = 0.1
    start = time.perf_counter()
    elapsed = 0

    while imageIsPresent(screenshotBGR(), img_path, threshold = threshold, use_mask = use_mask):
        print(f"{img_path} still present {round(elapsed, 3)}")
        time.sleep(interval)
        if (elapsed := time.perf_counter() - start) > timeout:
            print("waitUntilImageNotLocated(): timeout")
            return

def waitUntilScreenStable(interval: float = 0.1, timeout: int = 60) -> Optional[int]:
    """ Wait until the screen is no longer changing
    Return -1 if timeout, else return None """

    start = time.perf_counter()
    elapsed = 0

    while True:
        print(f"Waiting for screen stabilize {round(elapsed, 3)}")
        scr1 = pag.screenshot()
        time.sleep(interval)
        scr2 = pag.screenshot()
        if scr1 == scr2:
            break
        elif (elapsed := time.perf_counter() - start) > timeout:
            print("waitUntilScreenStable(): timeout")
            return -1

def screenshotBGR() -> np.ndarray:
    """ Take a screenshot in RGB and convert it to BGR """
    screenRGB = np.array(pag.screenshot())
    screenBGR = cv.cvtColor(screenRGB, cv.COLOR_RGB2BGR)
    return screenBGR

def imageIsPresent(screen: np.ndarray, template_file: str, threshold: float = 0.95, use_mask: bool = False) -> bool:
    """ Check if an image is present on the screen """
    return bool(findItemBoundingBoxes(screen, template_file, threshold = threshold, use_mask = use_mask, deduplicate = False))

def findItemBoundingBoxes(img: np.ndarray, template_file: str, method: int = cv.TM_CCORR_NORMED, threshold: float = 0.93, use_mask: bool = True, deduplicate: bool = True) -> list[BoundingBox]:
    """ Find occurrences of an item in a game screenshot, deduplication included.
    Input: image (screenshot), template file path
    Output: list of item bounding boxes [BoundingBox(x1,y1,x2,y2), ...] where x1 < x2 and y1 < y2 """

    # grab only region of interest to save computation time
    roi = config.roi
    subimg = img[roi.y1:roi.y2, roi.x1:roi.x2, :]

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
        res = cv.matchTemplate(subimg, template, method, mask = template_mask)
    else:
        template = cv.imread(template_file, cv.IMREAD_COLOR)
        res = cv.matchTemplate(subimg, template, method)

    # find coordinates [[y1, y2, ...], [x1, x2, ...]] of matching locations in img
    loc = np.where(res >= threshold)
    # if nothing matched, return None
    if len(loc[0]) == 0:
        return []

    # get the bounding boxes of each match
    # boxes are defined by their top-left corner (x1,y1) and bottom-right corner (x2,y2), where x1 < x2 and y1 < y2
    h, w, _ = template.shape
    x1 = loc[1] + roi.x1
    y1 = loc[0] + roi.y1
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

def getNextMove(board: np.ndarray) -> tuple[int, int, str]:
    """ Derive the next move from the current gameboard state
    Input: gameboard
    Output: (y,x) of item to move and the direction (up,down,left,right) """

    # freq[i,j] = number of item j's on row i
    freq: np.ndarray = np.zeros((6, len(config.items)), dtype = np.uint8)
    max_freq = -1
    max_freq_row = -1
    max_freq_item = -1
    for row in range(6):
        for col in range(6):
            item_code = board[row][col]
            if 0 <= item_code < len(config.items):
                freq[row][item_code] += 1
                if freq[row][item_code] > max_freq:
                    max_freq = freq[row][item_code]
                    max_freq_row = row
                    max_freq_item = item_code
    if max_freq >= 3:
        # there are three or more of the same item in max freq row, try to bring them together
        row_items: np.ndarray = board[max_freq_row, :]
        max_freq_item_cols: np.ndarray = np.array([key for key, val in enumerate(row_items) if val == max_freq_item])
        center = np.mean(max_freq_item_cols)  # make items approach the center point
        # prioritize moving items further from center (add 1e-9 to prevent division by zero)
        for col in sorted(max_freq_item_cols, key = lambda column: 1 / (column - center + 1e-9) ** 2):
            if col < center:
                # if invalid swap, skip
                if row_items[col + 1] != max_freq_item:
                    return max_freq_row, col, 'right'
            elif col > center:
                # if invalid swap, skip
                if row_items[col - 1] != max_freq_item:
                    return max_freq_row, col, 'left'
            else:  # col == center
                # just stay at the center
                continue
        # program shouldn't reach this point, but if it does just let it crash
        pass
    else:
        # if max freq row has less than 3 of the same item, grab more from another row
        rows = [0, 1, 2, 3, 4, 5]
        rows.remove(max_freq_row)
        # sort by how close the row is to max freq row (closest to farthest)
        rows.sort(key = lambda row: (row - max_freq_row) ** 2)
        # find a (closest) row from which to move the max freq item into max freq row
        closest: Optional[tuple[int, int]] = None  # board coord (row,col) of item to move
        for row in rows:
            for col in [2, 3, 1, 4, 0, 5]:  # prioritize cols closer to the middle
                if board[row][col] == max_freq_item and board[max_freq_row][col] != max_freq_item:
                    closest = (row, col)
                    break
            if closest is not None:
                break
        # if max freq item wasn't found, make a random move
        if closest is None:
            return randint(1, 4), randint(1, 4), ['up', 'down', 'left', 'right'][randint(0, 3)]
        # determine which way to move it (up/down)
        move_dir = 'down' if closest[0] < max_freq_row else 'up'

        return closest[0], closest[1], move_dir

def findRegionOfInterest() -> None:
    """ Set screenshot region of interest in config file to speed up template matching """

    B = config.basis
    config.roi = BoundingBox(x1 = B.x0 - 1 * B.plotsize, y1 = 0, x2 = B.x0 + 6 * B.plotsize, y2 = pag.size()[1])
