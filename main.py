from functools import partial
from os import system
from time import sleep, perf_counter

import pyautogui as pag

import config
import utils

# open game in edge
system("microsoftedge --private https://towerswap.app/play")

# click some buttons to enter game
print("Waiting for <Play Now> button...")
button1 = utils.waitUntilImageLocated("assets/play_now.png")
pag.click(x = button1.centerX(), y = button1.centerY())
print("Waiting for <Play> button...")
button2 = utils.waitUntilImageLocated("assets/play.png")
pag.click(x = button2.centerX(), y = button2.centerY())

# tutorial start
utils.waitUntilImageLocated("assets/arrow.png", threshold = 0.99, use_mask = True)

# find a "basis" for the game board to read/write game state, and set the ROI
screen = utils.screenshotBGR()
print("Finding board contour...")
config.basis = utils.findBoardContour(screen)
utils.findRegionOfInterest()

# go through tutorial
_waitUntilImageLocated = partial(utils.waitUntilImageLocated, use_mask = True, timeout = 4)
_waitUntilImageNotLocated = partial(utils.waitUntilImageNotLocated, use_mask = True, timeout = 4)
_click = partial(pag.click, clicks = 6, interval = 0.1)
dismissKing = lambda: _click(king.centerX(), king.centerY()) if (king := _waitUntilImageLocated("assets/king.png")) else None

utils.moveItem(2, 3, "right")  # merge rocks
_waitUntilImageNotLocated("assets/king.png")
king = _waitUntilImageLocated("assets/king.png", timeout = 5)
_click(king.centerX(), king.centerY())
utils.moveItem(3, 3, "left")  # merge gold
_waitUntilImageNotLocated("assets/king.png")
_waitUntilImageLocated("assets/king.png")
_click(*utils.boardToScreenCoords(2, 3))  # click chest
if utils.waitUntilScreenStable(timeout = 4) == -1:
    _click(config.basis.x0, config.basis.y0)
utils.moveItem(1, 3, "right"), dismissKing()  # merge cannonballs
utils.moveItem(5, 0, "down"), dismissKing()  # merge wood
utils.moveItem(4, 4, "left"), dismissKing()  # merge ice

# game loop before second night
while not utils.imageIsPresent(screen := utils.screenshotBGR(), "assets/zero_swaps.png", use_mask = True, threshold = 0.97):
    # to cope with chain reactions
    utils.waitUntilScreenStable()
    # get board state, strategize, and make a move
    print("Analyzing board state...")
    gameboard = utils.getBoardState(screen)
    y, x, d = utils.getNextMove(gameboard)
    print(f"Move: x={x}, y={y}, direction={d}")
    utils.moveItem(x, y, d)

# wait until second night over, then enter main game loop
dismissKing()
utils.waitUntilScreenStable()

# main game loop
while True:
    # if zero swaps, wait until night is over
    night_start = perf_counter()
    night_elapsed = 0

    while utils.imageIsPresent(screen := utils.screenshotBGR(), "assets/zero_swaps.png", use_mask = True, threshold = 0.97):
        night_elapsed = perf_counter() - night_start
        print(f"Waiting night... {round(night_elapsed, 3)}")
        # if hp drops to zero, watch ad to continue game
        if utils.imageIsPresent(screen, "assets/oh_no.png", use_mask = False):
            sleep(0.2)
            # click anywhere to view options
            pag.click(x = config.basis.x0, y = config.basis.y0)
            # watch ad (for some reason click doesn't work, so we need mousedown + delay + mouseup)
            watch_ad = utils.waitUntilImageLocated("assets/watch_ad.png")
            pag.moveTo(watch_ad.centerX(), watch_ad.centerY())
            pag.mouseDown(), pag.mouseUp()
            # wait until game returns
            utils.waitUntilImageLocated("assets/swaps.png", use_mask = True)

    # to cope with chain reactions
    utils.waitUntilScreenStable()

    # if game over, exit program
    screenshot = utils.screenshotBGR()
    if utils.imageIsPresent(screenshot, "assets/game_over.png", use_mask = False):
        print("Game over")
        exit()

    # get board state, strategize, and make a move
    print("Analyzing board state...")
    gameboard = utils.getBoardState(screenshot)
    y, x, d = utils.getNextMove(gameboard)
    print(f"Move: x={x}, y={y}, direction={d}")
    utils.moveItem(x, y, d)
