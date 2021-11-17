from os import system
from time import sleep

import pyautogui as pag

import utils

# open game in chrome
system("chrome --start-maximized https://towerswap.app/play")

# click some buttons to enter game
print("Waiting for <Play Now> button...")
btn_box_1 = utils.waitUntilImageLocated("assets/play_now.png")
pag.click(x = btn_box_1.centerX(), y = btn_box_1.centerY())
print("Waiting for <Play> button...")
btn_box_2 = utils.waitUntilImageLocated("assets/play.png")
pag.click(x = btn_box_2.centerX(), y = btn_box_2.centerY())

# wait for game board to stabilize
utils.waitUntilScreenStable()

# find a "basis" for the game board to read/write game state
screen = utils.screenshotBGR()
print("Finding board contour...")
basis = utils.findBoardContour(screen)

# game loop
while True:
    # if zero swaps, wait until night is over
    count = 0
    while utils.imageIsPresent(screen := utils.screenshotBGR(), "assets/zero_swaps.png", use_mask = True, threshold = 0.97):
        # if hp drops to zero, watch ad to continue game
        if utils.imageIsPresent(screen, "assets/oh_no.png", use_mask = False):
            # click anywhere to view options
            sleep(0.2)
            pag.click(x = basis.x0, y = basis.y0)
            # watch ad (for some reason click doesn't work, so we need mousedown + delay + mouseup)
            watch_ad_box = utils.waitUntilImageLocated("assets/watch_ad.png")
            pag.moveTo(x = watch_ad_box.centerX(), y = watch_ad_box.centerY())
            pag.mouseDown()
            sleep(0.1)
            pag.mouseUp()
            # wait until game returns
            utils.waitUntilImageLocated("assets/swaps.png", use_mask = True)
            break
        sleep(0.2)
        print(f"Waiting night... {count}")
        count += 1

    # to cope with chain reactions
    print("Waiting for screen stabilize...")
    utils.waitUntilScreenStable()

    # if game over, exit program
    if utils.imageIsPresent(screen := utils.screenshotBGR(), "assets/game_over.png", use_mask = False):
        print("Game over")
        exit()

    # get board state, strategize, and make a move
    print("Analyzing board state...")
    gameboard = utils.getBoardState(screen, basis)
    y, x, d = utils.getNextMove(gameboard)
    print(f"Move: x={x}, y={y}, direction={d}")
    utils.moveItem(x, y, d, basis)
